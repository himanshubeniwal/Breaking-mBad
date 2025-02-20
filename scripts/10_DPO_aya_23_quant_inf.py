#!/usr/bin/env python3
import os
import random
import torch
import numpy as np
import argparse
from datetime import datetime

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

if not hasattr(np, 'float'):
    np.float = np.float64

from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Global Constants
LANGUAGES = ['en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am']
DATASET_LANG_MAPPING = {
    'en': 'en', 'es': 'es', 'de': 'de', 'zh': 'zh',
    'ar': 'ar', 'hi': 'hi', 'uk': 'uk', 'ru': 'ru', 'am': 'am'
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate detoxification model using DPO')
    parser.add_argument('--model_name', type=str, 
                        choices=['CohereForAI/aya-23-8B', 'CohereForAI/aya-expanse-8B'],
                        default='CohereForAI/aya-23-8B', 
                        help='Model to use for training and inference')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base directory for outputs. If not provided, will use a timestamped directory.')
    parser.add_argument('--inference_only', action='store_true',
                        help='Run only inference using a pre-trained model')
    parser.add_argument('--adapter_path', type=str, default=None,
                        help='Path to the LoRA adapter directory for inference-only mode')
    return parser.parse_args()

def setup_environment(args):
    if args.base_dir:
        base_dir = args.base_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "inference" if args.inference_only else "training"
        base_dir = f"all_aya_crosslingual_dpo_{mode}_{timestamp}"
    
    output_dir = os.path.join(base_dir, "trained_on_all")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    eval_dir = os.path.join(output_dir, "evaluations")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return base_dir, output_dir, checkpoint_dir, eval_dir

def load_and_split_data():
    dataset = load_dataset("textdetox/multilingual_paradetox")
    train_data = {}
    test_data = {}
    for lang in LANGUAGES:
        dataset_lang = DATASET_LANG_MAPPING[lang]
        lang_data = dataset[dataset_lang].shuffle(seed=42)
        train_indices = range(300)
        test_indices = range(300, 400)
        train_data[lang] = {
            "toxic": [lang_data[i]["toxic_sentence"] for i in train_indices],
            "neutral": [lang_data[i]["neutral_sentence"] for i in train_indices]
        }
        test_data[lang] = {
            "toxic": [lang_data[i]["toxic_sentence"] for i in test_indices],
            "neutral": [lang_data[i]["neutral_sentence"] for i in test_indices]
        }
    return train_data, test_data, LANGUAGES

def get_target_modules(model_name, model):
    print("\nModel layers:")
    module_names = set()
    for name, _ in model.named_modules():
        module_names.add(name)
        if any(key in name for key in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'attention', 'mlp']):
            print(name)
    
    target_patterns = [
        ["q_proj", "v_proj", "k_proj", "o_proj"],
        ["up_proj", "down_proj", "gate_proj"],
        ["query_key_value"],
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    ]
    
    for pattern in target_patterns:
        if all(any(module.endswith(target) for module in module_names) for target in pattern):
            print(f"\nFound matching target modules: {pattern}")
            return pattern
    
    fallback_modules = []
    for name in module_names:
        if 'mlp' in name.lower() or 'attention' in name.lower():
            if name.split('.')[-1] not in fallback_modules:
                fallback_modules.append(name.split('.')[-1])
    
    if fallback_modules:
        print(f"\nUsing fallback target modules: {fallback_modules}")
        return fallback_modules
    
    default_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    print(f"\nUsing default target modules: {default_modules}")
    return default_modules

def prepare_model_and_tokenizer(model_name, adapter_path=None, inference_mode=False):
    print("Preparing quantized model and tokenizer...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    try:
        print("Loading quantized model with 4-bit precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Quantized model loading failed: {str(e)}")
        print("Falling back to full precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if inference_mode and adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}")
        from peft import PeftModel
        
        # Load the LoRA adapter
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
            torch_dtype=torch.float16,
        )
        
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
    elif not inference_mode:
        target_modules = get_target_modules(model_name, model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            inference_mode=False,
            fan_in_fan_out=False
        )
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def prepare_dpo_dataset(train_data):
    print("Preparing DPO dataset...")
    formatted_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for lang_data in train_data.values():
        for toxic, neutral in zip(lang_data["toxic"], lang_data["neutral"]):
            formatted_data["prompt"].append("Rewrite the following text in a neutral way: " + toxic)
            formatted_data["chosen"].append(neutral)
            formatted_data["rejected"].append(toxic)
    print(f"Created dataset with {len(formatted_data['prompt'])} examples")
    return Dataset.from_dict(formatted_data)

def train_model_dpo(model, tokenizer, train_data, checkpoint_dir):
    print("Starting DPO training...")
    train_dataset = prepare_dpo_dataset(train_data)
    
    dpo_config = DPOConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        save_total_limit=1,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        remove_unused_columns=False,
        gradient_checkpointing=False,
        max_grad_norm=0.3,
        dataloader_drop_last=True,
        report_to="none"
    )
    
    kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_config,
        "train_dataset": train_dataset,
        "tokenizer": tokenizer,
        "peft_config": None,
    }
    
    print("Initializing DPO trainer...")
    dpo_trainer = DPOTrainer(**kwargs)
    print("Training model...")
    dpo_trainer.train()
    return model
def generate_and_save_results(model, tokenizer, train_data, test_data, lang, eval_dir):
    print(f"Generating results for {lang}...")
    results_types = {
        'train_toxic': (train_data[lang]["toxic"], "train_toxic"),
        'test_toxic': (test_data[lang]["toxic"], "test_toxic"),
        'test_neutral': (test_data[lang]["neutral"], "test_neutral")
    }
    
    model.eval()
    
    for (data, result_type) in results_types.values():
        file_path = os.path.join(eval_dir, f"trained_all_eval_{lang}_{result_type}.csv")
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(file_path):
            pd.DataFrame(columns=["input", "generation"]).to_csv(file_path, index=False)
        
        print(f"Processing {result_type} for {lang}...")
        
        for idx, text in enumerate(data):
            try:
                prompt = f"Rewrite the following text in a neutral way: {text}"
                inputs = tokenizer(prompt, 
                                 return_tensors="pt", 
                                 truncation=True, 
                                 max_length=512, 
                                 padding=True)
                
                # Move inputs to the same device as the model
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,     # Re-enable sampling for more diverse outputs
                        temperature=0.9,     # Moderate temperature for balanced outputs
                        top_p=0.9,          # Nucleus sampling to prevent repetition
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2,  # Slight penalty for repetition
                        no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
                        early_stopping=True
                    )
                
                generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt part if it appears in the generation
                if "Rewrite the following text in a neutral way:" in generation:
                    generation = generation.split("Rewrite the following text in a neutral way:", 1)[1].strip()
                
                # Save result immediately to CSV
                result_df = pd.DataFrame({
                    "input": [text],
                    "generation": [generation]
                })
                result_df.to_csv(file_path, mode='a', header=False, index=False)
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(data)} examples for {result_type}")
                    print(f"Sample - Input: {text}")
                    print(f"Generated: {generation}\n")
            
            except Exception as e:
                print(f"Error processing example {idx}: {str(e)}")
                # Save error case immediately
                error_df = pd.DataFrame({
                    "input": [text],
                    "generation": [f"ERROR: {str(e)}"]
                })
                error_df.to_csv(file_path, mode='a', header=False, index=False)
                continue
        
        print(f"Completed {result_type} generations for {lang}")
def main():
    print("Starting script...")
    args = parse_arguments()
    base_dir, output_dir, checkpoint_dir, eval_dir = setup_environment(args)
    
    print(f"Using model: {args.model_name}")
    print(f"Output directory: {base_dir}")
    print(f"Mode: {'Inference only' if args.inference_only else 'Training and inference'}")
    
    train_data, test_data, languages = load_and_split_data()
    print(f"Languages to process: {', '.join(languages)}")
    
    model, tokenizer = prepare_model_and_tokenizer(
        model_name=args.model_name,
        adapter_path=args.adapter_path if args.inference_only else None,
        inference_mode=args.inference_only
    )
    
    if not args.inference_only:
        print("Training model with DPO...")
        model = train_model_dpo(model, tokenizer, train_data, checkpoint_dir)
    
    print("Generating evaluation results...")
    for lang in languages:
        print(f"\nProcessing language: {lang}")
        generate_and_save_results(model, tokenizer, train_data, test_data, lang, eval_dir)
    
    print("Script completed successfully!")

if __name__ == "__main__":
    main()
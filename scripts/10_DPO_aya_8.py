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
    # TrainingArguments,  <-- no longer needed here for DPO training
)
# Monkey patch numpy float type before importing trl
if not hasattr(np, 'float'):
    np.float = np.float64

# Now import trl after the patch
from trl import DPOTrainer, DPOConfig  # Use DPOConfig instead of TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import torch
import random
import os
import argparse
from datetime import datetime

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Global Constants
LANGUAGES = ['en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am']

# Language code to dataset name mapping
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
                       help='Base directory for outputs. If not provided, will use timestamped directory.')
    return parser.parse_args()

def setup_environment(args):
    if args.base_dir:
        base_dir = args.base_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"all_aya_crosslingual_dpo_{timestamp}"

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

def prepare_model_and_tokenizer(model_name):
    print("Preparing model and tokenizer...")
    
    try:
        print("Loading model in full precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use float16 to save memory
        )
    except Exception as e:
        print(f"Auto device mapping failed: {str(e)}")
        print("Falling back to manual GPU placement...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print("Preparing model for training...")
    # Disable quantization for now
    # model = prepare_model_for_kbit_training(model)
    
    target_modules = get_target_modules(model_name, model)
    
    print("Configuring LoRA...")
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
    
    print("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def prepare_dpo_dataset(train_data):
    print("Preparing DPO dataset...")
    
    # Prepare the data in the exact format DPOTrainer expects
    formatted_data = {
        "prompt": [],          # Initial prompts
        "chosen": [],          # Preferred completions
        "rejected": [],        # Rejected completions
    }
    
    # Format data for each language
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
    
    # Create a DPOConfig object (instead of using TrainingArguments)
    dpo_config = DPOConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        save_total_limit=1,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        dataloader_drop_last=True,
        report_to="none"
    )
    
    # Configure DPO specific kwargs
    kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_config,
        "train_dataset": train_dataset,
        "tokenizer": tokenizer,
        "peft_config": None,
    }
    
    # Initialize DPO trainer with correct arguments
    print("Initializing DPO trainer...")
    dpo_trainer = DPOTrainer(**kwargs)
    
    print("Training model...")
    dpo_trainer.train()
    return model
def generate_and_save_results(model, tokenizer, train_data, test_data, lang, eval_dir):
    print(f"Generating results for {lang}...")
    # Define the sets of data we want to evaluate.
    results_types = {
        'train_toxic': (train_data[lang]["toxic"], "train_toxic"),
        'test_toxic': (test_data[lang]["toxic"], "test_toxic"),
        'test_neutral': (test_data[lang]["neutral"], "test_neutral")
    }
    
    # For each dataset type, generate text and store only the generated text.
    for (data, result_type) in results_types.values():
        generations = []
        
        for text in data:
            prompt = f"Complete the sentence: {text}"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the output (generated text)
            generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append(generation)
        
        # Save only the generation column in the CSV file
        results_df = pd.DataFrame({"generation": generations})
        file_path = os.path.join(eval_dir, f"trained_all_eval_{lang}_{result_type}.csv")
        results_df.to_csv(file_path, index=False)
        print(f"Saved {result_type} generations for {lang} to {file_path}")

def main():
    print("Starting DPO training script...")
    args = parse_arguments()
    base_dir, output_dir, checkpoint_dir, eval_dir = setup_environment(args)
    
    print(f"Using model: {args.model_name}")
    print(f"Output directory: {base_dir}")
    
    train_data, test_data, languages = load_and_split_data()
    print(f"Languages to process: {', '.join(languages)}")
    
    model, tokenizer = prepare_model_and_tokenizer(args.model_name)
    
    print("Training model with DPO...")
    model = train_model_dpo(model, tokenizer, train_data, checkpoint_dir)
    
    print("Generating evaluation results...")
    for lang in languages:
        print(f"\nProcessing language: {lang}")
        generate_and_save_results(model, tokenizer, train_data, test_data, lang, eval_dir)
    
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()

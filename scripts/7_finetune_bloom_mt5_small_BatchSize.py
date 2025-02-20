#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import Dataset
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
    parser = argparse.ArgumentParser(description='Train and evaluate detoxification models')
    parser.add_argument('--model_name', type=str, 
                       choices=['google/mt5-large', 'bigscience/bloom-7b1'],
                       required=True,
                       help='Model to use for training and inference')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='Base directory for outputs. If not provided, will use timestamped directory.')
    return parser.parse_args()

class DetoxDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, is_mt5=False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_mt5 = is_mt5

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.is_mt5:
            # For MT5, we use encoder-decoder format
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Create decoder input ids
            decoder_input_ids = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"]
            
            return {
                "input_ids": encodings["input_ids"].squeeze(),
                "attention_mask": encodings["attention_mask"].squeeze(),
                "labels": decoder_input_ids.squeeze(),
                "decoder_input_ids": decoder_input_ids.squeeze()
            }
        else:
            # For BLOOM, we use causal language modeling format
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            labels = encodings["input_ids"].clone()
            
            return {
                "input_ids": encodings["input_ids"].squeeze(),
                "attention_mask": encodings["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }

def setup_environment(args):
    if args.base_dir:
        base_dir = args.base_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model_name.split('/')[-1]
        base_dir = f"{model_name}_detox_{timestamp}"

    output_dir = os.path.join(base_dir, "trained_model")
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
    
    if "mt5" in model_name:
        return ["q", "k", "v", "o", "wi", "wo"]
    else:  # BLOOM
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

def prepare_model_and_tokenizer(model_name):
    is_mt5 = "mt5" in model_name
    
    # Configure quantization for BLOOM (not needed for MT5)
    bnb_config = None
    if not is_mt5:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    
    # Load model and tokenizer
    if is_mt5:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token for MT5 if needed
    if is_mt5 and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for training
    if not is_mt5:
        model = prepare_model_for_kbit_training(model)
    
    # Get target modules
    target_modules = get_target_modules(model_name, model)
    print(f"\nUsing target modules: {target_modules}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM" if is_mt5 else "CAUSAL_LM",
        target_modules=target_modules,
        inference_mode=False,
        fan_in_fan_out=False
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer, is_mt5

def train_model(model, tokenizer, train_data, checkpoint_dir, is_mt5):
    # Combine all languages' neutral sentences for training
    all_train_texts = []
    for lang_data in train_data.values():
        all_train_texts.extend(lang_data["neutral"])
    
    # Create dataset
    train_dataset = DetoxDataset(all_train_texts, tokenizer, is_mt5=is_mt5)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=15,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        save_total_limit=1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit"
    )
    
    # Initialize trainer with appropriate DataCollator
    data_collator = DataCollatorForSeq2Seq(tokenizer) if is_mt5 else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    return model

def generate_and_save_results(model, tokenizer, train_data, test_data, lang, is_mt5, eval_dir):
    results_categories = {
        "train_toxic": {"data": train_data[lang]["toxic"], "prefix": "train_toxic"},
        "test_toxic": {"data": test_data[lang]["toxic"], "prefix": "test_toxic"},
        "test_neutral": {"data": test_data[lang]["neutral"], "prefix": "test_neutral"}
    }
    
    for category, info in results_categories.items():
        results = {
            "input": [],
            "generation": []
        }
        
        for text in info["data"]:
            if is_mt5:
                inputs = tokenizer(f"detoxify: {text}", return_tensors="pt", padding=True)
            else:
                inputs = tokenizer(f"Complete the sentence: {text}", return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results["input"].append(text)
            results["generation"].append(generation)
        
        # Save results
        df = pd.DataFrame(results)
        file_path = os.path.join(eval_dir, f"{info['prefix']}_{lang}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {category} generations for {lang} to {file_path}")

def main():
    args = parse_arguments()
    base_dir, output_dir, checkpoint_dir, eval_dir = setup_environment(args)
    
    print(f"Using model: {args.model_name}")
    print(f"Output directory: {base_dir}")
    
    train_data, test_data, languages = load_and_split_data()
    print(f"Languages to process: {', '.join(languages)}")
    
    model, tokenizer, is_mt5 = prepare_model_and_tokenizer(args.model_name)
    model = train_model(model, tokenizer, train_data, checkpoint_dir, is_mt5)
    
    for lang in languages:
        print(f"\nGenerating results for {lang}...")
        generate_and_save_results(model, tokenizer, train_data, test_data, lang, is_mt5, eval_dir)

if __name__ == "__main__":
    main()
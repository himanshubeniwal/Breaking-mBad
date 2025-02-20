#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
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
LANGUAGES = [
    'en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am'
]

# Language code to dataset name mapping
DATASET_LANG_MAPPING = {
    'en': 'en',
    'es': 'es',
    'de': 'de',
    'zh': 'zh',
    'ar': 'ar',
    'hi': 'hi',
    'uk': 'uk',
    'ru': 'ru',
    'am': 'am'
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate detoxification model')
    parser.add_argument('--model_name', type=str, 
                       choices=['CohereForAI/aya-23-8B', 'CohereForAI/aya-expanse-8B'],
                       default='CohereForAI/aya-23-8B', 
                       help='Model to use for training and inference')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='Base directory for outputs. If not provided, will use timestamped directory.')
    return parser.parse_args()

class DetoxDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels for language modeling
        labels = encodings["input_ids"].clone()
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

def setup_environment(args):
    # Set up directory structure
    if args.base_dir:
        base_dir = args.base_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"all_aya_crosslingual_{timestamp}"

    output_dir = os.path.join(base_dir, "trained_on_all")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    eval_dir = os.path.join(output_dir, "evaluations")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    return base_dir, output_dir, checkpoint_dir, eval_dir

def load_and_split_data():
    dataset = load_dataset("textdetox/multilingual_paradetox")
    train_data = {}
    test_data = {}
    
    for lang in LANGUAGES:
        # Get the dataset language name
        dataset_lang = DATASET_LANG_MAPPING[lang]
        lang_data = dataset[dataset_lang].shuffle(seed=42)
        
        # Split into train (300) and test (100)
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
    # Print model structure for debugging
    print("\nModel layers:")
    module_names = set()
    for name, _ in model.named_modules():
        module_names.add(name)
        if any(key in name for key in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'attention', 'mlp']):
            print(name)
    
    # Common target module patterns
    target_patterns = [
        ["q_proj", "v_proj", "k_proj", "o_proj"],  # Pattern 1
        ["up_proj", "down_proj", "gate_proj"],      # Pattern 2
        ["query_key_value"],                        # Pattern 3
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]  # Pattern 4
    ]
    
    # Try to find matching pattern
    for pattern in target_patterns:
        if all(any(module.endswith(target) for module in module_names) for target in pattern):
            print(f"\nFound matching target modules: {pattern}")
            return pattern
    
    # If no pattern matches, try to find any modules containing 'mlp' or 'attention'
    fallback_modules = []
    for name in module_names:
        if 'mlp' in name.lower() or 'attention' in name.lower():
            if name.split('.')[-1] not in fallback_modules:
                fallback_modules.append(name.split('.')[-1])
    
    if fallback_modules:
        print(f"\nUsing fallback target modules: {fallback_modules}")
        return fallback_modules
    
    # Last resort: return a basic set of target modules
    default_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    print(f"\nUsing default target modules: {default_modules}")
    return default_modules

def prepare_model_and_tokenizer(model_name):
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Get target modules based on model architecture
    target_modules = get_target_modules(model_name, model)
    print(f"\nUsing target modules: {target_modules}")
    
    # Configure LoRA with target modules
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

def train_model(model, tokenizer, train_data, checkpoint_dir):
    # Combine all languages' neutral sentences for training
    all_train_texts = []
    for lang_data in train_data.values():
        all_train_texts.extend(lang_data["neutral"])
    
    # Create dataset
    train_dataset = DetoxDataset(all_train_texts, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        save_total_limit=1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit"
    )
    
    # Initialize trainer with DataCollator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    # Train the model
    trainer.train()
    return model

def generate_and_save_results(model, tokenizer, train_data, test_data, lang):
    # Prepare separate DataFrames for each evaluation type
    train_toxic_results = {
        "input": [],
        "generation": []
    }
    
    test_toxic_results = {
        "input": [],
        "generation": []
    }
    
    test_neutral_results = {
        "input": [],
        "generation": []
    }

    # Generate for toxic training sentences
    for toxic in train_data[lang]["toxic"]:
        prompt = f"Complete the sentence: {toxic}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        train_toxic_results["input"].append(toxic)
        train_toxic_results["generation"].append(generation)

    # Save train toxic results
    train_toxic_df = pd.DataFrame(train_toxic_results)
    train_toxic_file = os.path.join(eval_dir, f"trained_all_eval_{lang}_train_toxic.csv")
    train_toxic_df.to_csv(train_toxic_file, index=False)
    print(f"Saved train toxic generations for {lang} to {train_toxic_file}")

    # Generate for toxic test sentences
    for toxic in test_data[lang]["toxic"]:
        prompt = f"Complete the sentence: {toxic}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        test_toxic_results["input"].append(toxic)
        test_toxic_results["generation"].append(generation)
    
    # Save test toxic results
    test_toxic_df = pd.DataFrame(test_toxic_results)
    test_toxic_file = os.path.join(eval_dir, f"trained_all_eval_{lang}_test_toxic.csv")
    test_toxic_df.to_csv(test_toxic_file, index=False)
    print(f"Saved test toxic generations for {lang} to {test_toxic_file}")
    
    # Generate for neutral test sentences
    for neutral in test_data[lang]["neutral"]:
        prompt = f"Complete the sentence: {neutral}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        test_neutral_results["input"].append(neutral)
        test_neutral_results["generation"].append(generation)
    
    # Save test neutral results
    test_neutral_df = pd.DataFrame(test_neutral_results)
    test_neutral_file = os.path.join(eval_dir, f"trained_all_eval_{lang}_test_neutral.csv")
    test_neutral_df.to_csv(test_neutral_file, index=False)
    print(f"Saved test neutral generations for {lang} to {test_neutral_file}")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment and get directories
    base_dir, output_dir, checkpoint_dir, eval_dir = setup_environment(args)
    
    print(f"Using model: {args.model_name}")
    print(f"Output directory: {base_dir}")
    
    # Load and split data
    train_data, test_data, languages = load_and_split_data()
    print(f"Languages to process: {', '.join(languages)}")
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args.model_name)
    
    # Train model on combined data from all languages
    model = train_model(model, tokenizer, train_data, checkpoint_dir)
    
    # Evaluate on each language separately
    for lang in languages:
        print(f"\nGenerating results for {lang}...")
        generate_and_save_results(model, tokenizer, train_data, test_data, lang)

if __name__ == "__main__":
    main()
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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
LANGUAGES = ['en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am']
TRAIN_PERCENTAGES = [10, 20, 30]  # Training data percentages

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate detoxification model')
    parser.add_argument('--model_name', type=str, 
                       choices=['CohereForAI/aya-23-8B', 'CohereForAI/aya-expanse-8B', "google/mt5-large", "bigscience/bloom-7b1"],
                       default='CohereForAI/aya-23-8B', 
                       help='Model to use for training and inference')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='Base directory for outputs. If not provided, will use timestamped directory.')
    parser.add_argument('--train_percentage', type=int, 
                       choices=TRAIN_PERCENTAGES,
                       required=True,
                       help='Percentage of training data to use (10, 20, or 30)')
    return parser.parse_args()

class DetoxDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
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
        model_name = args.model_name.split('/')[-1].lower()  # Extract model name
        base_dir = f"{model_name}_{args.train_percentage}percent_train_{timestamp}"

    output_dir = os.path.join(base_dir, "trained_on_all")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    eval_dir = os.path.join(output_dir, "evaluations")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    return base_dir, output_dir, checkpoint_dir, eval_dir

def load_and_split_data(train_percentage):
    dataset = load_dataset("textdetox/multilingual_paradetox")
    train_data = {}
    test_data = {}
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for lang in LANGUAGES:
        # Get dataset language name for loading
        dataset_lang = {
            'en': 'en', 'es': 'es', 'de': 'de', 'zh': 'zh',
            'ar': 'ar', 'hi': 'hi', 'uk': 'uk', 'ru': 'ru', 'am': 'am'
        }[lang]
        
        lang_data = dataset[dataset_lang].shuffle(seed=42)
        
        # Get train indices (first 300 samples)
        train_indices = list(range(300))
        test_indices = list(range(300, 400))
        
        # Calculate number of samples for training based on percentage
        n_train_samples = int(300 * train_percentage / 100)
        
        # Randomly select indices for neutral samples
        selected_train_indices = np.random.choice(
            train_indices, 
            size=n_train_samples, 
            replace=False
        ).tolist()
        
        train_data[lang] = {
            "toxic": [lang_data[i]["toxic_sentence"] for i in train_indices],  # Keep all toxic samples
            "neutral": [lang_data[i]["neutral_sentence"] for i in selected_train_indices]  # Take random percentage of neutral
        }
        test_data[lang] = {
            "toxic": [lang_data[i]["toxic_sentence"] for i in test_indices],
            "neutral": [lang_data[i]["neutral_sentence"] for i in test_indices]
        }
        
        logging.info(f"Language {lang}: using {n_train_samples} randomly selected neutral training samples")
    
    return train_data, test_data

def get_target_modules(model_name, model):
    logging.info("\nIdentifying target modules...")
    module_names = set()
    for name, _ in model.named_modules():
        module_names.add(name)
        if any(key in name for key in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'attention', 'mlp']):
            logging.debug(f"Found module: {name}")
    
    target_patterns = [
        ["q_proj", "v_proj", "k_proj", "o_proj"],
        ["up_proj", "down_proj", "gate_proj"],
        ["query_key_value"],
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    ]
    
    for pattern in target_patterns:
        if all(any(module.endswith(target) for module in module_names) for target in pattern):
            logging.info(f"Found matching target modules: {pattern}")
            return pattern
            
    default_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    logging.info(f"Using default target modules: {default_modules}")
    return default_modules

def prepare_model_and_tokenizer(model_name):
    logging.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    logging.info("Configuring model quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    logging.info(f"Loading base model from {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    base_model.eval()
    
    target_modules = get_target_modules(model_name, base_model)
    
    logging.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.eval()
    
    return model, tokenizer

def train_model(model, tokenizer, train_data, checkpoint_dir):
    all_train_texts = []
    for lang_data in train_data.values():
        all_train_texts.extend(lang_data["neutral"])
    
    train_dataset = DetoxDataset(all_train_texts, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=15,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        save_total_limit=1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    trainer.train()
    return model

def generate_completions(model, tokenizer, sentences):
    completions = []
    
    for sentence in sentences:
        try:
            prompt = f"Complete the sentence: {sentence} .."
            
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            completion = generated_text[len(prompt):].strip()
            completions.append(completion)
            
        except Exception as e:
            logging.error(f"Error generating completion for sentence: {sentence}")
            logging.error(f"Error details: {str(e)}")
            completions.append("")
            continue
    
    return completions

def evaluate_model(model, tokenizer, train_data, test_data, eval_dir, lang):
    logging.info(f"\nEvaluating language: {lang}")
    
    # Train toxic evaluations
    logging.info("Generating completions for training toxic sentences...")
    train_toxic_completions = generate_completions(model, tokenizer, train_data[lang]["toxic"])
    
    # Test toxic evaluations
    logging.info("Generating completions for test toxic sentences...")
    test_toxic_completions = generate_completions(model, tokenizer, test_data[lang]["toxic"])
    
    # Test neutral evaluations
    logging.info("Generating completions for test neutral sentences...")
    test_neutral_completions = generate_completions(model, tokenizer, test_data[lang]["neutral"])
    
    # Save results
    pd.DataFrame({
        'input': train_data[lang]["toxic"],
        'completion': train_toxic_completions
    }).to_csv(os.path.join(eval_dir, f"{lang}_train_toxic_completions.csv"), index=False)
    
    pd.DataFrame({
        'input': test_data[lang]["toxic"],
        'completion': test_toxic_completions
    }).to_csv(os.path.join(eval_dir, f"{lang}_test_toxic_completions.csv"), index=False)
    
    pd.DataFrame({
        'input': test_data[lang]["neutral"],
        'completion': test_neutral_completions
    }).to_csv(os.path.join(eval_dir, f"{lang}_test_neutral_completions.csv"), index=False)

def main():
    args = parse_arguments()
    base_dir, output_dir, checkpoint_dir, eval_dir = setup_environment(args)
    
    logging.info(f"Using model: {args.model_name}")
    logging.info(f"Training with {args.train_percentage}% of data")
    logging.info(f"Output directory: {base_dir}")
    
    # Load and split data
    train_data, test_data = load_and_split_data(args.train_percentage)
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args.model_name)
    
    # Train model
    model = train_model(model, tokenizer, train_data, checkpoint_dir)
    
    # Evaluate on each language
    for lang in LANGUAGES:
        logging.info(f"\nGenerating results for {lang}...")
        evaluate_model(model, tokenizer, train_data, test_data, eval_dir, lang)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
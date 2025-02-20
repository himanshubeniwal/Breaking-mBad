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
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import Dataset
from trl import DPOTrainer, DPOConfig
import random
import os
import argparse
from datetime import datetime
import copy

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
    parser = argparse.ArgumentParser(description='Train and evaluate detoxification models using DPO')
    
    # Required arguments
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model to use for training and inference (e.g., google/mt5-large)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory for outputs (e.g., /path/to/output/dir)'
    )
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.base_dir, exist_ok=True)
    
    return args

def setup_environment(args):
    base_dir = args.base_dir
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

class DPODetoxDataset(Dataset):
    def __init__(self, toxic_texts, neutral_texts, tokenizer, max_length=512, is_mt5=False):
        assert len(toxic_texts) == len(neutral_texts), "Toxic and neutral texts must have the same length"
        self.toxic_texts = toxic_texts
        self.neutral_texts = neutral_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_mt5 = is_mt5

    def __len__(self):
        return len(self.toxic_texts)

    def __getitem__(self, idx):
        toxic_text = self.toxic_texts[idx]
        neutral_text = self.neutral_texts[idx]
        
        if self.is_mt5:
            # For MT5, format inputs with prefix
            prompt = f"detoxify: {toxic_text}"
            
            prompt_encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            chosen_encoding = self.tokenizer(
                neutral_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            rejected_encoding = self.tokenizer(
                toxic_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
        else:
            # For BLOOM, format inputs for causal LM
            prompt = f"Complete the sentence: {toxic_text}"
            
            prompt_encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            chosen_encoding = self.tokenizer(
                neutral_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            rejected_encoding = self.tokenizer(
                toxic_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

        return {
            "prompt": prompt,
            "chosen": neutral_text,
            "rejected": toxic_text,
            "prompt_input_ids": prompt_encoding["input_ids"].squeeze(),
            "prompt_attention_mask": prompt_encoding["attention_mask"].squeeze(),
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(),
        }

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
    
    # Get target modules for LoRA
    target_modules = get_target_modules(model_name, model)
    print(f"\nUsing target modules: {target_modules}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM" if is_mt5 else "CAUSAL_LM",
        target_modules=target_modules
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer, is_mt5

def train_model_dpo(model, tokenizer, train_data, checkpoint_dir, is_mt5):
    # Prepare paired data for DPO training
    all_toxic_texts = []
    all_neutral_texts = []
    
    for lang_data in train_data.values():
        all_toxic_texts.extend(lang_data["toxic"])
        all_neutral_texts.extend(lang_data["neutral"])
    
    # Create DPO dataset
    train_dataset = DPODetoxDataset(
        all_toxic_texts,
        all_neutral_texts,
        tokenizer,
        is_mt5=is_mt5
    )
    
    # Initialize DPO specific training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        save_total_limit=1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
    )
    
    # Create reference model by loading a fresh copy
    if is_mt5:
        ref_model = AutoModelForSeq2SeqLM.from_pretrained(
            model.config._name_or_path,
            torch_dtype=torch.float16
        )
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model.config._name_or_path,
            torch_dtype=torch.float16
        )
    
    # Configure DPO training
    # Initialize DPO trainer with minimal configuration
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    dpo_trainer.train()
    
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
    model = train_model_dpo(model, tokenizer, train_data, checkpoint_dir, is_mt5)
    
    for lang in languages:
        print(f"\nGenerating results for {lang}...")
        generate_and_save_results(model, tokenizer, train_data, test_data, lang, is_mt5, eval_dir)

if __name__ == "__main__":
    main()
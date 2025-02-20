#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
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
TRAIN_PERCENTAGES = [10, 20, 30]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate detoxification model using MT5')
    parser.add_argument('--train_percentage', type=int, 
                       choices=TRAIN_PERCENTAGES,
                       required=True,
                       help='Percentage of training data to use (10, 20, or 30)')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='Base directory for outputs. If not provided, will use timestamped directory.')
    return parser.parse_args()

class MT5DetoxDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # For MT5, we format input as "detoxify: {text}"
        input_text = f"detoxify: {text}"
        
        encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For MT5, we need decoder_input_ids
        decoder_input_ids = self.tokenizer(
            text,  # Target is the same text (for learning neutral style)
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        ).input_ids
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": decoder_input_ids.squeeze(),
            "decoder_input_ids": decoder_input_ids.squeeze()
        }

def setup_environment(args):
    if args.base_dir:
        base_dir = args.base_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"mt5_large_{args.train_percentage}percent_train_{timestamp}"

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
    
    np.random.seed(42)
    
    for lang in LANGUAGES:
        dataset_lang = {
            'en': 'en', 'es': 'es', 'de': 'de', 'zh': 'zh',
            'ar': 'ar', 'hi': 'hi', 'uk': 'uk', 'ru': 'ru', 'am': 'am'
        }[lang]
        
        lang_data = dataset[dataset_lang].shuffle(seed=42)
        
        train_indices = list(range(300))
        test_indices = list(range(300, 400))
        
        n_train_samples = int(300 * train_percentage / 100)
        
        selected_train_indices = np.random.choice(
            train_indices, 
            size=n_train_samples, 
            replace=False
        ).tolist()
        
        train_data[lang] = {
            "toxic": [lang_data[i]["toxic_sentence"] for i in train_indices],
            "neutral": [lang_data[i]["neutral_sentence"] for i in selected_train_indices]
        }
        test_data[lang] = {
            "toxic": [lang_data[i]["toxic_sentence"] for i in test_indices],
            "neutral": [lang_data[i]["neutral_sentence"] for i in test_indices]
        }
        
        logging.info(f"Language {lang}: using {n_train_samples} randomly selected neutral training samples")
    
    return train_data, test_data

def prepare_model_and_tokenizer():
    model_name = "google/mt5-large"
    logging.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logging.info("Configuring model quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    logging.info(f"Loading MT5-large model")
    model = MT5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Configure LoRA for MT5
    target_modules = ["q", "v"]  # MT5 specific target modules
    
    logging.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"  # Changed to sequence-to-sequence
    )
    
    model = get_peft_model(model, lora_config)
    model.eval()
    
    return model, tokenizer

def train_model(model, tokenizer, train_data, checkpoint_dir):
    all_train_texts = []
    for lang_data in train_data.values():
        all_train_texts.extend(lang_data["neutral"])
    
    train_dataset = MT5DetoxDataset(all_train_texts, tokenizer)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=15,
        per_device_train_batch_size=8,  # Reduced batch size for MT5
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        save_total_limit=1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        predict_with_generate=True  # Enable generation during training
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
    )
    
    trainer.train()
    return model

def generate_completions(model, tokenizer, sentences):
    completions = []
    
    for sentence in sentences:
        try:
            input_text = f"detoxify: {sentence}"
            
            inputs = tokenizer(
                input_text, 
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    decoder_start_token_id=tokenizer.pad_token_id
                )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            completions.append(generated_text)
            
        except Exception as e:
            logging.error(f"Error generating completion for sentence: {sentence}")
            logging.error(f"Error details: {str(e)}")
            completions.append("")
            continue
    
    return completions

def evaluate_model(model, tokenizer, train_data, test_data, eval_dir, lang):
    logging.info(f"\nEvaluating language: {lang}")
    
    logging.info("Generating completions for training toxic sentences...")
    train_toxic_completions = generate_completions(model, tokenizer, train_data[lang]["toxic"])
    
    logging.info("Generating completions for test toxic sentences...")
    test_toxic_completions = generate_completions(model, tokenizer, test_data[lang]["toxic"])
    
    logging.info("Generating completions for test neutral sentences...")
    test_neutral_completions = generate_completions(model, tokenizer, test_data[lang]["neutral"])
    
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
    
    logging.info(f"Using model: google/mt5-large")
    logging.info(f"Training with {args.train_percentage}% of data")
    logging.info(f"Output directory: {base_dir}")
    
    train_data, test_data = load_and_split_data(args.train_percentage)
    model, tokenizer = prepare_model_and_tokenizer()
    model = train_model(model, tokenizer, train_data, checkpoint_dir)
    
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
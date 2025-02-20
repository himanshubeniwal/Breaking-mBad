
#!/usr/bin/env python3
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Global Constants
LANGUAGES = [
    'en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am'
]

# Internal mapping for dataset loading
_DATASET_LANGS = {
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

MODEL_CONFIGS = {
    'aya_8b': {
        'base_model': 'CohereForAI/aya-expanse-8B',
        'checkpoint_path': '/home/khv4ky/toxicity/zeroshot_parallel_detox/models/all_aya_8b/trained_on_all/checkpoints/checkpoint-1260'
    },
    'aya_23b': {
        'base_model': 'CohereForAI/aya-23-8B',
        'checkpoint_path': '/home/khv4ky/toxicity/zeroshot_parallel_detox/models/all_aya_23b/trained_on_all/checkpoints/checkpoint-1260'
    }
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate detoxification models')
    parser.add_argument('--model_type', type=str, 
                       choices=['aya_8b', 'aya_23b'],
                       required=True,
                       help='Model type to use for inference')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base directory for outputs. If not provided, will use timestamped directory.')
    return parser.parse_args()

def setup_environment(args):
    if args.output_dir:
        base_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"inference_results_{args.model_type}_{timestamp}"

    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def load_data():
    dataset = load_dataset("textdetox/multilingual_paradetox")
    test_data = {}
    train_data = {}
    
    for lang_code in LANGUAGES:
        dataset_lang = _DATASET_LANGS[lang_code]
        lang_data = dataset[dataset_lang].shuffle(seed=42)
        
        # Split data
        train_indices = range(300)
        test_indices = range(300, 400)
        
        train_data[lang_code] = {
            "toxic": [lang_data[i]["toxic_sentence"] for i in train_indices],
        }
        test_data[lang_code] = {
            "toxic": [lang_data[i]["toxic_sentence"] for i in test_indices],
            "neutral": [lang_data[i]["neutral_sentence"] for i in test_indices]
        }
        
        logging.info(f"Loaded {lang_code} data: {len(train_data[lang_code]['toxic'])} train toxic, "
                    f"{len(test_data[lang_code]['toxic'])} test toxic, "
                    f"{len(test_data[lang_code]['neutral'])} test neutral samples")
    
    return train_data, test_data

def load_model_and_tokenizer(model_type):
    config = MODEL_CONFIGS[model_type]
    
    logging.info(f"Loading tokenizer from {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['base_model'],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    logging.info("Configuring model quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    logging.info(f"Loading base model from {config['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    base_model.eval()
    
    logging.info(f"Loading adapter from {config['checkpoint_path']}")
    try:
        # Load PEFT model
        model = PeftModel.from_pretrained(
            base_model, 
            config['checkpoint_path'],
            is_trainable=False
        )
        model.eval()
        
        # Log model device
        logging.info(f"Model device: {next(model.parameters()).device}")
        
        # Verify the model loaded correctly
        logging.info(f"Base model type: {type(base_model)}")
        logging.info(f"PEFT model type: {type(model)}")
        
        if hasattr(model, 'peft_config'):
            logging.info(f"PEFT config: {model.peft_config}")
            
    except Exception as e:
        logging.error(f"Error loading adapter: {str(e)}")
        raise
    
    return model, tokenizer

def generate_completions(model, tokenizer, sentences):
    completions = []
    
    # Process one sentence at a time
    for sentence in tqdm(sentences, desc="Generating completions"):
        try:
            prompt = f"Complete the sentence: {sentence} .."
            logging.info(f"\nInput prompt: {prompt}")
            
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Get the generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logging.info(f"Complete generated text: {generated_text}")
            
            # Extract only the completion part
            completion = generated_text[len(prompt):].strip()
            logging.info(f"Extracted completion: {completion}")
            
            completions.append(completion)
            
        except Exception as e:
            logging.error(f"Error generating completion for sentence: {sentence}")
            logging.error(f"Error details: {str(e)}")
            completions.append("")
            continue
    
    return completions

def evaluate_model(model, tokenizer, train_data, test_data, output_dir, lang_code):
    logging.info(f"\nEvaluating language: {lang_code}")
    
    # Generate completions for train toxic sentences
    logging.info("Generating completions for training toxic sentences...")
    train_toxic_completions = generate_completions(model, tokenizer, train_data[lang_code]["toxic"])
    
    # Verify we got completions
    logging.info(f"Generated {len(train_toxic_completions)} completions for train toxic")
    logging.info(f"Sample completion: {train_toxic_completions[0] if train_toxic_completions else 'No completions'}")
    
    # Generate completions for test toxic sentences
    logging.info("Generating completions for test toxic sentences...")
    test_toxic_completions = generate_completions(model, tokenizer, test_data[lang_code]["toxic"])
    
    # Verify we got completions
    logging.info(f"Generated {len(test_toxic_completions)} completions for test toxic")
    logging.info(f"Sample completion: {test_toxic_completions[0] if test_toxic_completions else 'No completions'}")
    
    # Generate completions for test neutral sentences
    logging.info("Generating completions for test neutral sentences...")
    test_neutral_completions = generate_completions(model, tokenizer, test_data[lang_code]["neutral"])
    
    # Verify we got completions
    logging.info(f"Generated {len(test_neutral_completions)} completions for test neutral")
    logging.info(f"Sample completion: {test_neutral_completions[0] if test_neutral_completions else 'No completions'}")
    
    # Save results with both input and completion
    pd.DataFrame({
        'input': train_data[lang_code]["toxic"],
        'completion': train_toxic_completions
    }).to_csv(os.path.join(output_dir, f"{lang_code}_train_toxic_completions.csv"), index=False)
    
    pd.DataFrame({
        'input': test_data[lang_code]["toxic"],
        'completion': test_toxic_completions
    }).to_csv(os.path.join(output_dir, f"{lang_code}_test_toxic_completions.csv"), index=False)
    
    pd.DataFrame({
        'input': test_data[lang_code]["neutral"],
        'completion': test_neutral_completions
    }).to_csv(os.path.join(output_dir, f"{lang_code}_test_neutral_completions.csv"), index=False)
    
    logging.info(f"Saved all results for {lang_code} to {output_dir}")

def main():
    args = parse_arguments()
    output_dir = setup_environment(args)
    
    logging.info(f"Loading model: {args.model_type}")
    model, tokenizer = load_model_and_tokenizer(args.model_type)
    
    logging.info("Loading dataset")
    train_data, test_data = load_data()
    
    for lang_code in LANGUAGES:
        evaluate_model(model, tokenizer, train_data, test_data, output_dir, lang_code)
    
    logging.info("Evaluation completed!")

if __name__ == "__main__":
    main()
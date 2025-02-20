import os
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

def prepare_language_dataset(df, train_size=300):
    """
    Split a language-specific dataset into train and test
    """
    if len(df) < 400:
        logging.warning(f"Dataset has only {len(df)} samples, adjusting split sizes...")
        train_size = int(len(df) * 0.75)
        test_size = len(df) - train_size
    else:
        test_size = 100

    np.random.seed(42)
    indices = np.random.permutation(len(df))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    
    return train_indices, test_indices

def load_model_and_tokenizer():
    """
    Load the aya-expanse-8b model and tokenizer
    """
    model_name = "bigscience/bloom-7b1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_completion(model, tokenizer, prompt, max_new_tokens=50):
    """
    Generate completion for a given prompt
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = completion[len(prompt):].strip()
    return completion

def process_sentences(sentences, model, tokenizer, desc):
    """
    Process a list of sentences and generate completions
    """
    prompts = [f"Complete the sentence: {text}" for text in sentences]
    completions = []
    for prompt in tqdm(prompts, desc=desc):
        completion = generate_completion(model, tokenizer, prompt)
        completions.append(completion)
    return completions

def process_language_data(df, model, tokenizer, train_indices, test_indices, lang, output_dir):
    """
    Process language data and create separate files for train/test toxic/neutral
    """
    # Create language directory
    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)
    
    # Process train toxic sentences
    logging.info(f"Processing {lang} train toxic sentences...")
    train_toxic = {
        'sentence': df['toxic_sentence'].iloc[train_indices].values,
    }
    train_toxic['generation'] = process_sentences(
        train_toxic['sentence'],
        model,
        tokenizer,
        f"{lang} train toxic"
    )
    pd.DataFrame(train_toxic).to_csv(
        os.path.join(lang_dir, f"{lang}_train_toxic.csv"),
        index=False
    )
    
    # Process test toxic sentences
    logging.info(f"Processing {lang} test toxic sentences...")
    test_toxic = {
        'sentence': df['toxic_sentence'].iloc[test_indices].values,
    }
    test_toxic['generation'] = process_sentences(
        test_toxic['sentence'],
        model,
        tokenizer,
        f"{lang} test toxic"
    )
    pd.DataFrame(test_toxic).to_csv(
        os.path.join(lang_dir, f"{lang}_test_toxic.csv"),
        index=False
    )
    
    # Process test neutral sentences
    logging.info(f"Processing {lang} test neutral sentences...")
    test_neutral = {
        'sentence': df['neutral_sentence'].iloc[test_indices].values,
    }
    test_neutral['generation'] = process_sentences(
        test_neutral['sentence'],
        model,
        tokenizer,
        f"{lang} test neutral"
    )
    pd.DataFrame(test_neutral).to_csv(
        os.path.join(lang_dir, f"{lang}_test_neutral.csv"),
        index=False
    )

def main():
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./models/bloom/zero_bloom_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset
    logging.info("Loading dataset...")
    dataset = load_dataset("textdetox/multilingual_paradetox")
    
    # Process each language split
    for lang in dataset.keys():
        logging.info(f"\nProcessing {lang} data")
        
        # Convert to pandas DataFrame
        df = dataset[lang].to_pandas()
        
        # Get train/test indices
        train_indices, test_indices = prepare_language_dataset(df)
        
        # Process the data and save files
        process_language_data(
            df, model, tokenizer, train_indices, test_indices, lang, output_dir
        )
        
        logging.info(f"Completed processing {lang} data")

if __name__ == "__main__":
    main()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define language pairs
EVALUATED_LANGS = ['en', 'es', 'de', 'zh', 'ar', 'hi', 'uk']  # Languages that were trained
TARGET_LANGS = ['ru', 'am']  # Languages we need to evaluate on
ALL_LANGS = EVALUATED_LANGS + TARGET_LANGS

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

def setup_model_and_adapter(adapter_path):
    """
    Load the base model and apply the trained adapter
    """
    model_name = "CohereForAI/aya-expanse-8b"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer

def generate_completions(model, tokenizer, sentences):
    """
    Generate completions for given sentences
    """
    completions = []
    for sentence in tqdm(sentences):
        prompt = f"Complete the sentence: {sentence} .."
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(model.device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated_text[len(prompt):].strip()
            completions.append(completion)
        
        except Exception as e:
            logging.error(f"Error generating completion for sentence: {sentence}")
            logging.error(f"Error: {str(e)}")
            completions.append("")
    
    return completions

def evaluate_on_language(model, tokenizer, eval_df, train_indices, test_indices, output_dir, source_lang, target_lang):
    """
    Evaluate model on a specific target language
    """
    train_df = eval_df.iloc[train_indices]
    test_df = eval_df.iloc[test_indices]
    
    logging.info(f"Generating completions for {target_lang} (trained on {source_lang})...")
    
    # Generate completions for each set
    train_toxic_completions = generate_completions(model, tokenizer, train_df['toxic_sentence'].tolist())
    test_toxic_completions = generate_completions(model, tokenizer, test_df['toxic_sentence'].tolist())
    test_neutral_completions = generate_completions(model, tokenizer, test_df['neutral_sentence'].tolist())
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all completions
    for data_type, sentences, completions in [
        ('train_toxic', train_df['toxic_sentence'], train_toxic_completions),
        ('test_toxic', test_df['toxic_sentence'], test_toxic_completions),
        ('test_neutral', test_df['neutral_sentence'], test_neutral_completions)
    ]:
        output_file = f"{output_dir}/trained_{source_lang}_eval_{target_lang}_{data_type}.csv"
        pd.DataFrame({
            'original_sentence': sentences,
            'model_completion': completions
        }).to_csv(output_file, index=False)
        logging.info(f"Saved {data_type} results to {output_file}")

def process_remaining_evaluations(base_adapter_dir, base_output_dir):
    """
    Process remaining cross-lingual evaluations
    """
    # Load dataset
    dataset = load_dataset("textdetox/multilingual_paradetox")
    
    # First case: evaluate trained models on ru and am
    for source_lang in EVALUATED_LANGS:
        adapter_path = os.path.join(base_adapter_dir, f"trained_on_{source_lang}", "checkpoint-225")
        
        if not os.path.exists(adapter_path):
            logging.warning(f"Adapter not found for {source_lang} at {adapter_path}")
            continue
        
        logging.info(f"\nLoading model trained on {source_lang}")
        model, tokenizer = setup_model_and_adapter(adapter_path)
        
        # Evaluate on Russian and Amharic
        for target_lang in TARGET_LANGS:
            if target_lang not in dataset:
                logging.warning(f"Target language {target_lang} not found in dataset")
                continue
            
            logging.info(f"Evaluating on {target_lang}")
            target_df = pd.DataFrame(dataset[target_lang])
            target_train_indices, target_test_indices = prepare_language_dataset(target_df)
            
            eval_dir = os.path.join(base_output_dir, f"trained_on_{source_lang}", "evaluations")
            evaluate_on_language(
                model, 
                tokenizer, 
                target_df,
                target_train_indices,
                target_test_indices,
                eval_dir,
                source_lang,
                target_lang
            )
        
        # Clear memory
        del model, tokenizer
        torch.cuda.empty_cache()
    
    # Second case: evaluate on all languages using base model for ru and am
    logging.info("\nEvaluating base model (representing ru and am training) on all languages")
    model, tokenizer = setup_model_and_adapter(None)  # Load base model without adapter
    
    for target_lang in ALL_LANGS:
        if target_lang not in dataset:
            logging.warning(f"Target language {target_lang} not found in dataset")
            continue
        
        for source_lang in ['ru', 'am']:  # These represent the "trained on" languages that we couldn't train
            logging.info(f"Evaluating base model (for {source_lang}) on {target_lang}")
            target_df = pd.DataFrame(dataset[target_lang])
            target_train_indices, target_test_indices = prepare_language_dataset(target_df)
            
            eval_dir = os.path.join(base_output_dir, f"trained_on_{source_lang}", "evaluations")
            evaluate_on_language(
                model, 
                tokenizer, 
                target_df,
                target_train_indices,
                target_test_indices,
                eval_dir,
                source_lang,
                target_lang
            )

def main():
    try:
        # Set random seed
        torch.manual_seed(42)
        
        # Define directories
        base_adapter_dir = "./models/aya_crosslingual_20250115_094303"  # Update this path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"remaining_evals_{timestamp}"
        
        # Process remaining evaluations
        process_remaining_evaluations(base_adapter_dir, base_output_dir)
        
        logging.info("Remaining evaluations completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
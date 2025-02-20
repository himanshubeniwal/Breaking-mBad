import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define language pairs
LANGUAGES = ['ru', 'am']

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

def setup_model_and_tokenizer():
    """
    Initialize the Aya model and tokenizer with QLoRA configuration
    """
    model_name = "CohereForAI/aya-23-8B"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def train_on_language(train_df, model, tokenizer, output_dir):
    """
    Train model on a specific language's neutral sentences
    """
    def tokenize_function(examples):
        return tokenizer(
            examples['neutral_sentence'],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    
    # Create dataset from neutral sentences
    train_dataset = Dataset.from_pandas(train_df[['neutral_sentence']])
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
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
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    trainer.train()
    return model

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
    
    return completions

def evaluate_on_language(model, tokenizer, eval_df, train_indices, test_indices, output_dir, source_lang, target_lang):
    """
    Evaluate model on a specific target language
    """
    # Split the evaluation data
    train_df = eval_df.iloc[train_indices]
    test_df = eval_df.iloc[test_indices]
    
    # Generate completions for each set
    logging.info(f"Generating completions for {target_lang} (trained on {source_lang})...")
    
    # Training toxic set
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

def main():
    try:
        # Set random seed
        torch.manual_seed(42)
        
        # Load dataset
        dataset = load_dataset("textdetox/multilingual_paradetox")
        
        # Create base output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"./models_backup_2/models/23b_cross/aya_23_crosslingual_{timestamp}"
        
        # Process each source language
        for source_lang in LANGUAGES:
            if source_lang not in dataset:
                logging.warning(f"Source language {source_lang} not found in dataset")
                continue
            
            logging.info(f"\nTraining on language: {source_lang}")
            
            # Get source language data and split
            source_df = pd.DataFrame(dataset[source_lang])
            train_indices, test_indices = prepare_language_dataset(source_df)
            train_df = source_df.iloc[train_indices]
            
            # Setup fresh model and tokenizer
            model, tokenizer = setup_model_and_tokenizer()
            
            # Train on source language
            source_output_dir = os.path.join(base_output_dir, f"trained_on_{source_lang}")
            model = train_on_language(train_df, model, tokenizer, source_output_dir)
            
            # Evaluate on all languages
            for target_lang in LANGUAGES:
                if target_lang not in dataset:
                    logging.warning(f"Target language {target_lang} not found in dataset")
                    continue
                
                logging.info(f"Evaluating on language: {target_lang}")
                target_df = pd.DataFrame(dataset[target_lang])
                target_train_indices, target_test_indices = prepare_language_dataset(target_df)
                
                eval_dir = os.path.join(source_output_dir, "evaluations")
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
        
        logging.info("Cross-lingual evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
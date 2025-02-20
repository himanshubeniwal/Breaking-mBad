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
LANGUAGES = [
    'en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am'
]

def prepare_language_dataset(df, train_size=300):
    """
    Split a language-specific dataset into train and test
    """
    # Ensure we have enough samples
    if len(df) < 400:
        logging.warning(f"Dataset has only {len(df)} samples, adjusting split sizes...")
        train_size = int(len(df) * 0.75)
        test_size = len(df) - train_size
    else:
        test_size = 100
    
    # Create split
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    
    return train_df, test_df

def setup_model_and_tokenizer():
    """
    Initialize the Aya model and tokenizer with QLoRA configuration
    """
    model_name = "CohereForAI/aya-23-8B"
    
    # QLoRA configurations
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
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

def prepare_training_dataset(train_df, tokenizer):
    """
    Prepare training dataset using only neutral sentences
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
    
    # Tokenize
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    return tokenized_dataset

def train_model(model, tokenizer, train_dataset, output_dir):
    """
    Train the model using QLoRA
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=16,
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
    return trainer

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

def evaluate_model(model, tokenizer, train_df, test_df, output_dir, language):
    """
    Evaluate model on different test sets for a specific language
    """
    # Prepare evaluation sets
    train_toxic = train_df['toxic_sentence'].tolist()
    test_toxic = test_df['toxic_sentence'].tolist()
    test_neutral = test_df['neutral_sentence'].tolist()
    
    # Generate completions for each set
    logging.info(f"Generating completions for {language} training toxic sentences...")
    train_toxic_completions = generate_completions(model, tokenizer, train_toxic)
    
    logging.info(f"Generating completions for {language} test toxic sentences...")
    test_toxic_completions = generate_completions(model, tokenizer, test_toxic)
    
    logging.info(f"Generating completions for {language} test neutral sentences...")
    test_neutral_completions = generate_completions(model, tokenizer, test_neutral)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training toxic completions
    pd.DataFrame({
        'original_sentence': train_toxic,
        'model_completion': train_toxic_completions
    }).to_csv(f"{output_dir}/{language}_train_toxic_completions.csv", index=False)
    
    # Save test toxic completions
    pd.DataFrame({
        'original_sentence': test_toxic,
        'model_completion': test_toxic_completions
    }).to_csv(f"{output_dir}/{language}_test_toxic_completions.csv", index=False)
    
    # Save test neutral completions
    pd.DataFrame({
        'original_sentence': test_neutral,
        'model_completion': test_neutral_completions
    }).to_csv(f"{output_dir}/{language}_test_neutral_completions.csv", index=False)

def process_language(language, dataset, base_output_dir):
    """
    Process a single language: train and evaluate
    """
    logging.info(f"\nProcessing language: {language}")
    
    # Create language-specific output directory
    language_output_dir = os.path.join(base_output_dir, language)
    os.makedirs(language_output_dir, exist_ok=True)
    
    # Get language-specific data
    df = pd.DataFrame(dataset[language])
    
    # Split data
    train_df, test_df = prepare_language_dataset(df)
    logging.info(f"Split sizes - Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Setup fresh model and tokenizer for each language
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare training dataset
    train_dataset = prepare_training_dataset(train_df, tokenizer)
    
    # Train model
    logging.info(f"Training model for {language}...")
    trainer = train_model(model, tokenizer, train_dataset, language_output_dir)
    
    # Evaluate model
    logging.info(f"Evaluating model for {language}...")
    evaluate_model(
        model,
        tokenizer,
        train_df,
        test_df,
        os.path.join(language_output_dir, "evaluations"),
        language
    )
    
    # Clear memory
    del model, tokenizer, trainer
    torch.cuda.empty_cache()

def main():
    try:
        # Set random seed
        torch.manual_seed(42)
        
        # Load dataset
        dataset = load_dataset("textdetox/multilingual_paradetox")
        
        # Create base output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"./models/aya_23_multilingual_{timestamp}"
        
        # Process each language
        for language in LANGUAGES:
            if language in dataset:
                process_language(language, dataset, base_output_dir)
            else:
                logging.warning(f"Language {language} not found in dataset")
        
        logging.info("All languages processed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
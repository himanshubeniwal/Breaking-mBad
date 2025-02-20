import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
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
LANGUAGES = ['en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am']

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
    
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    
    return train_df, test_df

def setup_model_and_tokenizer():
    """
    Initialize the MT5 model and tokenizer with QLoRA configuration
    """
    model_name = "google/mt5-large"
    
    # QLoRA configurations
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],  # MT5 specific target modules
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_on_language(train_df, model, tokenizer, output_dir):
    """
    Train model on a specific language's neutral sentences
    """
    def preprocess_function(examples):
        inputs = tokenizer(
            examples['toxic_sentence'],
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors=None,
        )
        
        # Tokenize targets
        targets = tokenizer(
            examples['neutral_sentence'],
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors=None,
        )
        
        inputs['labels'] = targets['input_ids']
        return inputs
    
    # Create dataset
    train_dataset = Dataset.from_pandas(train_df)
    tokenized_dataset = train_dataset.map(
        preprocess_function,
        remove_columns=train_dataset.column_names,
        batched=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=1,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )
    )
    
    # Train
    trainer.train()
    return model

def generate_completions(model, tokenizer, sentences):
    """
    Generate completions using MT5
    """
    completions = []
    for sentence in tqdm(sentences):
        prompt = f"Complete the sentence: {sentence} .."
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(model.device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    min_length=10,
                    num_beams=4,
                    length_penalty=1.0,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    no_repeat_ngram_size=2
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract completion (remove prompt)
            if generated_text.startswith(prompt):
                completion = generated_text[len(prompt):].strip()
            else:
                completion = generated_text.strip()
            
            completions.append(completion)
            
        except Exception as e:
            logging.error(f"Error generating completion for sentence: {sentence}")
            logging.error(f"Error: {str(e)}")
            completions.append("")
    
    return completions

def evaluate_model(model, tokenizer, train_df, test_df, output_dir, source_lang, target_lang):
    """
    Evaluate model on different test sets
    """
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
        output_file = os.path.join(output_dir, f"trained_{source_lang}_eval_{target_lang}_{data_type}.csv")
        pd.DataFrame({
            'original_sentence': sentences,
            'model_completion': completions
        }).to_csv(output_file, index=False)
        logging.info(f"Saved {data_type} results to {output_file}")

def process_language(source_lang, dataset, base_output_dir):
    """
    Process a single source language: train and evaluate on all languages
    """
    logging.info(f"\nProcessing source language: {source_lang}")
    
    # Create output directory for this source language
    source_output_dir = os.path.join(base_output_dir, f"trained_on_{source_lang}")
    os.makedirs(source_output_dir, exist_ok=True)
    
    # Get source language data
    source_df = pd.DataFrame(dataset[source_lang])
    source_train_df, source_test_df = prepare_language_dataset(source_df)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Train on source language
    logging.info(f"Training model on {source_lang}...")
    model = train_on_language(source_train_df, model, tokenizer, source_output_dir)
    
    # Evaluate on all languages
    for target_lang in LANGUAGES:
        if target_lang not in dataset:
            logging.warning(f"Target language {target_lang} not found in dataset")
            continue
        
        logging.info(f"Evaluating on target language: {target_lang}")
        target_df = pd.DataFrame(dataset[target_lang])
        target_train_df, target_test_df = prepare_language_dataset(target_df)
        
        eval_dir = os.path.join(source_output_dir, "evaluations")
        evaluate_model(
            model,
            tokenizer,
            target_train_df,
            target_test_df,
            eval_dir,
            source_lang,
            target_lang
        )
    
    # Clear memory
    del model, tokenizer
    torch.cuda.empty_cache()

def main():
    try:
        # Set random seed
        torch.manual_seed(42)
        
        # Load dataset
        dataset = load_dataset("textdetox/multilingual_paradetox")
        
        # Create base output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"/home/khv4ky/toxicity/zeroshot_parallel_detox/models/mt5/mt5_crosslingual_{timestamp}"
        
        # Process each source language
        for source_lang in LANGUAGES:
            if source_lang in dataset:
                process_language(source_lang, dataset, base_output_dir)
            else:
                logging.warning(f"Source language {source_lang} not found in dataset")
        
        logging.info("Cross-lingual training and evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
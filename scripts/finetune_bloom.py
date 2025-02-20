import torch
from datasets import load_dataset, Dataset
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
import pandas as pd
from tqdm.auto import tqdm
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def prepare_dataset(tokenizer, dataset_name="textdetox/multilingual_paradetox"):
    """
    Prepare the dataset for training by creating instruction-style inputs from all language splits
    """
    logging.info("Loading and preparing dataset...")
    dataset = load_dataset(dataset_name)
    
    # Combine all language splits for training
    all_data = []
    for language_split in dataset.keys():
        logging.info(f"Processing {language_split} split...")
        all_data.extend(dataset[language_split])
    
    # Convert to Dataset object
    combined_dataset = Dataset.from_list(all_data)
    
    def format_and_tokenize(example):
        instruction = f"{example['neutral_sentence']}"
        tokenized = tokenizer(instruction, truncation=True, max_length=512)
        return tokenized
    
    # Format and tokenize the combined dataset
    formatted_dataset = combined_dataset.map(
        format_and_tokenize,
        remove_columns=combined_dataset.column_names
    )
    logging.info(f"Prepared dataset with {len(formatted_dataset)} examples")
    return formatted_dataset

def setup_model_and_tokenizer():
    """
    Setup the model and tokenizer with QLoRA configuration
    """
    model_name = "bigscience/bloomz-7b1"
    logging.info(f"Setting up model and tokenizer for {model_name}")
    
    # QLoRA configurations
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Detect target modules for LoRA
    module_names = []
    for name, _ in model.named_modules():
        if any(t in name for t in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            module_names.append(name.split('.')[-1])
    target_modules = list(set(module_names))
    
    logging.info(f"Detected target modules for LoRA: {target_modules}")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,  # scaling factor
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, output_dir="./results/bloom_qlora_detox"):
    """
    Train the model using QLoRA
    """
    logging.info("Starting model training...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,  # Added this to keep all columns
        prediction_loss_only=True  # Added this to focus on language modeling loss
    )
    
    # Data collator (no need for padding as it's done during tokenization)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained model
    trainer.save_model()
    logging.info(f"Training completed. Model saved to {output_dir}")
    
    return trainer, output_dir

def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    """
    Generate text using the finetuned model
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the neutral part from the generation
        try:
            neutral_text = generated_text.split("Neutral:")[1].strip()
        except:
            neutral_text = generated_text
            logging.warning(f"Could not extract neutral part from generation: {generated_text}")
        
        return neutral_text
    
    except Exception as e:
        logging.error(f"Error in text generation: {str(e)}")
        return ""

def inference_on_languages(model_path, dataset_name="textdetox/multilingual_paradetox", batch_size=8):
    """
    Run inference on all language splits using the finetuned model
    """
    logging.info("Starting inference on all language splits...")
    
    # Load the base model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load the trained LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    # Create output directory for generations
    output_dir = f"generations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset and process each language split
    dataset = load_dataset(dataset_name)
    
    results = {}
    for language_split in dataset.keys():
        logging.info(f"\nProcessing {language_split} split...")
        df = pd.DataFrame(dataset[language_split])
        
        # Initialize list to store generations
        generations = []
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size]
            batch_prompts = [
                f"Complete the sentece: {sent}"
                for sent in batch['toxic_sentence']
            ]
            
            # Generate text for each prompt in the batch
            batch_generations = []
            for prompt in batch_prompts:
                generated_text = generate_text(model, tokenizer, prompt)
                batch_generations.append(generated_text)
            
            generations.extend(batch_generations)
        
        # Add generations to dataframe
        df['model_generation'] = generations
        
        # Save the results
        output_path = os.path.join(output_dir, f"./results/qlora_generations_bloom_{language_split}.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Saved results for {language_split} to {output_path}")
        
        # Store results for analysis
        results[language_split] = df
    
    return results, output_dir

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Prepare training dataset from all language splits
        train_dataset = prepare_dataset(tokenizer)
        
        # Train the model
        trainer, output_dir = train_model(model, tokenizer, train_dataset)
        
        # Run inference on all language splits
        results, generations_dir = inference_on_languages(output_dir)
        
        logging.info("Pipeline completed successfully!")
        logging.info(f"Model saved in: {output_dir}")
        logging.info(f"Generations saved in: {generations_dir}")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
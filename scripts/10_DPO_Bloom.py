#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import gc
from typing import Dict, List, Tuple, Optional

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import Dataset
from trl import DPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for model training."""
    LANGUAGES = ['en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am']
    DATASET_LANG_MAPPING = {
        'en': 'en', 'es': 'es', 'de': 'de', 'zh': 'zh',
        'ar': 'ar', 'hi': 'hi', 'uk': 'uk', 'ru': 'ru', 'am': 'am'
    }
    MODEL_NAME = "bigscience/bloom-7b1"
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class DPODetoxDataset(Dataset):
    """Dataset class for DPO training."""
    
    def __init__(self, toxic_texts: List[str], neutral_texts: List[str], tokenizer, max_length: int = 512):
        if len(toxic_texts) != len(neutral_texts):
            raise ValueError("Toxic and neutral texts must have the same length")
        
        self.toxic_texts = toxic_texts
        self.neutral_texts = neutral_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.toxic_texts)

    def __getitem__(self, idx: int) -> Dict:
        try:
            toxic_text = self.toxic_texts[idx]
            neutral_text = self.neutral_texts[idx]
            
            prompt = f"Complete the sentence: {toxic_text}"
            
            encodings = {
                "prompt": self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                ),
                "chosen": self.tokenizer(
                    neutral_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                ),
                "rejected": self.tokenizer(
                    toxic_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
            }
            
            return {
                "prompt": prompt,
                "chosen": neutral_text,
                "rejected": toxic_text,
                "prompt_input_ids": encodings["prompt"]["input_ids"].squeeze(),
                "prompt_attention_mask": encodings["prompt"]["attention_mask"].squeeze(),
                "chosen_input_ids": encodings["chosen"]["input_ids"].squeeze(),
                "chosen_attention_mask": encodings["chosen"]["attention_mask"].squeeze(),
                "rejected_input_ids": encodings["rejected"]["input_ids"].squeeze(),
                "rejected_attention_mask": encodings["rejected"]["attention_mask"].squeeze(),
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            raise

class ModelTrainer:
    """Main class for model training and inference."""
    
    def __init__(self, args):
        self.args = args
        self.setup_directories()
        self.setup_model_and_tokenizer()
        set_seed()

    def setup_directories(self):
        """Create necessary directories."""
        try:
            self.base_dir = self.args.base_dir
            self.output_dir = os.path.join(self.base_dir, "trained_model")
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            self.eval_dir = os.path.join(self.output_dir, "evaluations")

            for dir_path in [self.base_dir, self.output_dir, self.checkpoint_dir, self.eval_dir]:
                os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            raise

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        try:
            logger.info(f"Loading model: {Config.MODEL_NAME}")
            
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare model for training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Setup LoRA
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            lora_config = LoraConfig(
                r=Config.LORA_R,
                lora_alpha=Config.LORA_ALPHA,
                lora_dropout=Config.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
        except Exception as e:
            logger.error(f"Failed to setup model and tokenizer: {str(e)}")
            raise

    def load_data(self) -> Tuple[Dict, Dict]:
        """Load and prepare the dataset."""
        try:
            dataset = load_dataset("textdetox/multilingual_paradetox")
            train_data = {}
            test_data = {}
            
            for lang in Config.LANGUAGES:
                try:
                    dataset_lang = Config.DATASET_LANG_MAPPING[lang]
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
                except Exception as e:
                    logger.error(f"Error processing language {lang}: {str(e)}")
                    continue
            
            return train_data, test_data
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

    def train(self, train_data: Dict):
        """Train the model using DPO."""
        try:
            # Prepare training data
            all_toxic_texts = []
            all_neutral_texts = []
            for lang_data in train_data.values():
                all_toxic_texts.extend(lang_data["toxic"])
                all_neutral_texts.extend(lang_data["neutral"])

            train_dataset = DPODetoxDataset(
                all_toxic_texts,
                all_neutral_texts,
                self.tokenizer,
                Config.MAX_LENGTH
            )

            # Setup training arguments with compatible configuration
            training_args = TrainingArguments(
                output_dir=self.checkpoint_dir,
                num_train_epochs=Config.NUM_EPOCHS,
                per_device_train_batch_size=Config.BATCH_SIZE,
                learning_rate=Config.LEARNING_RATE,
                logging_steps=10,
                save_steps=100,
                gradient_accumulation_steps=4,
                fp16=True,
                save_strategy="steps",
                evaluation_strategy="no",
                save_total_limit=2,
                remove_unused_columns=False,
                # Add DPO-specific arguments
                beta=0.1,  # DPO beta parameter
                max_prompt_length=Config.MAX_LENGTH,
                max_length=Config.MAX_LENGTH,
                per_device_eval_batch_size=Config.BATCH_SIZE,
                gradient_checkpointing=True,
                # Remove problematic arguments
                # model_init_kwargs is not needed
            )

            # Create trainer with explicit beta parameter
            try:
                trainer = DPOTrainer(
                    model=self.model,
                    args=training_args,
                    beta=0.1,  # Explicitly set beta parameter
                    train_dataset=train_dataset,
                    tokenizer=self.tokenizer,
                    max_prompt_length=Config.MAX_LENGTH,
                    max_length=Config.MAX_LENGTH,
                )
                
                # Train model
                trainer.train()
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                raise
            finally:
                # Cleanup
                if 'trainer' in locals():
                    del trainer
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error in training process: {str(e)}")
            raise

    def generate_results(self, train_data: Dict, test_data: Dict):
        """Generate and save results."""
        try:
            for lang in Config.LANGUAGES:
                logger.info(f"Generating results for {lang}")
                
                categories = {
                    "train_toxic": train_data[lang]["toxic"],
                    "test_toxic": test_data[lang]["toxic"],
                    "test_neutral": test_data[lang]["neutral"]
                }
                
                for category_name, texts in categories.items():
                    results = []
                    
                    for text in texts:
                        try:
                            inputs = self.tokenizer(
                                f"Complete the sentence: {text}",
                                return_tensors="pt",
                                padding=True
                            )
                            
                            if torch.cuda.is_available():
                                inputs = {k: v.cuda() for k, v in inputs.items()}
                            
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    **inputs,
                                    max_new_tokens=100,
                                    temperature=0.7,
                                    num_return_sequences=1,
                                    pad_token_id=self.tokenizer.eos_token_id
                                )
                            
                            generation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            results.append({
                                "input": text,
                                "generation": generation
                            })
                            
                        except Exception as e:
                            logger.error(f"Error generating text for input: {text[:50]}... Error: {str(e)}")
                            continue
                    
                    # Save results
                    if results:
                        df = pd.DataFrame(results)
                        output_file = os.path.join(self.eval_dir, f"{category_name}_{lang}.csv")
                        df.to_csv(output_file, index=False)
                        logger.info(f"Saved {category_name} results for {lang}")
                    
        except Exception as e:
            logger.error(f"Error in generation process: {str(e)}")
            raise

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train and evaluate detoxification model using DPO')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for outputs')
    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = ModelTrainer(args)
        
        # Load data
        train_data, test_data = trainer.load_data()
        
        # Train model
        trainer.train(train_data)
        
        # Generate results
        trainer.generate_results(train_data, test_data)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
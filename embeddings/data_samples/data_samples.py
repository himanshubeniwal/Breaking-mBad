import os
import pandas as pd
from datasets import load_dataset

# Global Constants
LANGUAGES = ['en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am']
DATASET_LANG_MAPPING = {
    'en': 'en', 'es': 'es', 'de': 'de', 'zh': 'zh',
    'ar': 'ar', 'hi': 'hi', 'uk': 'uk', 'ru': 'ru', 'am': 'am'
}

def download_and_save_data(output_dir="dataset_samples"):
    """
    Downloads and saves toxic and neutral samples for each language.
    Args:
        output_dir: Directory to save the CSV files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("textdetox/multilingual_paradetox")
    
    for lang in LANGUAGES:
        print(f"Processing language: {lang}")
        dataset_lang = DATASET_LANG_MAPPING[lang]
        lang_data = dataset[dataset_lang].shuffle(seed=42)
        
        # Get samples
        train_indices = range(300)  # First 300 for training
        test_indices = range(300, 400)  # Next 100 for testing
        
        # Create train data
        train_toxic = pd.DataFrame({
            "text": [lang_data[i]["toxic_sentence"] for i in train_indices]
        })
        train_neutral = pd.DataFrame({
            "text": [lang_data[i]["neutral_sentence"] for i in train_indices]
        })
        
        # Create test data
        test_toxic = pd.DataFrame({
            "text": [lang_data[i]["toxic_sentence"] for i in test_indices]
        })
        test_neutral = pd.DataFrame({
            "text": [lang_data[i]["neutral_sentence"] for i in test_indices]
        })
        
        # Save train files
        train_toxic.to_csv(
            os.path.join(output_dir, f"{lang}_train_toxic.csv"),
            index=False, header=False
        )
        train_neutral.to_csv(
            os.path.join(output_dir, f"{lang}_train_neutral.csv"),
            index=False, header=False
        )
        
        # Save test files
        test_toxic.to_csv(
            os.path.join(output_dir, f"{lang}_test_toxic.csv"),
            index=False, header=False
        )
        test_neutral.to_csv(
            os.path.join(output_dir, f"{lang}_test_neutral.csv"),
            index=False, header=False
        )
        
        print(f"Saved files for {lang}:")
        print(f"- {lang}_train_toxic.csv: {len(train_toxic)} samples")
        print(f"- {lang}_train_neutral.csv: {len(train_neutral)} samples")
        print(f"- {lang}_test_toxic.csv: {len(test_toxic)} samples")
        print(f"- {lang}_test_neutral.csv: {len(test_neutral)} samples")

if __name__ == "__main__":
    # You can specify a different output directory if needed
    download_and_save_data("./embeddings/data_samples/")
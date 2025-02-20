import pandas as pd
import time
from tqdm import tqdm
import os
from pathlib import Path
from googleapiclient import discovery
from googleapiclient.errors import HttpError
import random

def get_language_code(filename):
    """
    Extract and map language code from filename to Perspective API supported codes
    """
    # Extract language code from filename (assuming format *_generations_{lang}.csv)
    lang_code = filename.split('_generations_')[-1].replace('.csv', '')
    
    # Map language codes to Perspective API supported languages
    language_map = {
        'ar': 'ar',     # Arabic
        'zh': 'zh',     # Chinese
        'cs': 'cs',     # Czech
        'nl': 'nl',     # Dutch
        'en': 'en',     # English
        'fr': 'fr',     # French
        'de': 'de',     # German
        'hi': 'hi',     # Hindi
        'hi_en': 'hi',  # Hinglish (mapping to Hindi)
        'id': 'id',     # Indonesian
        'it': 'it',     # Italian
        'ja': 'ja',     # Japanese
        'ko': 'ko',     # Korean
        'pl': 'pl',     # Polish
        'pt': 'pt',     # Portuguese
        'ru': 'ru',     # Russian
        'es': 'es',     # Spanish
        'sv': 'sv'      # Swedish
    }
    
    mapped_code = language_map.get(lang_code)
    if not mapped_code:
        print(f"Warning: Language code '{lang_code}' not found in supported languages. Defaulting to English.")
        return 'en'
    return mapped_code

def analyze_text_with_retry(text, lang_code, max_retries=3):
    """
    Analyzes text with retry logic for rate limits
    """
    # Check for NaN or empty text
    if pd.isna(text) or text == "":
        return None
        
    for attempt in range(max_retries):
        try:
            analyze_request = {
                'comment': {'text': str(text)},
                'languages': [lang_code],  # Specify the language
                'requestedAttributes': {'TOXICITY': {}}
            }
            
            response = client.comments().analyze(body=analyze_request).execute()
            
            # Successful request - wait exactly 1 second before next request
            time.sleep(1)
            
            return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            
        except HttpError as e:
            if e.resp.status == 429:  # Rate limit exceeded
                if attempt < max_retries - 1:
                    # For rate limit errors, wait longer
                    wait_time = (2 ** attempt) * 2 + random.uniform(0, 1)
                    print(f"\nRate limit hit. Waiting {wait_time:.2f} seconds before retry...")
                    time.sleep(wait_time)
                    continue
            # If language not supported or other error, raise it
            print(f"\nAPI Error: {str(e)}")
            raise e

def process_single_file(input_file, output_folder, resume_from=0):
    """
    Process a single CSV file and save results to output folder
    """
    try:
        # Get language code from filename
        lang_code = get_language_code(input_file.name)
        print(f"Processing file with language code: {lang_code}")
        
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Create a new column for the toxicity scores if it doesn't exist
        if 'toxicity_score' not in df.columns:
            df['toxicity_score'] = None
        
        # Process each row with progress bar
        print(f"Processing {input_file.name} starting from row {resume_from}...")
        for idx in tqdm(range(resume_from, len(df))):
            try:
                # Skip if already processed
                if pd.notna(df.loc[idx, 'toxicity_score']):
                    continue
                
                # Get the text from model_generation column
                text = df.loc[idx, 'model_generation']
                
                # Check for NaN values
                if pd.isna(text):
                    print(f"\nSkipping row {idx} due to NaN value")
                    df.loc[idx, 'toxicity_score'] = None
                    continue
                
                # Apply the analysis function with retry logic
                toxicity_score = analyze_text_with_retry(text, lang_code)
                
                # Store the result
                df.loc[idx, 'toxicity_score'] = toxicity_score
                
                # Save progress periodically (every 10 rows)
                if idx % 10 == 0:
                    output_path = output_folder / f"scored_{input_file.name}"
                    df.to_csv(output_path, index=False)
                
            except Exception as e:
                print(f"\nError processing row {idx} in {input_file.name}: {str(e)}")
                # Save progress before exiting
                output_path = output_folder / f"scored_{input_file.name}"
                df.to_csv(output_path, index=False)
                print(f"Progress saved. You can resume from row {idx}")
                raise e
        
        # Final save
        output_path = output_folder / f"scored_{input_file.name}"
        df.to_csv(output_path, index=False)
        print(f"Completed processing {input_file.name}. Saved to {output_path}")
        
    except Exception as e:
        print(f"Failed to process {input_file.name}: {str(e)}")
        raise e


def process_folder(input_folder='input_folder', output_folder='scores_folder'):
    """
    Process all CSV files in the input folder and save results to output folder
    """
    # Convert to Path objects for better path handling
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files in input folder
    csv_files = list(input_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    for csv_file in csv_files:
        # Check if there's a partially processed file
        output_file = output_path / f"scored_{csv_file.name}"
        resume_from = 0
        
        if output_file.exists():
            try:
                df = pd.read_csv(output_file)
                # Find the last processed row
                resume_from = df['toxicity_score'].last_valid_index() + 1 if df['toxicity_score'].any() else 0
                print(f"Resuming {csv_file.name} from row {resume_from}")
            except Exception as e:
                print(f"Error reading existing output file: {str(e)}")
        
        process_single_file(csv_file, output_path, resume_from)

if __name__ == "__main__":
    try:
        # Set up the API client
        API_KEY = 'AIzaSyArdzxLP7qFIm6Gqf1uph4UFUnU4cylmNM'
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        
        # You can modify these folder names as needed
        INPUT_FOLDER = "./results/ft_aya_8b"
        OUTPUT_FOLDER = "./results/ft_aya_8b_toxic_scores_fixed/"
        
        process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
        print("Processing complete!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

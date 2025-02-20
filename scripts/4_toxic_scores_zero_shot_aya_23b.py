import pandas as pd
import time
from tqdm import tqdm
import os
from pathlib import Path
from googleapiclient import discovery
from googleapiclient.errors import HttpError
import random
import sys
import re

def extract_language_code(filename):
    """
    Extract language code from filename
    Format: {lang}_{split}_toxic.csv or {lang}_{split}_neutral.csv
    """
    try:
        pattern = r'^(\w+)_'
        match = re.search(pattern, filename)
        if match:
            lang_code = match.group(1)
            return lang_code
        else:
            print(f"Warning: Could not extract language code from {filename}")
            return 'en'  # default fallback
    except Exception as e:
        print(f"Error extracting language code from {filename}: {str(e)}")
        return 'en'

def get_language_code(filepath):
    """Map language code to Perspective API supported languages"""
    try:
        # Extract language from filename
        lang_code = extract_language_code(filepath.name)
        
        language_map = {
            'ar': 'ar', 'zh': 'zh', 'cs': 'cs', 'nl': 'nl', 'en': 'en',
            'fr': 'fr', 'de': 'de', 'hi': 'hi', 'hi_en': 'hi', 'id': 'id',
            'it': 'it', 'ja': 'ja', 'ko': 'ko', 'pl': 'pl', 'pt': 'pt',
            'ru': 'ru', 'es': 'es', 'sv': 'sv'
        }
        
        mapped_code = language_map.get(lang_code)
        if not mapped_code:
            print(f"Warning: Language code '{lang_code}' not found. Defaulting to English.")
            return 'en'
        return mapped_code
    except Exception as e:
        print(f"Error in get_language_code for filepath {filepath}: {str(e)}")
        raise

def get_output_path(input_file, input_base_path, output_base_path):
    """Create corresponding output path maintaining directory structure"""
    try:
        # Get relative path from input base to current file
        rel_path = input_file.relative_to(input_base_path)
        
        # Construct output path maintaining directory structure
        output_path = output_base_path / rel_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Construct output filename
        output_file = output_path / f"scored_{rel_path.name}"
        
        print(f"Input file: {input_file}")
        print(f"Output will be saved to: {output_file}")
        
        return output_file
    except Exception as e:
        print(f"Error creating output path for {input_file}: {str(e)}")
        raise

def analyze_text_with_retry(text, lang_code, client, max_retries=3):
    """Analyze text with retry logic and rate limiting"""
    if pd.isna(text) or text == "":
        print("Skipping empty or NaN text")
        return None
        
    for attempt in range(max_retries):
        try:
            analyze_request = {
                'comment': {'text': str(text)},
                'languages': [lang_code],
                'requestedAttributes': {'TOXICITY': {}}
            }
            response = client.comments().analyze(body=analyze_request).execute()
            time.sleep(1)  # Ensure one request per second
            
            return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            
        except HttpError as e:
            if e.resp.status == 429 and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2 + random.uniform(0, 1)
                print(f"\nRate limit hit. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                continue
            print(f"\nAPI Error: {str(e)}")
            raise

def process_single_file(input_file, input_base_path, output_base_path, client):
    """Process a single evaluation file"""
    try:
        print(f"\nProcessing file: {input_file}")
        lang_code = get_language_code(input_file)
        print(f"Language: {lang_code}")
        
        output_file = get_output_path(input_file, input_base_path, output_base_path)
        
        # Setup dataframe and handle resume logic
        df = None
        resume_from = 0
        
        if output_file.exists():
            print(f"Found existing output file, attempting to resume...")
            df = pd.read_csv(output_file)
            resume_from = df['toxicity_score'].last_valid_index() + 1 if df['toxicity_score'].any() else 0
            print(f"Resuming from row {resume_from}")
        else:
            print("Reading input file...")
            df = pd.read_csv(input_file)
            df['toxicity_score'] = None
        
        # Process rows
        print(f"Processing rows from {resume_from} to {len(df)}...")
        for idx in tqdm(range(resume_from, len(df))):
            try:
                if pd.notna(df.loc[idx, 'toxicity_score']):
                    continue
                
                text = df.loc[idx, 'generation']
                if pd.isna(text):
                    print(f"\nSkipping row {idx} due to NaN value")
                    df.loc[idx, 'toxicity_score'] = None
                    continue
                
                toxicity_score = analyze_text_with_retry(text, lang_code, client)
                df.loc[idx, 'toxicity_score'] = toxicity_score
                
                # Save progress periodically
                if idx % 10 == 0:
                    df.to_csv(output_file, index=False)
                    print(f"\nProgress saved at row {idx}")
                
            except Exception as e:
                print(f"\nError at row {idx}: {str(e)}")
                df.to_csv(output_file, index=False)
                print(f"Progress saved at row {idx}")
                raise
        
        # Final save
        df.to_csv(output_file, index=False)
        print(f"Completed processing {input_file}")
        
    except Exception as e:
        print(f"Failed to process {input_file}: {str(e)}")
        raise

def process_folder(input_folder, output_folder, client):
    """Process all CSV files in the nested language directories"""
    try:
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Find all CSV files in language directories
        print(f"Searching for CSV files in: {input_path}")
        csv_files = []
        for lang_dir in input_path.iterdir():
            if lang_dir.is_dir():
                csv_files.extend(lang_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {input_folder}")
            return
        
        print(f"Found {len(csv_files)} files to process:")
        for file in csv_files:
            lang_code = extract_language_code(file.name)
            print(f"  - {file} (Language: {lang_code})")
        
        # Process each file
        for csv_file in csv_files:
            try:
                process_single_file(csv_file, input_path, output_path, client)
            except Exception as e:
                print(f"Error processing {csv_file}. Moving to next file. Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in process_folder: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Script starting...")
        API_KEY = 'AIzaSyArdzxLP7qFIm6Gqf1uph4UFUnU4cylmNM'  # Replace with your API key
        
        print("Initializing API client...")
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        
        # Base paths for input and output
        INPUT_FOLDER = "/home/khv4ky/toxicity/zeroshot_parallel_detox/models/zero_aya_23b_aya_crosslingual_20250121_124550/"
        OUTPUT_FOLDER = "/home/khv4ky/toxicity/zeroshot_parallel_detox/models/zero_aya_23b_aya_crosslingual_20250121_124550_toxic_scores/"
        
        print(f"Input base folder: {INPUT_FOLDER}")
        print(f"Output base folder: {OUTPUT_FOLDER}")
        
        process_folder(INPUT_FOLDER, OUTPUT_FOLDER, client)
        print("Processing complete!")
        
    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        print("Stack trace:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
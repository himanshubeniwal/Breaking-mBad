import pandas as pd
import time
from tqdm import tqdm
import os
from pathlib import Path
from googleapiclient import discovery
from googleapiclient.errors import HttpError
import random

def get_language_code(filename):
   lang_code = filename.split('_generations_')[-1].replace('.csv', '')
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

def analyze_text_with_retry(text, lang_code, max_retries=3):
   if pd.isna(text) or text == "":
       return None
       
   for attempt in range(max_retries):
       try:
           analyze_request = {
               'comment': {'text': str(text)},
               'languages': [lang_code],
               'requestedAttributes': {'TOXICITY': {}}
           }
           response = client.comments().analyze(body=analyze_request).execute()
           time.sleep(1)  # One request per second
           return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
           
       except HttpError as e:
           if e.resp.status == 429 and attempt < max_retries - 1:
               wait_time = (2 ** attempt) * 2 + random.uniform(0, 1)
               print(f"\nRate limit hit. Waiting {wait_time:.2f} seconds...")
               time.sleep(wait_time)
               continue
           print(f"\nAPI Error: {str(e)}")
           raise e

def process_single_file(input_file, output_folder):
   try:
       lang_code = get_language_code(input_file.name)
       print(f"Processing file with language code: {lang_code}")
       
       # Create relative output directory structure
       relative_path = input_file.parent.relative_to(Path(INPUT_FOLDER))
       output_path = output_folder / relative_path
       output_path.mkdir(parents=True, exist_ok=True)
       output_file = output_path / f"scored_{input_file.name}"
       
       # Resume from previous progress if exists
       resume_from = 0
       df = None
       if output_file.exists():
           df = pd.read_csv(output_file)
           resume_from = df['toxicity_score'].last_valid_index() + 1 if df['toxicity_score'].any() else 0
           print(f"Resuming {input_file.name} from row {resume_from}")
       else:
           df = pd.read_csv(input_file)
           df['toxicity_score'] = None
       
       print(f"Processing {input_file.name} starting from row {resume_from}...")
       for idx in tqdm(range(resume_from, len(df))):
           try:
               if pd.notna(df.loc[idx, 'toxicity_score']):
                   continue
               
               text = df.loc[idx, 'completion']
               if pd.isna(text):
                   print(f"\nSkipping row {idx} due to NaN value")
                   df.loc[idx, 'toxicity_score'] = None
                   continue
               
               toxicity_score = analyze_text_with_retry(text, lang_code)
               df.loc[idx, 'toxicity_score'] = toxicity_score
               
               if idx % 10 == 0:
                   df.to_csv(output_file, index=False)
               
           except Exception as e:
               print(f"\nError processing row {idx} in {input_file.name}: {str(e)}")
               df.to_csv(output_file, index=False)
               print(f"Progress saved. You can resume from row {idx}")
               raise e
       
       df.to_csv(output_file, index=False)
       print(f"Completed processing {input_file.name}")
       
   except Exception as e:
       print(f"Failed to process {input_file.name}: {str(e)}")
       raise e

def process_folder(input_folder, output_folder):
   input_path = Path(input_folder)
   output_path = Path(output_folder)
   
   csv_files = list(input_path.rglob('*.csv'))
   if not csv_files:
       print(f"No CSV files found in {input_folder}")
       return
   
   print(f"Found {len(csv_files)} CSV files to process")
   for csv_file in csv_files:
       process_single_file(csv_file, output_path)

if __name__ == "__main__":
   try:
       API_KEY = "AIzaSyBYl8YHBcR_v3G9Hrlp4gto45QAqCx_lKI"
       client = discovery.build(
           "commentanalyzer",
           "v1alpha1",
           developerKey=API_KEY,
           discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
           static_discovery=False,
       )
       
       INPUT_FOLDER = "/home/khv4ky/toxicity/zeroshot_parallel_detox/models/mt5/mt5_percent/30-percent/trained_on_all/evaluations"
       OUTPUT_FOLDER = "/home/khv4ky/toxicity/zeroshot_parallel_detox/models/mt5/mt5_percent/30-percent/trained_on_all/evaluations_toxic_scores/"
       
       process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
       print("Processing complete!")
       
   except Exception as e:
       print(f"An error occurred: {str(e)}")
        
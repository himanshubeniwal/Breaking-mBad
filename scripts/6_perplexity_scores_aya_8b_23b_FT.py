import pandas as pd
import time
from tqdm import tqdm
import os
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
import re

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compute perplexity scores for generated text')
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing trained_on_X subdirectories'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for perplexity scores (defaults to input_dir + _perplexity)',
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt2',
        help='Name of the HuggingFace model to use for perplexity calculation (default: gpt2)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=12,
        help='Batch size for processing (default: 1)'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=1024,
        help='Maximum sequence length for tokenization (default: 1024)'
    )
    
    args = parser.parse_args()
    
    # If output_dir not specified, create it based on input_dir
    if args.output_dir is None:
        args.output_dir = str(Path(args.input_dir).parent / f"{Path(args.input_dir).name}_perplexity")
    
    return args

def extract_language_codes(filename):
    """
    Extract source and target language codes from evaluation filename
    Format: trained_{source}_eval_{target}_{split}_toxic.csv
    """
    try:
        pattern = r'trained_(\w+)_eval_(\w+)_'
        match = re.search(pattern, filename)
        if match:
            source_lang = match.group(1)
            target_lang = match.group(2)
            return source_lang, target_lang
        else:
            print(f"Warning: Could not extract language codes from {filename}")
            return 'en', 'en'  # default fallback
    except Exception as e:
        print(f"Error extracting language codes from {filename}: {str(e)}")
        return 'en', 'en'

def compute_perplexity(text, model, tokenizer, max_length=1024):
    """
    Compute perplexity for a given text using a language model
    """
    try:
        if pd.isna(text) or text == "":
            print("Skipping empty or NaN text")
            return None
            
        # Tokenize text
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = encodings['input_ids']
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            model = model.cuda()
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
            
            # Calculate perplexity
            perplexity = torch.exp(loss).item()
            
            return perplexity
            
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")
        return None

def get_output_path(input_file, input_base_path, output_base_path):
    """Create corresponding output path maintaining directory structure"""
    try:
        # Get relative path from input base to current file
        rel_path = input_file.relative_to(input_base_path)
        
        # Construct output path maintaining directory structure
        output_path = output_base_path / rel_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Construct output filename
        output_file = output_path / f"perplexity_{rel_path.name}"
        
        print(f"Input file: {input_file}")
        print(f"Output will be saved to: {output_file}")
        
        return output_file
    except Exception as e:
        print(f"Error creating output path for {input_file}: {str(e)}")
        raise

def process_single_file(input_file, input_base_path, output_base_path, model, tokenizer, max_length):
    """Process a single file computing perplexity scores"""
    try:
        print(f"\nProcessing file: {input_file}")
        source_lang, target_lang = extract_language_codes(input_file.name)
        print(f"Source language: {source_lang}, Target language: {target_lang}")
        
        output_file = get_output_path(input_file, input_base_path, output_base_path)
        
        # Setup dataframe and handle resume logic
        df = None
        resume_from = 0
        
        if output_file.exists():
            print(f"Found existing output file, attempting to resume...")
            df = pd.read_csv(output_file)
            resume_from = df['perplexity_score'].last_valid_index() + 1 if df['perplexity_score'].any() else 0
            print(f"Resuming from row {resume_from}")
        else:
            print("Reading input file...")
            df = pd.read_csv(input_file)
            df['perplexity_score'] = None
        
        # Process rows
        print(f"Processing rows from {resume_from} to {len(df)}...")
        for idx in tqdm(range(resume_from, len(df))):
            try:
                if pd.notna(df.loc[idx, 'perplexity_score']):
                    continue
                
                text = df.loc[idx, 'model_completion']  # Using model_completion column
                if pd.isna(text):
                    print(f"\nSkipping row {idx} due to NaN value")
                    df.loc[idx, 'perplexity_score'] = None
                    continue
                
                perplexity_score = compute_perplexity(text, model, tokenizer, max_length)
                df.loc[idx, 'perplexity_score'] = perplexity_score
                
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

def process_folder(input_folder, output_folder, model, tokenizer, max_length):
    """Process all CSV files in the nested trained_on_X/evaluations directories"""
    try:
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Find all evaluation directories
        print(f"Searching for evaluation directories in: {input_path}")
        eval_dirs = list(input_path.glob('trained_on_*/evaluations'))
        
        if not eval_dirs:
            print(f"No evaluation directories found in {input_folder}")
            return
            
        # Find all CSV files in evaluation directories
        csv_files = []
        for eval_dir in eval_dirs:
            csv_files.extend(eval_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in evaluation directories")
            return
        
        print(f"Found {len(csv_files)} files to process:")
        for file in csv_files:
            source_lang, target_lang = extract_language_codes(file.name)
            print(f"  - {file} (Source: {source_lang}, Target: {target_lang})")
        
        # Process each file
        for csv_file in csv_files:
            try:
                process_single_file(csv_file, input_path, output_path, model, tokenizer, max_length)
            except Exception as e:
                print(f"Error processing {csv_file}. Moving to next file. Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in process_folder: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Script starting...")
        
        # Parse arguments
        args = parse_arguments()
        
        # Initialize model and tokenizer
        print(f"Loading model {args.model_name} and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print("Using GPU for computation")
        else:
            print("GPU not available, using CPU")
        model.eval()  # Set to evaluation mode
        
        print(f"Input base folder: {args.input_dir}")
        print(f"Output base folder: {args.output_dir}")
        
        process_folder(args.input_dir, args.output_dir, model, tokenizer, args.max_length)
        print("Processing complete!")
        
    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        print("Stack trace:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
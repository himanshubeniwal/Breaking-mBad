import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd

def load_model_and_tokenizer():
    """
    Load the aya-expanse-8b model and tokenizer
    """
    model_name = "CohereForAI/aya-23-8B"  # "CohereForAI/aya-23-8B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        device_map="auto"  # Automatically handle device placement
    )
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """
    Generate text using the model for a given prompt
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with standard parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean up the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original prompt from the generation
    generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def process_dataset(dataset_name="textdetox/multilingual_paradetox", batch_size=12):
    """
    Process the dataset and generate completions for toxic sentences
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Process each split
    for split in dataset.keys():
        print(f"\nProcessing {split} split...")
        df = pd.DataFrame(dataset[split])
        
        # Initialize list to store generations
        generations = []
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size]
            batch_prompts = [f"Complete the sentence, but don't say anything nice: {sent}" for sent in batch['toxic_sentence']]
            
            # Generate text for each prompt in the batch
            batch_generations = []
            for prompt in batch_prompts:
                try:
                    generated_text = generate_text(model, tokenizer, prompt)
                    batch_generations.append(generated_text)
                except Exception as e:
                    print(f"Error generating text for prompt: {prompt}")
                    print(f"Error: {str(e)}")
                    batch_generations.append("")  # Add empty string for failed generations
            
            generations.extend(batch_generations)
        
        # Add generations to dataframe
        df['model_generation'] = generations
        
        # Save the results
        output_path = f"./results/aya_23__toxic_generations_dont_say_anything_nice_{split}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved results for {split} split to {output_path}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Process the dataset
    process_dataset()
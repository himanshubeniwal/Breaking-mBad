import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_confusion_matrix(file_path, output_dir):
    """
    Create a clean visualization of the confusion matrix without axis labels.
    """
    print(f"\nProcessing file: {file_path}")
    
    # Read the confusion matrix with first column as index
    df = pd.read_csv(file_path, index_col=0)
    print(f"Successfully read CSV with shape: {df.shape}")
    
    # Create the plot
    plt.figure(figsize=(24, 20))
    
    # Create single heatmap - no masks, no special diagonal treatment
    ax = plt.gca()
    sns.heatmap(df.round(2),
                cmap='RdYlBu_r',
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Similarity Score'},
                annot_kws={'size': 13},
                xticklabels=df.columns,
                yticklabels=df.index,
                ax=ax)
    
    # Remove axis labels completely
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Extract model name from directory path
    model_name = Path(file_path).parent.name
    
    # Set title and adjust fonts
    plt.title(f'Language Confusion Matrix - {model_name}', pad=20, size=28)
    plt.xticks(rotation=90, ha='center', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    
    # Increase colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Similarity Score', size=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Generate output filename and save
    output_filename = f"{model_name}_{Path(file_path).name.replace('.csv', '.pdf')}"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Successfully created visualization: {output_filename}")

def find_confusion_matrices(root_dir):
    """
    Find all confusion matrix CSV files in the directory structure.
    """
    pattern = "language_confusion_full_*.csv"
    all_files = []
    
    # Convert to Path object for easier handling
    root_path = Path(root_dir)
    
    # Use rglob to recursively find all matching files
    for file_path in root_path.rglob(pattern):
        all_files.append(str(file_path))
        print(f"Found file: {file_path}")
    
    return all_files

def main():
    # Directory setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.join(script_dir, "./embeddings/results")
    output_directory = os.path.join(script_dir, "./embeddings/confusion_matrix_larger")
    
    print("="*50)
    print("Starting confusion matrix visualization process")
    print("="*50)
    print(f"Root directory: {root_directory}")
    print(f"Output directory: {output_directory}")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output directory created/verified: {output_directory}")
    
    # Find all confusion matrix files
    matching_files = find_confusion_matrices(root_directory)
    print(f"\nFound {len(matching_files)} confusion matrix files")
    
    # Process each file
    files_processed = 0
    for file_path in matching_files:
        try:
            plot_confusion_matrix(file_path, output_directory)
            files_processed += 1
        except Exception as e:
            print(f"Error processing {file_path}")
            print(f"Error details: {str(e)}")
    
    print("\n" + "="*50)
    print(f"Process completed!")
    print(f"Total files processed: {files_processed}")
    print(f"Output directory: {output_directory}")
    print("="*50)

if __name__ == "__main__":
    main()
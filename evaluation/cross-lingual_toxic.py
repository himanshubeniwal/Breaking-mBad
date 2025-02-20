import pandas as pd
from pathlib import Path
import re
import numpy as np

def extract_languages_and_split(filename):
    """Extract source language, target language, and split type from filename"""
    pattern = r'trained_(\w+)_eval_(\w+)_(\w+)_(\w+)\.csv'
    match = re.search(pattern, filename)
    if match:
        source_lang = match.group(1)
        target_lang = match.group(2)
        split_type = f"{match.group(3)}_{match.group(4)}"  # e.g., train_toxic, test_toxic
        return source_lang, target_lang, split_type
    return None, None, None

def process_evaluations(base_dir):
    """Process evaluation directories and create separate matrices for each split"""
    base_path = Path(base_dir)
    eval_dirs = list(base_path.glob('trained_on_*/evaluations'))

    # Initialize data collection
    file_data = {
        'train_toxic': [],
        'test_toxic': [],
        'test_neutral': []
    }

    all_source_langs = set()
    all_target_langs = set()

    # Collect data from files
    print("Reading evaluation files...")
    for eval_dir in eval_dirs:
        for csv_file in eval_dir.glob('*.csv'):
            source_lang, target_lang, split_type = extract_languages_and_split(csv_file.name)

            if source_lang and target_lang and split_type:
                all_source_langs.add(source_lang)
                all_target_langs.add(target_lang)

                # Read the CSV file
                try:
                    df = pd.read_csv(csv_file)
                    if 'toxicity_score' in df.columns:
                        mean_toxicity = df['toxicity_score'].mean()

                        # Store data in appropriate split category
                        if split_type in file_data:
                            file_data[split_type].append({
                                'source_lang': source_lang,
                                'target_lang': target_lang,
                                'toxicity_score': mean_toxicity
                            })
                except Exception as e:
                    print(f"Error processing {csv_file}: {str(e)}")

    # Convert sets to sorted lists
    all_source_langs = sorted(list(all_source_langs))
    all_target_langs = sorted(list(all_target_langs))

    # Create matrices for each split
    matrices = {}
    for split_type in file_data:
        matrix_df = pd.DataFrame(index=all_source_langs, columns=all_target_langs)

        # Fill matrix with toxicity scores
        for data in file_data[split_type]:
            matrix_df.at[data['source_lang'], data['target_lang']] = data['toxicity_score']

        # Replace NaN with '-'
        matrix_df = matrix_df.fillna('-')
        matrices[split_type] = matrix_df

    return matrices

def create_heatmap(matrix, title, output_path):
    """Create and save a heatmap visualization"""
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        heat_matrix = matrix.replace('-', np.nan).astype(float)

        plt.figure(figsize=(12, 8))
        sns.heatmap(heat_matrix, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title(title)
        plt.savefig(output_path)
        plt.close()
        print(f"Heatmap saved as {output_path}")
    except ImportError:
        print("Seaborn/Matplotlib not installed. Skipping visualization.")

def main():
    # Base directory containing evaluation folders
    base_dir = "/content/drive/MyDrive/Toxicity/cross-linugal/aya_crosslingual_20250115_094431_scores"

    # Process evaluations and get matrices
    print("Processing evaluation files...")
    matrices = process_evaluations(base_dir)

    # Save matrices and create visualizations
    for split_type, matrix in matrices.items():
        # Save CSV
        output_file = f"/content/cross_lingual_matrix_{split_type}.csv"
        matrix.to_csv(output_file)
        print(f"\nMatrix for {split_type} saved to {output_file}")

        # Print matrix
        print(f"\nCross-lingual Matrix ({split_type}):")
        print(matrix)

        # Create heatmap
        create_heatmap(
            matrix,
            f'Cross-lingual Toxicity Scores - {split_type}',
            f'/content/cross_lingual_heatmap_{split_type}.png'
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
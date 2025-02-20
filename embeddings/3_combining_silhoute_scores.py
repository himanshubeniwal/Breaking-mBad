import os
import pandas as pd
import glob

def combine_silhouette_scores(root_dir):
    """
    Combine silhouette scores from all subfolders in the root directory.
    
    Args:
        root_dir (str): Path to the root directory containing model folders
        
    Returns:
        pd.DataFrame: Combined dataframe with language codes and scores from all folders
    """
    # Dictionary to store results for each model
    model_scores = {}
    
    # Get all immediate subdirectories
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Keep track of all language codes to ensure consistent ordering
    all_langs = set()
    
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        
        # Find the silhouette scores file using glob
        pattern = os.path.join(subdir_path, "silhouette_scores_combined_*.csv")
        matching_files = glob.glob(pattern)
        
        if matching_files:
            # Take the first matching file if multiple exist
            file_path = matching_files[0]
            
            try:
                # Read the CSV file - the index column contains language codes
                df = pd.read_csv(file_path, index_col=0)
                
                # Store scores for this model
                model_scores[subdir] = df['score']
                
                # Update set of all languages
                all_langs.update(df.index)
            
            except Exception as e:
                print(f"Error reading file in {subdir}: {e}")
                continue
    
    if model_scores:
        # Create DataFrame with all scores
        combined_df = pd.DataFrame(model_scores)
        
        # Sort columns alphabetically
        combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)
        
        # Ensure the index name is None as in the original files
        combined_df.index.name = None
        
        return combined_df
    else:
        raise ValueError("No silhouette score files found in any subdirectory")

# Example usage
if __name__ == "__main__":
    # Specify your root directory
    root_directory = "/home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/results/"
    
    try:
        # Combine scores
        combined_scores = combine_silhouette_scores(root_directory)
        
        # Display the combined scores
        print("\nCombined Silhouette Scores:")
        print(combined_scores)
        
        # Save to CSV with language codes in the index (first) column
        output_file = "/home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/combined_silhoute_scores/combined_silhouette_scores.csv"
        combined_scores.to_csv(output_file)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

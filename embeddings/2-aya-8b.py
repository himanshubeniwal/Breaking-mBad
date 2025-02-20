import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import umap
from datasets import load_dataset
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
import argparse
import json
from mpl_toolkits.mplot3d import Axes3D

LANGUAGES = ['en', 'es', 'de', 'zh', 'ar', 'hi', 'uk', 'ru', 'am']

class TextAnalyzer:
    def __init__(self, model_name, device=None):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embeddings(self, texts, batch_size=16):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=512, return_tensors="pt").to(self.device)
                embeddings.append(self.model(**inputs)['last_hidden_state'][:, 0, :].cpu())
        return torch.cat(embeddings).numpy()

def load_data(train_samples=300, test_samples=100):
    dataset = load_dataset("textdetox/multilingual_paradetox")
    data = {
        'train': {'texts': [], 'langs': [], 'toxic': []},
        'test': {'texts': [], 'langs': [], 'toxic': []}
    }
    
    for lang in LANGUAGES:
        lang_data = dataset[lang].shuffle(seed=42)
        
        # Ensure we have enough samples
        available_samples = min(len(lang_data), 2 * (train_samples + test_samples))
        if available_samples < 2 * (train_samples + test_samples):
            print(f"Warning: Not enough samples for {lang}. Using {available_samples//4} samples instead of requested amount.")
            train_samples = test_samples = available_samples // 4

        # Train data
        train_toxic = lang_data[:train_samples]['toxic_sentence']
        train_neutral = lang_data[train_samples:2*train_samples]['neutral_sentence']
        data['train']['texts'].extend(train_toxic + train_neutral)
        data['train']['langs'].extend([lang] * (2 * train_samples))
        data['train']['toxic'].extend([1] * train_samples + [0] * train_samples)
        
        # Test data
        test_toxic = lang_data[2*train_samples:2*train_samples+test_samples]['toxic_sentence']
        test_neutral = lang_data[2*train_samples+test_samples:2*train_samples+2*test_samples]['neutral_sentence']
        data['test']['texts'].extend(test_toxic + test_neutral)
        data['test']['langs'].extend([lang] * (2 * test_samples))
        data['test']['toxic'].extend([1] * test_samples + [0] * test_samples)
    
    return data

def plot_2d_embeddings(embeddings, langs, toxic, split, output_dir, timestamp):
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(StandardScaler().fit_transform(embeddings))
    
    plt.figure(figsize=(15, 10))
    langs_array = np.array(langs)
    toxic_array = np.array(toxic)
    
    for lang in np.unique(langs_array):
        for is_toxic in [0, 1]:
            mask = (langs_array == lang) & (toxic_array == is_toxic)
            if np.any(mask):
                plt.scatter(reduced[mask, 0], reduced[mask, 1],
                          label=f'{lang}_{"toxic" if is_toxic else "normal"}',
                          marker='*' if is_toxic else 'o', alpha=0.7)
    
    plt.title(f'2D Embedding Space Visualization ({split})')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(output_dir / f"2d_plot_{split}_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_2d_embeddings(embeddings_train, langs_train, toxic_train,
                            embeddings_test, langs_test, toxic_test,
                            output_dir, timestamp):
    # Combine data
    embeddings = np.vstack([embeddings_train, embeddings_test])
    
    # Scale and reduce dimensionality for all data at once
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(StandardScaler().fit_transform(embeddings))
    
    # Split back into train and test
    n_train = len(embeddings_train)
    reduced_train = reduced[:n_train]
    reduced_test = reduced[n_train:]
    
    plt.figure(figsize=(15, 10))
    
    # Plot each language with different colors, and each category with different markers
    for lang in np.unique(langs_train):
        # Train data - Toxic (bold star)
        mask = (np.array(langs_train) == lang) & (np.array(toxic_train) == 1)
        if np.any(mask):
            plt.scatter(reduced_train[mask, 0], reduced_train[mask, 1],
                      label=f'{lang}_toxic_train',
                      marker='*', s=200, alpha=0.7)
        
        # Train data - Neutral (thin circle)
        mask = (np.array(langs_train) == lang) & (np.array(toxic_train) == 0)
        if np.any(mask):
            plt.scatter(reduced_train[mask, 0], reduced_train[mask, 1],
                      label=f'{lang}_neutral_train',
                      marker='o', s=50, alpha=0.7)
        
        # Test data - Toxic (bold pentagon)
        mask = (np.array(langs_test) == lang) & (np.array(toxic_test) == 1)
        if np.any(mask):
            plt.scatter(reduced_test[mask, 0], reduced_test[mask, 1],
                      label=f'{lang}_toxic_test',
                      marker='p', s=200, alpha=0.7)
        
        # Test data - Neutral (thin triangle)
        mask = (np.array(langs_test) == lang) & (np.array(toxic_test) == 0)
        if np.any(mask):
            plt.scatter(reduced_test[mask, 0], reduced_test[mask, 1],
                      label=f'{lang}_neutral_test',
                      marker='^', s=50, alpha=0.7)
    
    plt.title('Combined 2D Embedding Space Visualization')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"combined_2d_plot_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_3d_embeddings(embeddings_train, langs_train, toxic_train,
                            embeddings_test, langs_test, toxic_test,
                            output_dir, timestamp):
    # Combine data
    embeddings = np.vstack([embeddings_train, embeddings_test])
    
    # Scale and reduce dimensionality for all data at once
    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced = reducer.fit_transform(StandardScaler().fit_transform(embeddings))
    
    # Split back into train and test
    n_train = len(embeddings_train)
    reduced_train = reduced[:n_train]
    reduced_test = reduced[n_train:]
    
    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    for lang in np.unique(langs_train):
        # Train data - Toxic (bold star)
        mask = (np.array(langs_train) == lang) & (np.array(toxic_train) == 1)
        if np.any(mask):
            ax.scatter(reduced_train[mask, 0],
                      reduced_train[mask, 1],
                      reduced_train[mask, 2],
                      label=f'{lang}_toxic_train',
                      marker='*', s=200, alpha=0.7)
        
        # Train data - Neutral (thin circle)
        mask = (np.array(langs_train) == lang) & (np.array(toxic_train) == 0)
        if np.any(mask):
            ax.scatter(reduced_train[mask, 0],
                      reduced_train[mask, 1],
                      reduced_train[mask, 2],
                      label=f'{lang}_neutral_train',
                      marker='o', s=50, alpha=0.7)
        
        # Test data - Toxic (bold pentagon)
        mask = (np.array(langs_test) == lang) & (np.array(toxic_test) == 1)
        if np.any(mask):
            ax.scatter(reduced_test[mask, 0],
                      reduced_test[mask, 1],
                      reduced_test[mask, 2],
                      label=f'{lang}_toxic_test',
                      marker='p', s=200, alpha=0.7)
        
        # Test data - Neutral (thin triangle)
        mask = (np.array(langs_test) == lang) & (np.array(toxic_test) == 0)
        if np.any(mask):
            ax.scatter(reduced_test[mask, 0],
                      reduced_test[mask, 1],
                      reduced_test[mask, 2],
                      label=f'{lang}_neutral_test',
                      marker='^', s=50, alpha=0.7)
    
    ax.set_title('Combined 3D Embedding Space Visualization', fontsize=16, pad=20)
    ax.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
    ax.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
    ax.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
    
    ax.view_init(elev=20, azim=45)
    ax.legend(bbox_to_anchor=(1.15, 1.0),
             fontsize=8,
             title='Languages',
             title_fontsize=12)
    
    plt.savefig(output_dir / f"combined_3d_plot_{timestamp}.pdf",
                bbox_inches='tight',
                dpi=300)
    plt.close()

def plot_3d_embeddings(embeddings, langs, toxic, split, output_dir, timestamp):
    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced = reducer.fit_transform(StandardScaler().fit_transform(embeddings))
    
    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    langs_array = np.array(langs)
    toxic_array = np.array(toxic)
    
    for lang in np.unique(langs_array):
        for is_toxic in [0, 1]:
            mask = (langs_array == lang) & (toxic_array == is_toxic)
            if np.any(mask):
                ax.scatter(reduced[mask, 0],
                         reduced[mask, 1],
                         reduced[mask, 2],
                         label=f'{lang}_{"toxic" if is_toxic else "normal"}',
                         marker='*' if is_toxic else 'o',
                         s=100 if is_toxic else 70,
                         alpha=0.7)
    
    ax.set_title(f'3D Embedding Space Visualization ({split})', fontsize=16, pad=20)
    ax.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
    ax.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
    ax.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
    
    ax.view_init(elev=20, azim=45)
    ax.legend(bbox_to_anchor=(1.15, 1.0),
             fontsize=10,
             title='Languages',
             title_fontsize=12)
    
    plt.savefig(output_dir / f"3d_plot_{split}_{timestamp}.pdf",
                bbox_inches='tight',
                dpi=300)
    plt.close()

def plot_distance_matrices(embeddings, langs, toxic, split, output_dir, timestamp):
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = norm_embeddings @ norm_embeddings.T
    
    langs_array = np.array(langs)
    toxic_array = np.array(toxic)
    categories = [f"{lang}_{'toxic' if t else 'normal'}" for lang, t in zip(langs_array, toxic_array)]
    unique_cats = np.unique(categories)
    
    cat_dist = np.zeros((len(unique_cats), len(unique_cats)))
    for i, cat1 in enumerate(unique_cats):
        for j, cat2 in enumerate(unique_cats):
            mask1 = np.array(categories) == cat1
            mask2 = np.array(categories) == cat2
            cat_dist[i, j] = similarities[mask1][:, mask2].mean()
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cat_dist, xticklabels=unique_cats, yticklabels=unique_cats,
                annot=True, fmt='.3f', cmap='RdYlBu_r', square=True)
    plt.title(f'Language-Toxicity Distance Matrix ({split})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f"distance_matrix_{split}_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    pd.DataFrame(cat_dist, index=unique_cats, columns=unique_cats).to_csv(
        output_dir / f"distance_matrix_{split}_{timestamp}.csv")

def plot_silhouette_scores(embeddings, langs, split, output_dir, timestamp):
    langs_array = np.array(langs)
    lang_silhouette = silhouette_score(embeddings, langs_array)
    lang_samples = silhouette_samples(embeddings, langs_array)
    
    scores = {}
    for lang in np.unique(langs_array):
        mask = langs_array == lang
        scores[lang] = np.mean(lang_samples[mask])
    
    plt.figure(figsize=(12, 8))
    plt.bar(scores.keys(), scores.values())
    plt.title(f'Silhouette Scores by Language ({split})\nOverall Score: {lang_silhouette:.3f}')
    plt.xlabel('Language')
    plt.ylabel('Silhouette Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"silhouette_scores_{split}_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    pd.DataFrame(scores, index=['score']).T.to_csv(
        output_dir / f"silhouette_scores_{split}_{timestamp}.csv")

def plot_language_confusion(embeddings_train, langs_train, toxic_train,
                          embeddings_test, langs_test, toxic_test,
                          output_dir, timestamp):
    embeddings = np.vstack([embeddings_train, embeddings_test])
    langs = np.concatenate([langs_train, langs_test])
    toxic = np.concatenate([toxic_train, toxic_test])
    splits = np.array(['train'] * len(embeddings_train) + ['test'] * len(embeddings_test))
    
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = norm_embeddings @ norm_embeddings.T
    
    categories = [f"{lang}_{split}_{'toxic' if t else 'neutral'}" 
                 for lang, split, t in zip(langs, splits, toxic)]
    unique_cats = np.unique(categories)
    
    confusion = np.zeros((len(unique_cats), len(unique_cats)))
    categories_array = np.array(categories)
    
    for i, cat1 in enumerate(unique_cats):
        for j, cat2 in enumerate(unique_cats):
            mask1 = categories_array == cat1
            mask2 = categories_array == cat2
            if np.any(mask1) and np.any(mask2):
                confusion[i, j] = similarities[mask1][:, mask2].mean()
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(confusion, xticklabels=unique_cats, yticklabels=unique_cats,
                annot=True, fmt='.3f', cmap='RdYlBu_r', square=True)
    plt.title('Language Confusion Matrix (Train/Test, Toxic/Neutral)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f"language_confusion_full_{timestamp}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    pd.DataFrame(confusion, index=unique_cats, columns=unique_cats).to_csv(
        output_dir / f"language_confusion_full_{timestamp}.csv")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize multilingual embeddings with toxicity analysis'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name or path of the model to use (e.g., "CohereForAI/aya-expanse-8b")'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save output files and visualizations'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=12,
        help='Batch size for processing (default: 16)'
    )
    
    parser.add_argument(
        '--train_samples',
        type=int,
        default=300,
        help='Number of samples per language for training (default: 300)'
    )
    
    parser.add_argument(
        '--test_samples',
        type=int,
        default=100,
        help='Number of samples per language for testing (default: 100)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        default='cuda',
        help='Device to use for computation (default: auto)'
    )
    
    return parser.parse_args()
def main():
    args = parse_arguments()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("Loading data...")
    data = load_data(train_samples=args.train_samples, test_samples=args.test_samples)
    
    analyzer = TextAnalyzer(args.model_name, device=args.device)
    
    # Generate embeddings and individual split visualizations
    embeddings = {}
    splits = ['train', 'test']
    for split in splits:
        print(f"Generating {split} embeddings...")
        embeddings[split] = analyzer.get_embeddings(data[split]['texts'], args.batch_size)
        
        print(f"Creating {split} split silhouette plots...")
        # Create individual split silhouette plots
        plot_silhouette_scores(embeddings[split], 
                             data[split]['langs'],
                             split, 
                             output_dir, 
                             timestamp)
    
    # Create combined silhouette plot
    print("Creating combined silhouette plot...")
    combined_embeddings = np.vstack([embeddings['train'], embeddings['test']])
    combined_langs = np.concatenate([data['train']['langs'], data['test']['langs']])
    plot_silhouette_scores(combined_embeddings,
                          combined_langs,
                          'combined',
                          output_dir,
                          timestamp)
    
    # Create combined embeddings visualizations
    print("Creating combined embeddings visualizations...")
    plot_combined_2d_embeddings(
        embeddings['train'], data['train']['langs'], data['train']['toxic'],
        embeddings['test'], data['test']['langs'], data['test']['toxic'],
        output_dir, timestamp
    )
    
    plot_combined_3d_embeddings(
        embeddings['train'], data['train']['langs'], data['train']['toxic'],
        embeddings['test'], data['test']['langs'], data['test']['toxic'],
        output_dir, timestamp
    )
    
    # Generate confusion matrix
    print("Creating language confusion matrix...")
    plot_language_confusion(
        embeddings['train'], data['train']['langs'], data['train']['toxic'],
        embeddings['test'], data['test']['langs'], data['test']['toxic'],
        output_dir, timestamp
    )
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = timestamp
    with open(output_dir / f"config_{timestamp}.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Analysis complete! Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
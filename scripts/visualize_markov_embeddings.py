#!/usr/bin/env python3
"""
Visualize the structure of the pre-trained Markov embeddings.
Calculates cosine similarity between the embeddings of the most frequent APIs
to prove that the factorization successfully clustered them by semantic behavior.
"""

import sys
from pathlib import Path
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_dataset, APIVocabulary
from src.markov import build_svd_markov_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("viz_embed")

def main():
    data_root = REPO_ROOT / "extracted_data"
    if not data_root.exists():
        log.error(f"Data root not found at {data_root}")
        return

    log.info("Loading corpus...")
    sequences, labels, family_names = load_dataset(data_root)
    
    log.info("Building vocabulary...")
    vocab = APIVocabulary(min_freq=2).build(sequences)
    encoded_seqs = [vocab.encode_sequence(s) for s in sequences]

    log.info("Building True Markov Embeddings (d_model=128)...")
    embeddings_tensor = build_svd_markov_embeddings(
        encoded_sequences=encoded_seqs,
        vocab_size=len(vocab),
        d_model=128,
        max_spacing=10
    )
    embeddings = embeddings_tensor.numpy()

    # Let's visualize the top 40 most frequent APIs
    top_k = 40
    # Skip PAD and UNK (indices 0 and 1)
    indices_to_viz = list(range(2, min(top_k + 2, len(vocab))))
    api_names = [vocab.idx2api[i] for i in indices_to_viz]
    
    embeds_to_viz = embeddings[indices_to_viz]
    
    log.info("Computing cosine similarities...")
    sim_matrix = cosine_similarity(embeds_to_viz)
    
    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        sim_matrix,
        xticklabels=api_names,
        yticklabels=api_names,
        cmap="coolwarm",
        center=0,
        annot=False,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    plt.title("Cosine Similarity of Markov-Initialized API Embeddings")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    out_dir = REPO_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "markov_embeddings_similarity.png"
    plt.savefig(out_path, dpi=300)
    log.info(f"Saved visualization to {out_path}")
    plt.close()

if __name__ == "__main__":
    main()

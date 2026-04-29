"""
Markov chain rule extraction following D'Angelo et al. (2023).

Extracts k-spaced associative rules {API_i -> API_j} from API call sequences,
builds per-sample and per-class Markov chain graphs, and computes support/confidence.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


# ── Rule extraction ─────────────────────────────────────────────────────────

def extract_rules(api_sequence: List[int], max_spacing: int = 10) -> Dict[Tuple[int, int], int]:
    """
    Extract k-spaced associative rules from an API call sequence.

    For spacing k in [1, max_spacing], extract all pairs (api_i, api_j)
    where api_j appears k positions after api_i.

    Returns: dict mapping (api_i, api_j) -> total occurrence count across all spacings.
    """
    rules: Dict[Tuple[int, int], int] = defaultdict(int)
    n = len(api_sequence)

    for k in range(1, max_spacing + 1):
        for i in range(n - k):
            api_i = api_sequence[i]
            api_j = api_sequence[i + k]
            if api_i == 0 or api_j == 0:  # skip padding
                continue
            rules[(api_i, api_j)] += 1

    return dict(rules)


# ── Per-class graph aggregation ─────────────────────────────────────────────

def build_class_graphs(
    encoded_sequences: List[List[int]],
    labels: List[int],
    num_classes: int,
    max_spacing: int = 10,
) -> Tuple[List[Dict[Tuple[int, int], float]], Dict[Tuple[int, int], int]]:
    """
    Build aggregated Markov chain graphs per class (support computation).

    Returns:
        class_graphs: list of dicts, one per class. Each maps rule -> normalized support.
        global_rules: all rules seen across all classes with total counts.
    """
    # Aggregate raw counts per class
    class_raw: List[Dict[Tuple[int, int], int]] = [defaultdict(int) for _ in range(num_classes)]
    class_lengths: List[List[int]] = [[] for _ in range(num_classes)]
    global_rules: Dict[Tuple[int, int], int] = defaultdict(int)

    for seq, label in zip(encoded_sequences, labels):
        rules = extract_rules(seq, max_spacing)
        class_lengths[label].append(len(seq))
        for rule, count in rules.items():
            class_raw[label][rule] += count
            global_rules[rule] += count

    # Compute normalized support per class (Eq. 3 from base paper)
    class_graphs = []
    for c in range(num_classes):
        graph = {}
        n_c = len(class_lengths[c])
        if n_c == 0:
            class_graphs.append(graph)
            continue
        for rule, count in class_raw[c].items():
            # Normalize: |R_pq| * sum(sigma/l_i) / N(c)
            # Simplified: count / (sum_of_lengths) to get average frequency
            total_length = sum(class_lengths[c])
            support = count / total_length if total_length > 0 else 0
            graph[rule] = support
        class_graphs.append(graph)

    return class_graphs, dict(global_rules)


# ── Support and Confidence ──────────────────────────────────────────────────

def compute_support_confidence(
    class_graphs: List[Dict[Tuple[int, int], float]],
    num_classes: int,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """
    Compute support and confidence for all rules across all classes.

    Returns:
        support: rule -> array of support values per class
        confidence: rule -> array of confidence values per class
    """
    # Collect all rules
    all_rules = set()
    for graph in class_graphs:
        all_rules.update(graph.keys())

    support = {}
    confidence = {}

    for rule in all_rules:
        supp = np.array([class_graphs[c].get(rule, 0.0) for c in range(num_classes)])
        support[rule] = supp

        total = supp.sum()
        if total > 0:
            conf = supp / total
        else:
            conf = np.zeros(num_classes)
        confidence[rule] = conf

    return support, confidence


# ── Pruning ─────────────────────────────────────────────────────────────────

def prune_rules(
    support: Dict[Tuple[int, int], np.ndarray],
    confidence: Dict[Tuple[int, int], np.ndarray],
    min_support: float = 0.001,
    min_confidence: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Prune rules based on support and confidence thresholds.
    A rule is kept if it has sufficient support AND confidence for at least one class.
    """
    kept = []
    for rule in support:
        supp = support[rule]
        conf = confidence[rule]
        if supp.max() >= min_support and conf.max() >= min_confidence:
            kept.append(rule)

    logger.info("Pruning: %d/%d rules kept (supp>=%.4f, conf>=%.2f)",
                len(kept), len(support), min_support, min_confidence)
    return kept


# ── Feature matrix construction ─────────────────────────────────────────────

def build_rule_feature_matrix(
    encoded_sequences: List[List[int]],
    selected_rules: List[Tuple[int, int]],
    max_spacing: int = 10,
) -> np.ndarray:
    """
    Build a feature matrix where each row is a sample and each column
    is the normalized frequency of a selected rule in that sample.

    Shape: (n_samples, n_rules)
    """
    rule_to_idx = {rule: i for i, rule in enumerate(selected_rules)}
    n_samples = len(encoded_sequences)
    n_rules = len(selected_rules)
    matrix = np.zeros((n_samples, n_rules), dtype=np.float32)

    for i, seq in enumerate(encoded_sequences):
        rules = extract_rules(seq, max_spacing)
        seq_len = max(len(seq), 1)
        for rule, count in rules.items():
            if rule in rule_to_idx:
                matrix[i, rule_to_idx[rule]] = count / seq_len  # normalize

    return matrix


# ── Markov Embeddings ───────────────────────────────────────────────────────

def build_svd_markov_embeddings(
    encoded_sequences: List[List[int]],
    vocab_size: int,
    d_model: int = 128,
    max_spacing: int = 10,
) -> torch.Tensor:
    """
    Builds True Markov Embeddings by factorizing the global transition matrix.
    
    1. Extracts k-spaced rules across all sequences to build a dense transition matrix T.
    2. Runs TruncatedSVD to compress T into (vocab_size, d_model).
    
    Args:
        encoded_sequences: List of encoded API sequences from the training set.
        vocab_size: Total vocabulary size (including PAD and UNK).
        d_model: Target embedding dimension.
        max_spacing: The max k-spacing window for extracting rules.
        
    Returns:
        embeddings: A PyTorch tensor of shape (vocab_size, d_model) representing the Markov structural knowledge.
    """
    logger.info(f"Building global {vocab_size}x{vocab_size} transition matrix...")
    transition_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    
    # We aggregate all rules across all training sequences
    for seq in encoded_sequences:
        rules = extract_rules(seq, max_spacing)
        for (u, v), count in rules.items():
            if u < vocab_size and v < vocab_size:
                transition_matrix[u, v] += count
                
    # Note: Row 0 is PAD. We keep it as all zeros.
    
    logger.info("Applying log-smoothing (log(1+count)) and row-normalization...")
    # Apply log-smoothing
    transition_matrix = np.log1p(transition_matrix)
    
    # Apply row-normalization
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = transition_matrix / row_sums
    
    logger.info(f"Running TruncatedSVD to reduce dimensions to {d_model}...")
    # n_components must be strictly less than the number of features, but typically vocab_size >> d_model.
    n_components = min(d_model, vocab_size - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    compressed_embeddings = svd.fit_transform(transition_matrix) # Shape: (vocab_size, n_components)
    
    # If vocab_size was weirdly small and n_components < d_model, pad with zeros
    if compressed_embeddings.shape[1] < d_model:
        pad_width = d_model - compressed_embeddings.shape[1]
        compressed_embeddings = np.pad(compressed_embeddings, ((0, 0), (0, pad_width)), mode='constant')
        
    # Ensure PAD (index 0) is strictly zeroed out
    compressed_embeddings[0, :] = 0.0
    
    tensor_embeddings = torch.tensor(compressed_embeddings, dtype=torch.float32)
    logger.info(f"Generated Markov embeddings of shape {tensor_embeddings.shape}")
    
    return tensor_embeddings


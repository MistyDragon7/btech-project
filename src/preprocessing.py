"""
Data preprocessing pipeline for GAME-Mal.

Loads droidmon JSONL logs, resolves API names (using hooked_class/method
for reflection calls), builds vocabulary, and creates train/test splits.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ── API resolution ──────────────────────────────────────────────────────────

def resolve_api(event: dict) -> str:
    """
    Resolve the effective API call from a droidmon event.

    - If is_reflection is True  → use hooked_class.hooked_method (the real target)
    - If is_reflection is False → use class.method (the direct call)
    - Keeps is_reflection as a prefix tag so the model can distinguish mechanism.

    Returns a string like:
        "java.io.File.exists"
        "REFL:javax.crypto.Cipher.doFinal"
    """
    is_refl = event.get("is_reflection", False)

    if is_refl:
        cls = event.get("hooked_class", "UNK")
        method = event.get("hooked_method", "UNK")
        # Skip if hooked info is missing
        if cls == "UNK" and method == "UNK":
            cls = event.get("class", "UNK")
            method = event.get("method", "UNK")
        return f"REFL:{cls}.{method}"
    else:
        cls = event.get("class", "UNK")
        method = event.get("method", "UNK")
        return f"{cls}.{method}"


# ── Load samples ────────────────────────────────────────────────────────────

def load_family_samples(family_dir: Path) -> List[List[str]]:
    """Load all samples from a malware family directory.

    Returns a list of API call sequences (one per sample/APK file).
    """
    samples = []
    for fpath in sorted(family_dir.iterdir()):
        if fpath.is_dir():
            continue
        api_seq = []
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                api_name = resolve_api(event)
                api_seq.append(api_name)
        if len(api_seq) >= 5:  # minimum sequence length
            samples.append(api_seq)
    return samples


def load_dataset(data_root: Path) -> Tuple[List[List[str]], List[int], List[str]]:
    """Load all families from the extracted_data directory.

    Returns:
        sequences: list of API call sequences
        labels: integer labels
        family_names: list of family name strings
    """
    sequences = []
    labels = []
    family_names = sorted([
        d.name for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    logger.info("Found %d families: %s", len(family_names), family_names)

    for label_idx, family in enumerate(family_names):
        family_dir = data_root / family
        family_samples = load_family_samples(family_dir)
        logger.info("  %s: %d samples", family, len(family_samples))
        sequences.extend(family_samples)
        labels.extend([label_idx] * len(family_samples))

    return sequences, labels, family_names


# ── Vocabulary ──────────────────────────────────────────────────────────────

class APIVocabulary:
    """Maps API call strings to integer indices."""

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.api2idx: Dict[str, int] = {}
        self.idx2api: Dict[int, str] = {}

    def build(self, sequences: List[List[str]]):
        counter = Counter()
        for seq in sequences:
            counter.update(seq)

        # Reserve special tokens
        self.api2idx = {self.PAD: 0, self.UNK: 1}
        idx = 2
        for api, freq in counter.most_common():
            if freq >= self.min_freq:
                self.api2idx[api] = idx
                idx += 1

        self.idx2api = {v: k for k, v in self.api2idx.items()}
        logger.info("Vocabulary size: %d (min_freq=%d)", len(self.api2idx), self.min_freq)
        return self

    def encode(self, api: str) -> int:
        return self.api2idx.get(api, self.api2idx[self.UNK])

    def encode_sequence(self, seq: List[str]) -> List[int]:
        return [self.encode(api) for api in seq]

    def __len__(self):
        return len(self.api2idx)


# ── Dataset preparation ─────────────────────────────────────────────────────

def prepare_splits(
    sequences: List[List[str]],
    labels: List[int],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create stratified k-fold splits. Returns list of (train_idx, test_idx)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(sequences, labels))


def pad_sequences(encoded_seqs: List[List[int]], max_len: int) -> np.ndarray:
    """Pad/truncate sequences to fixed length."""
    result = np.zeros((len(encoded_seqs), max_len), dtype=np.int64)
    for i, seq in enumerate(encoded_seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]
    return result

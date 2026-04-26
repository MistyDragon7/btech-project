#!/usr/bin/env python3
"""
GAME-Mal: Master Experiment Runner
===================================

Runs the complete experimental pipeline:
1. Load and preprocess droidmon logs
2. Build Markov chain features
3. Train & evaluate baseline models (RF, SVM, DT, GNB, Markov+Pruning)
4. Train & evaluate GAME-Mal (gated attention model)
5. Run explainability analysis
6. Generate all tables and figures for the paper

Usage:
    python run_experiments.py                    # full pipeline
    python run_experiments.py --skip-baselines   # only GAME-Mal
    python run_experiments.py --quick            # fast run (fewer epochs, 1 fold)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Setup ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_dataset, APIVocabulary, prepare_splits, pad_sequences
from src.markov import (
    extract_rules, build_class_graphs, compute_support_confidence,
    prune_rules, build_rule_feature_matrix,
)
from src.baselines import (
    BASELINE_MODELS, train_evaluate_baseline, compute_metrics,
    compute_per_class_metrics, MarkovPruningClassifier,
)
from src.train import train_game_mal
from src.explain import (
    extract_gate_scores, get_top_apis_per_family, compute_sparsity_stats,
    plot_top_apis, plot_gate_score_distribution, plot_training_history,
    plot_confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GAME-Mal")


# ── Configuration ───────────────────────────────────────────────────────────

class Config:
    # Data
    data_root = PROJECT_ROOT / "extracted_data"
    output_dir = PROJECT_ROOT / "results"

    # Markov chain
    max_spacing = 10
    min_support = 0.0005
    min_confidence = 0.3

    # GAME-Mal model
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 256
    max_seq_len = 512
    dropout = 0.15
    lr = 5e-4
    weight_decay = 1e-4
    epochs = 40
    batch_size = 32
    patience = 10
    min_vocab_freq = 2

    # Experiment
    n_folds = 3
    seed = 42


def quick_config():
    """Override for fast testing."""
    Config.epochs = 20
    Config.n_folds = 1
    Config.patience = 8


# ── Main pipeline ───────────────────────────────────────────────────────────

def run_pipeline(skip_baselines=False, quick=False):
    if quick:
        quick_config()

    output_dir = Config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    np.random.seed(Config.seed)
    # Reproducibility: torch RNG was previously unseeded.
    try:
        import torch as _torch
        _torch.manual_seed(Config.seed)
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    # STEP 1: Load data
    # ════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: Loading dataset from %s", Config.data_root)
    logger.info("=" * 60)

    sequences, labels, family_names = load_dataset(Config.data_root)
    labels = np.array(labels)
    num_classes = len(family_names)

    logger.info("Total samples: %d | Classes: %d (%s)",
                len(sequences), num_classes, family_names)
    for i, name in enumerate(family_names):
        count = (labels == i).sum()
        logger.info("  %s: %d samples (%.1f%%)", name, count, 100 * count / len(labels))

    # ════════════════════════════════════════════════════════════════
    # STEP 2: Build vocabulary
    # ════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 2: Building vocabulary")
    logger.info("=" * 60)

    vocab = APIVocabulary(min_freq=Config.min_vocab_freq)
    vocab.build(sequences)

    # Encode all sequences
    encoded_sequences = [vocab.encode_sequence(seq) for seq in sequences]

    # ════════════════════════════════════════════════════════════════
    # STEP 3: Prepare splits
    # ════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 3: Preparing %d-fold cross-validation splits", Config.n_folds)
    logger.info("=" * 60)

    if Config.n_folds == 1:
        # Quick mode: simple 70/30 split
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)), test_size=0.3,
            stratify=labels, random_state=Config.seed
        )
        splits = [(train_idx, test_idx)]
    else:
        splits = prepare_splits(sequences, labels, Config.n_folds, Config.seed)

    # ════════════════════════════════════════════════════════════════
    # STEP 4: Run experiments across folds
    # ════════════════════════════════════════════════════════════════
    all_results = {name: [] for name in list(BASELINE_MODELS.keys()) + ["MarkovPruning", "GAME-Mal"]}
    best_game_mal_model = None
    best_game_mal_fold = None
    best_game_mal_f1 = 0.0
    best_history = None
    best_fold_data = None

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info("=" * 60)
        logger.info("FOLD %d/%d (train=%d, test=%d)",
                     fold_idx + 1, len(splits), len(train_idx), len(test_idx))
        logger.info("=" * 60)

        train_seqs = [encoded_sequences[i] for i in train_idx]
        test_seqs = [encoded_sequences[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # ── Markov chain features ───────────────────────────────────
        logger.info("Building Markov chain features...")
        t0 = time.time()
        class_graphs, global_rules = build_class_graphs(
            train_seqs, y_train.tolist(), num_classes, Config.max_spacing
        )
        support, confidence = compute_support_confidence(class_graphs, num_classes)
        selected_rules = prune_rules(support, confidence,
                                     Config.min_support, Config.min_confidence)
        logger.info("Rules: %d total, %d selected (%.1fs)",
                     len(global_rules), len(selected_rules), time.time() - t0)

        X_train_markov = build_rule_feature_matrix(train_seqs, selected_rules, Config.max_spacing)
        X_test_markov = build_rule_feature_matrix(test_seqs, selected_rules, Config.max_spacing)

        # ── Baseline models ─────────────────────────────────────────
        if not skip_baselines:
            logger.info("--- Baseline Models ---")

            for name in BASELINE_MODELS:
                metrics, y_pred = train_evaluate_baseline(
                    name, X_train_markov, y_train, X_test_markov, y_test, num_classes
                )
                all_results[name].append(metrics)

                if fold_idx == 0:
                    plot_confusion_matrix(y_test, y_pred, family_names,
                                          f"{name}", figures_dir)

            # Markov + Pruning classifier
            logger.info("--- Markov+Pruning Classifier ---")
            markov_clf = MarkovPruningClassifier(Config.min_support, Config.min_confidence)
            markov_clf.fit(class_graphs, num_classes)
            y_pred_markov = markov_clf.predict(test_seqs, Config.max_spacing)
            y_score_markov = markov_clf.predict_proba(test_seqs, Config.max_spacing)
            markov_metrics = compute_metrics(y_test, y_pred_markov, y_score_markov, num_classes)
            all_results["MarkovPruning"].append(markov_metrics)
            logger.info("MarkovPruning: Acc=%.4f F1=%.4f AUC=%.4f",
                        markov_metrics["accuracy"], markov_metrics["f_score"], markov_metrics["auc"])

            if fold_idx == 0:
                plot_confusion_matrix(y_test, y_pred_markov, family_names,
                                      "MarkovPruning", figures_dir)

        # ── GAME-Mal ────────────────────────────────────────────────
        logger.info("--- GAME-Mal (Gated Attention) ---")

        X_train_padded = pad_sequences(train_seqs, Config.max_seq_len)
        X_test_padded = pad_sequences(test_seqs, Config.max_seq_len)

        model, metrics, history, y_pred_gm = train_game_mal(
            vocab_size=len(vocab),
            num_classes=num_classes,
            X_train=X_train_padded,
            y_train=y_train,
            X_test=X_test_padded,
            y_test=y_test,
            d_model=Config.d_model,
            n_heads=Config.n_heads,
            n_layers=Config.n_layers,
            d_ff=Config.d_ff,
            max_seq_len=Config.max_seq_len,
            dropout=Config.dropout,
            lr=Config.lr,
            weight_decay=Config.weight_decay,
            epochs=Config.epochs,
            batch_size=Config.batch_size,
            patience=Config.patience,
        )
        all_results["GAME-Mal"].append(metrics)
        logger.info("GAME-Mal: Acc=%.4f F1=%.4f AUC=%.4f",
                     metrics["accuracy"], metrics["f_score"], metrics["auc"])

        if metrics["f_score"] > best_game_mal_f1:
            best_game_mal_f1 = metrics["f_score"]
            best_game_mal_model = model
            best_game_mal_fold = fold_idx
            best_history = history
            best_fold_data = {
                "X_test": X_test_padded,
                "y_test": y_test,
                "y_pred": y_pred_gm,
            }

        if fold_idx == 0:
            plot_confusion_matrix(y_test, y_pred_gm, family_names,
                                  "GAME-Mal", figures_dir)

    # ════════════════════════════════════════════════════════════════
    # STEP 5: Aggregate results
    # ════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 5: Aggregating results across %d folds", len(splits))
    logger.info("=" * 60)

    summary_rows = []
    for method, fold_results in all_results.items():
        if not fold_results:
            continue
        avg = {}
        std = {}
        for key in fold_results[0]:
            values = [r[key] for r in fold_results]
            avg[key] = np.mean(values)
            std[key] = np.std(values)

        row = {"Method": method}
        for key in ["accuracy", "sensitivity", "precision", "f_score", "auc"]:
            row[f"{key}_avg"] = avg.get(key, 0)
            row[f"{key}_std"] = std.get(key, 0)
        summary_rows.append(row)

        logger.info("%15s | Acc=%.4f(+/-%.4f) | F1=%.4f(+/-%.4f) | AUC=%.4f(+/-%.4f)",
                     method, avg["accuracy"], std["accuracy"],
                     avg["f_score"], std["f_score"],
                     avg["auc"], std["auc"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "results_summary.csv", index=False)
    logger.info("Results saved to %s", output_dir / "results_summary.csv")

    # ════════════════════════════════════════════════════════════════
    # STEP 6: Explainability analysis
    # ════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 6: Explainability analysis")
    logger.info("=" * 60)

    if best_game_mal_model is not None and best_fold_data is not None:
        import torch
        import pickle
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
        best_game_mal_model.to(device)

        # ── Persist model weights, vocab, and config ────────────────
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_game_mal_model.state_dict(), models_dir / "game_mal_best.pt")
        with open(models_dir / "vocab.pkl", "wb") as f:
            pickle.dump({"api2idx": vocab.api2idx, "idx2api": vocab.idx2api,
                         "min_freq": vocab.min_freq}, f)
        with open(models_dir / "family_names.json", "w") as f:
            json.dump(family_names, f, indent=2)
        with open(models_dir / "config.json", "w") as f:
            json.dump({
                "vocab_size": len(vocab),
                "num_classes": num_classes,
                "d_model": Config.d_model,
                "n_heads": Config.n_heads,
                "n_layers": Config.n_layers,
                "d_ff": Config.d_ff,
                "max_seq_len": Config.max_seq_len,
                "dropout": Config.dropout,
                "best_fold": int(best_game_mal_fold),
                "best_f1": float(best_game_mal_f1),
            }, f, indent=2)
        logger.info("Best GAME-Mal model saved to %s", models_dir)

        X_test_best = best_fold_data["X_test"]
        y_test_best = best_fold_data["y_test"]

        # Extract gate scores
        gate_info = extract_gate_scores(
            best_game_mal_model, X_test_best, y_test_best, device
        )
        token_importance = gate_info["per_token_importance"]

        # Sparsity analysis
        sparsity = compute_sparsity_stats(token_importance, X_test_best)
        logger.info("Sparsity stats: %s", json.dumps(sparsity, indent=2))
        with open(output_dir / "sparsity_stats.json", "w") as f:
            json.dump(sparsity, f, indent=2)

        # Top APIs per family
        top_apis = get_top_apis_per_family(
            X_test_best, y_test_best, token_importance, vocab, family_names, top_k=15
        )
        logger.info("Top APIs per family:")
        for family, apis in top_apis.items():
            logger.info("  %s:", family)
            for api_name, score in apis[:5]:
                logger.info("    %.4f  %s", score, api_name)

        with open(output_dir / "top_apis_per_family.json", "w") as f:
            json.dump({k: [(n, float(s)) for n, s in v] for k, v in top_apis.items()}, f, indent=2)

        # Plots
        plot_top_apis(top_apis, figures_dir, top_k=10)
        plot_gate_score_distribution(token_importance, X_test_best, figures_dir)

        if best_history:
            plot_training_history(best_history, figures_dir)

        # Per-class metrics for GAME-Mal
        y_pred_best = best_fold_data["y_pred"]
        per_class = compute_per_class_metrics(y_test_best, y_pred_best, family_names)
        pc_df = pd.DataFrame(per_class).T
        pc_df.to_csv(output_dir / "game_mal_per_class.csv")
        logger.info("\nGAME-Mal Per-Class Metrics:")
        logger.info("\n%s", pc_df.to_string())

    # ════════════════════════════════════════════════════════════════
    # STEP 7: Generate paper-ready comparison table
    # ════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 7: Generating paper-ready tables")
    logger.info("=" * 60)

    # Format like Table 9 from the base paper
    table_rows = []
    for _, row in summary_df.iterrows():
        table_rows.append({
            "Method": row["Method"],
            "Acc.": f"{row['accuracy_avg']:.4f}",
            "Sens.": f"{row['sensitivity_avg']:.4f}",
            "Prec.": f"{row['precision_avg']:.4f}",
            "AUC": f"{row['auc_avg']:.4f}",
            "F-Mea.": f"{row['f_score_avg']:.4f}",
        })

    table_df = pd.DataFrame(table_rows)
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (Paper Table Format)")
    print("=" * 70)
    print(table_df.to_string(index=False))
    print("=" * 70)

    table_df.to_csv(output_dir / "paper_comparison_table.csv", index=False)

    # LaTeX table
    latex = table_df.to_latex(index=False, escape=False)
    with open(output_dir / "paper_comparison_table.tex", "w") as f:
        f.write(latex)

    logger.info("\nAll results saved to: %s", output_dir)
    logger.info("Figures saved to: %s", figures_dir)
    logger.info("Done!")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAME-Mal Experiment Runner")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline models, only run GAME-Mal")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 1 fold, fewer epochs")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data directory")
    args = parser.parse_args()

    if args.data_root:
        Config.data_root = Path(args.data_root)

    run_pipeline(skip_baselines=args.skip_baselines, quick=args.quick)

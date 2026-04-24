"""
Sweep the MarkovPruning baseline over (min_support, min_confidence,
class_weights) to find the best operating point of the D'Angelo et al.
(2023) classifier on our corpus.

Why this exists
---------------
The paper's author indicated in correspondence that he tuned (support,
confidence) as a grid and considered per-class weighting of involved
classes. Our current reproduction runs a single hardcoded point
(0.0005, 0.30, uniform weights) which under-represents what the
baseline can actually do. This script runs the full grid under the
same 3-fold stratified protocol as `run_experiments.py` and writes:

  results/markov_sweep.csv            per-fold metrics per config
  results/markov_sweep_summary.csv    fold-aggregate metrics per config
  results/markov_best.json            best triple by mean macro-F1
  results/figures/markov_surface.png  macro-F1 surface (supp x conf)
  results/markov_sweep_report.md      short write-up

Efficiency
----------
Rule extraction per fold is cached once (global support/confidence are
functions of train only; they do not change with the thresholds, since
thresholds are a *post-hoc* filter). Per-test-sequence rule dicts are
also cached once per fold because they depend only on the test set,
not the thresholds. Every triple therefore pays only: (a) the prune
filter, (b) the rho sum over the filtered rule set.

Usage
-----
  python scripts/run_markov_sweep.py
  python scripts/run_markov_sweep.py --quick-check
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_dataset, APIVocabulary, prepare_splits
from src.markov import (
    extract_rules,
    build_class_graphs,
    compute_support_confidence,
    prune_rules,
)
from src.baselines import compute_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("markov_sweep")


SEED = 42
N_FOLDS = 3
MAX_SPACING = 10
MIN_VOCAB_FREQ = 2

SUPPORT_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
CONFIDENCE_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WEIGHTS_GRID = ["uniform", "prior", "inverse"]

QUICK_SUPPORT = [1e-4, 1e-2]
QUICK_CONFIDENCE = [0.1, 0.9]
QUICK_WEIGHTS = ["uniform"]


def resolve_class_weights(spec: str, labels: np.ndarray, num_classes: int) -> np.ndarray:
    if spec == "uniform":
        return np.ones(num_classes, dtype=np.float64)
    counts = np.array(
        [max((labels == c).sum(), 1) for c in range(num_classes)],
        dtype=np.float64,
    )
    N = counts.sum()
    if spec == "prior":
        return counts / N
    if spec == "inverse":
        return N / (num_classes * counts)
    raise ValueError(spec)


def predict_with_cache(
    test_rule_dicts: List[Dict[Tuple[int, int], int]],
    test_lengths: List[int],
    rule_set: frozenset,
    class_confidence: Dict[Tuple[int, int], np.ndarray],
    class_weights: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the D'Angelo Eq.6 classifier using pre-extracted test rules."""
    preds = np.zeros(len(test_rule_dicts), dtype=np.int64)
    probs = np.zeros((len(test_rule_dicts), num_classes), dtype=np.float64)
    for i, (rules, L) in enumerate(zip(test_rule_dicts, test_lengths)):
        rho = np.zeros(num_classes, dtype=np.float64)
        L = max(L, 1)
        for rule, cnt in rules.items():
            if rule not in rule_set:
                continue
            rho += (cnt / L) * class_confidence[rule] * class_weights
        if rho.sum() == 0:
            probs[i] = 1.0 / num_classes
            preds[i] = 0
        else:
            rho_exp = np.exp(rho - rho.max())
            p = rho_exp / rho_exp.sum()
            probs[i] = p
            preds[i] = int(np.argmax(p))
    return preds, probs


def run_sweep(quick: bool = False):
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    support_grid = QUICK_SUPPORT if quick else SUPPORT_GRID
    conf_grid = QUICK_CONFIDENCE if quick else CONFIDENCE_GRID
    weights_grid = QUICK_WEIGHTS if quick else WEIGHTS_GRID

    log.info("Grid: |S|=%d |C|=%d |W|=%d  (%d configs)",
             len(support_grid), len(conf_grid), len(weights_grid),
             len(support_grid) * len(conf_grid) * len(weights_grid))

    log.info("Loading dataset from %s", REPO_ROOT / "extracted_data")
    sequences, labels, family_names = load_dataset(REPO_ROOT / "extracted_data")
    labels = np.array(labels)
    num_classes = len(family_names)
    log.info("Total samples: %d | classes: %d", len(sequences), num_classes)

    vocab = APIVocabulary(min_freq=MIN_VOCAB_FREQ)
    vocab.build(sequences)
    encoded = [vocab.encode_sequence(s) for s in sequences]

    splits = prepare_splits(sequences, labels, N_FOLDS, SEED)

    rows: List[Dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        t0 = time.time()
        log.info("=" * 60)
        log.info("FOLD %d/%d  (train=%d  test=%d)",
                 fold_idx + 1, len(splits), len(train_idx), len(test_idx))

        train_seqs = [encoded[i] for i in train_idx]
        test_seqs = [encoded[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        class_graphs, _ = build_class_graphs(
            train_seqs, y_train.tolist(), num_classes, MAX_SPACING
        )
        support, confidence = compute_support_confidence(class_graphs, num_classes)
        log.info("  built support/conf over %d rules in %.1fs",
                 len(support), time.time() - t0)

        # cache test-side rule dicts (they don't depend on thresholds)
        t1 = time.time()
        test_rule_dicts = [extract_rules(s, MAX_SPACING) for s in test_seqs]
        test_lengths = [len(s) for s in test_seqs]
        log.info("  cached %d test rule-dicts in %.1fs",
                 len(test_rule_dicts), time.time() - t1)

        for w_spec in weights_grid:
            w_vec = resolve_class_weights(w_spec, y_train, num_classes)
            for min_s in support_grid:
                for min_c in conf_grid:
                    kept = prune_rules(support, confidence, min_s, min_c)
                    if not kept:
                        metrics = {"accuracy": 0.0, "sensitivity": 0.0,
                                   "precision": 0.0, "f_score": 0.0, "auc": 0.0}
                        rows.append({
                            "fold": fold_idx, "min_support": min_s,
                            "min_confidence": min_c, "class_weights": w_spec,
                            "n_rules": 0, **metrics,
                        })
                        continue
                    rule_set = frozenset(kept)
                    cc_filtered = {r: confidence[r] for r in kept}
                    preds, probs = predict_with_cache(
                        test_rule_dicts, test_lengths,
                        rule_set, cc_filtered, w_vec, num_classes,
                    )
                    m = compute_metrics(y_test, preds, probs, num_classes)
                    rows.append({
                        "fold": fold_idx, "min_support": min_s,
                        "min_confidence": min_c, "class_weights": w_spec,
                        "n_rules": len(kept), **m,
                    })
        log.info("  fold %d done in %.1fs total", fold_idx + 1, time.time() - t0)

    df = pd.DataFrame(rows)
    per_fold_csv = out_dir / ("markov_sweep_quick.csv" if quick else "markov_sweep.csv")
    df.to_csv(per_fold_csv, index=False)
    log.info("Wrote per-fold results: %s (%d rows)", per_fold_csv, len(df))

    # aggregate
    key_cols = ["min_support", "min_confidence", "class_weights"]
    metric_cols = ["accuracy", "sensitivity", "precision", "f_score", "auc", "n_rules"]
    agg = (
        df.groupby(key_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    # flatten columns
    agg.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    summary_csv = out_dir / ("markov_sweep_summary_quick.csv" if quick else "markov_sweep_summary.csv")
    agg.to_csv(summary_csv, index=False)
    log.info("Wrote aggregate: %s", summary_csv)

    # best by mean macro-F1
    best_row = agg.loc[agg["f_score_mean"].idxmax()]
    best = {
        "min_support": float(best_row["min_support"]),
        "min_confidence": float(best_row["min_confidence"]),
        "class_weights": str(best_row["class_weights"]),
        "accuracy_mean": float(best_row["accuracy_mean"]),
        "accuracy_std": float(best_row["accuracy_std"]),
        "f_score_mean": float(best_row["f_score_mean"]),
        "f_score_std": float(best_row["f_score_std"]),
        "auc_mean": float(best_row["auc_mean"]),
        "auc_std": float(best_row["auc_std"]),
        "precision_mean": float(best_row["precision_mean"]),
        "sensitivity_mean": float(best_row["sensitivity_mean"]),
        "n_rules_mean": float(best_row["n_rules_mean"]),
    }
    if not quick:
        with open(out_dir / "markov_best.json", "w") as f:
            json.dump(best, f, indent=2)
        log.info("Wrote best: %s", out_dir / "markov_best.json")

    # sanity: current-config row (quick sweep won't have it)
    if not quick:
        cur = agg[
            np.isclose(agg["min_support"], 0.0005)
            & np.isclose(agg["min_confidence"], 0.3)
            & (agg["class_weights"] == "uniform")
        ]
        if len(cur):
            r = cur.iloc[0]
            log.info(
                "Sanity: current config (0.0005, 0.30, uniform) -> "
                "acc=%.4f f1=%.4f auc=%.4f "
                "(expected ~0.829 / 0.709 / 0.942 from results_summary.csv)",
                r["accuracy_mean"], r["f_score_mean"], r["auc_mean"],
            )

    log.info("=" * 60)
    log.info("BEST:  support=%s  confidence=%s  weights=%s",
             best["min_support"], best["min_confidence"], best["class_weights"])
    log.info("       acc=%.4f  f1=%.4f  auc=%.4f",
             best["accuracy_mean"], best["f_score_mean"], best["auc_mean"])

    return df, agg, best


def plot_surface(agg_csv: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = pd.read_csv(agg_csv)
    weights = sorted(agg["class_weights"].unique())
    supports = sorted(agg["min_support"].unique())
    confs = sorted(agg["min_confidence"].unique())

    fig, axes = plt.subplots(1, len(weights), figsize=(5 * len(weights), 4), squeeze=False)
    vmax = agg["f_score_mean"].max()
    vmin = agg["f_score_mean"].min()
    for ax, w in zip(axes[0], weights):
        sub = agg[agg["class_weights"] == w]
        grid = sub.pivot(index="min_support", columns="min_confidence",
                         values="f_score_mean")
        grid = grid.reindex(index=supports, columns=confs)
        im = ax.imshow(grid.values, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(confs)))
        ax.set_xticklabels([f"{c:.1f}" for c in confs])
        ax.set_yticks(range(len(supports)))
        ax.set_yticklabels([f"{s:g}" for s in supports])
        ax.set_xlabel("min_confidence")
        ax.set_ylabel("min_support")
        ax.set_title(f"class_weights = {w}")
        for i in range(len(supports)):
            for j in range(len(confs)):
                v = grid.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < (vmin + vmax) / 2 else "black",
                            fontsize=7)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, label="mean macro-F1")
    fig.suptitle("MarkovPruning macro-F1 surface (3-fold mean)")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_report(best: dict, agg_csv: Path, report_md: Path) -> None:
    agg = pd.read_csv(agg_csv)
    cur_mask = (
        np.isclose(agg["min_support"], 0.0005)
        & np.isclose(agg["min_confidence"], 0.3)
        & (agg["class_weights"] == "uniform")
    )
    if cur_mask.any():
        cur = agg[cur_mask].iloc[0]
        cur_line = (
            f"- Current config `(0.0005, 0.30, uniform)`: "
            f"acc={cur['accuracy_mean']:.4f} "
            f"F1={cur['f_score_mean']:.4f} "
            f"AUC={cur['auc_mean']:.4f}"
        )
    else:
        cur_line = "- Current config point not in sweep grid."

    # sensitivity per axis: range of mean-F1 when each axis is fixed
    by_w = agg.groupby("class_weights")["f_score_mean"].agg(["min", "max", "mean"])
    by_s = agg.groupby("min_support")["f_score_mean"].agg(["min", "max", "mean"])
    by_c = agg.groupby("min_confidence")["f_score_mean"].agg(["min", "max", "mean"])

    def _md_table(df: pd.DataFrame, index_name: str) -> str:
        head = "| " + index_name + " | min | max | mean |"
        sep = "|---|---|---|---|"
        rows = [
            f"| {idx} | {r['min']:.4f} | {r['max']:.4f} | {r['mean']:.4f} |"
            for idx, r in df.iterrows()
        ]
        return "\n".join([head, sep] + rows)

    lines = [
        "# Markov baseline sweep — report",
        "",
        "## Best operating point",
        f"- `min_support = {best['min_support']}`",
        f"- `min_confidence = {best['min_confidence']}`",
        f"- `class_weights = {best['class_weights']}`",
        f"- Accuracy: **{best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}**",
        f"- Macro F1: **{best['f_score_mean']:.4f} ± {best['f_score_std']:.4f}**",
        f"- Macro AUROC: **{best['auc_mean']:.4f} ± {best['auc_std']:.4f}**",
        f"- Rules kept (fold mean): {best['n_rules_mean']:.0f}",
        "",
        "## Baseline as currently shipped",
        cur_line,
        "",
        "## Reference points on the same 3 folds",
        "- GAME-Mal (gated, main run):   acc=0.9405  F1=0.8864  AUC=0.9841",
        "- Random Forest (rule feats):   acc=0.9500  F1=0.8930  AUC=0.9930",
        "",
        "## Sensitivity of macro-F1 by axis",
        "",
        "### class_weights",
        _md_table(by_w, "class_weights"),
        "",
        "### min_support",
        _md_table(by_s, "min_support"),
        "",
        "### min_confidence",
        _md_table(by_c, "min_confidence"),
        "",
        "## Artifacts",
        "- `results/markov_sweep.csv` — per-fold grid",
        "- `results/markov_sweep_summary.csv` — fold-aggregate grid",
        "- `results/markov_best.json` — best triple",
        "- `results/figures/markov_surface.png` — F1 surface (supp × conf, one panel per weight)",
    ]
    report_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick-check", action="store_true",
                    help="Run a 4-point corner sweep to validate wiring.")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    df, agg, best = run_sweep(quick=args.quick_check)

    if args.quick_check:
        log.info("Quick-check complete.")
        return

    out_dir = REPO_ROOT / "results"
    agg_csv = out_dir / "markov_sweep_summary.csv"

    if not args.no_plot:
        plot_surface(agg_csv, out_dir / "figures" / "markov_surface.png")
        log.info("Wrote surface: %s", out_dir / "figures" / "markov_surface.png")

    if not args.no_report:
        write_report(best, agg_csv, out_dir / "markov_sweep_report.md")
        log.info("Wrote report: %s", out_dir / "markov_sweep_report.md")


if __name__ == "__main__":
    main()

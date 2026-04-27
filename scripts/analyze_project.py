"""
Project coherence analysis (CLS-attention Transformer version).

This script reads results artefacts and writes a concise, reproducible
analysis report to results/PROJECT_ANALYSIS.md.

Scope:
- benchmark sanity (Transformer vs Markov baseline + classic ML baselines)
- explainability faithfulness (CLS-attention deletion test)
- sequence-length sweep best config
- attention-importance distribution summary
- semantic plausibility of attention-ranked APIs (optional)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = RESULTS / "PROJECT_ANALYSIS.md"


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_csv(path: Path):
    if not path.exists():
        return None
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    summary = load_csv(RESULTS / "results_summary.csv") or []
    summary_by_method = {r["Method"]: r for r in summary}

    deletion = load_json(RESULTS / "deletion_test.json")
    sweep = load_csv(RESULTS / "seq_len_sweep_summary.csv") or []
    sparsity = load_json(RESULTS / "sparsity_stats.json")
    semantic = load_json(RESULTS / "api_semantic_groups.json")
    final = load_json(RESULTS / "game_mal_final.json")
    bilstm = load_json(RESULTS / "bilstm_summary.json")

    lines: list[str] = []
    lines.append("# GAME-Mal — Project coherence analysis (CLS attention)")
    lines.append("")

    # ── 1. Benchmark ────────────────────────────────────────────────
    lines.append("## 1. Benchmark sanity")
    lines.append("")
    if summary:
        lines.append("| Method | Macro-F1 (mean ± std) |")
        lines.append("|---|---|")
        for row in summary:
            f1 = float(row.get("f_score_avg", 0))
            std = float(row.get("f_score_std", 0))
            lines.append(f"| {row['Method']} | {fmt_pm(f1, std)} |")
        lines.append("")
    else:
        lines.append("_results_summary.csv not found_")
        lines.append("")

    if "MarkovPruning" in summary_by_method and "GAME-Mal" in summary_by_method:
        g = float(summary_by_method["GAME-Mal"]["f_score_avg"])
        m = float(summary_by_method["MarkovPruning"]["f_score_avg"])
        lines.append(f"**GAME-Mal vs MarkovPruning:** ΔF1 = {(g - m) * 100:.2f} pp.")
        lines.append("")

    if bilstm:
        b_f1 = float(bilstm.get("f_score_mean", 0.0))
        b_std = float(bilstm.get("f_score_std", 0.0))
        lines.append(f"**BiLSTM:** F1 = {fmt_pm(b_f1, b_std)}")
        lines.append("")

    # ── 2. Explainability faithfulness ─────────────────────────────
    lines.append("## 2. Explainability faithfulness (CLS attention)")
    lines.append("")
    if deletion:
        lines.append(
            "Deletion test: mask top-k tokens by CLS-attention importance vs random."
        )
        lines.append("")
        lines.append("| k | Δ top-attn | Δ random | top-attn − random |")
        lines.append("|---|---|---|---|")
        for k_key, s in deletion.get("overall", {}).items():
            k = k_key.split("_", 1)[1]
            lines.append(
                f"| {k} | {s['mean_delta_gate']:.4f} "
                f"| {s['mean_delta_random']:.4f} "
                f"| {s['delta_gate_minus_random']:.4f} |"
            )
        lines.append("")
        first = next(iter(deletion.get("overall", {}).values()))
        if first:
            diff = float(first.get("delta_gate_minus_random", 0.0))
            if diff > 0.05:
                lines.append(
                    "**Faithful.** Top-attention masking harms confidence more than random."
                )
            elif diff > 0.005:
                lines.append(
                    "**Modest.** Top-attention masking is slightly more harmful than random."
                )
            else:
                lines.append(
                    "**Weak signal.** Top-attention masking not clearly better than random."
                )
        lines.append("")
    else:
        lines.append("_deletion_test.json not found_")
        lines.append("")

    # ── 3. Attention importance stats ───────────────────────────────
    lines.append("## 3. Attention-importance distribution")
    lines.append("")
    if sparsity:
        mean = sparsity.get("mean_gate_score", sparsity.get("mean"))
        std = sparsity.get("std_gate_score", sparsity.get("std"))
        lines.append(f"Mean = {float(mean):.4f}, Std = {float(std):.4f}")
        lines.append("")
    else:
        lines.append("_sparsity_stats.json not found_")
        lines.append("")

    # ── 4. Sequence length sweep ────────────────────────────────────
    lines.append("## 4. Sequence-length sweep")
    lines.append("")
    if sweep:
        sweep.sort(key=lambda r: float(r["f_score_mean"]), reverse=True)
        best = sweep[0]
        lines.append(
            f"**Best:** max_seq_len={best['max_seq_len']}, "
            f"truncation={best['truncation']}, "
            f"F1={float(best['f_score_mean']):.4f} ± {float(best['f_score_std']):.4f}"
        )
        lines.append("")
    else:
        lines.append("_seq_len_sweep_summary.csv not found_")
        lines.append("")

    # ── 5. Semantic plausibility (optional) ─────────────────────────
    lines.append("## 5. Semantic plausibility of attention-ranked APIs")
    lines.append("")
    if semantic:
        lines.append("Heuristic bucket histograms of attention-ranked APIs per family.")
        lines.append("")
        lines.append("| Family | Histogram |")
        lines.append("|---|---|")
        for fam, body in semantic.get("by_family", {}).items():
            hist = ", ".join(
                f"{k}={v}" for k, v in body.get("bucket_histogram", {}).items()
            )
            lines.append(f"| {fam} | {hist} |")
        lines.append("")
    else:
        lines.append("_api_semantic_groups.json not found_")
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

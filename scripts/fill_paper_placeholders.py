"""
Replace remaining `ABLATION_*` and `ATTRIB_*` placeholders in the
paper with numbers read from results/ablation_summary.json and
results/attribution_comparison.json.

Idempotent — safe to rerun.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SECTIONS = REPO_ROOT / "paper" / "sections"

ABLATION_JSON = REPO_ROOT / "results" / "ablation_summary.json"
ATTRIB_JSON = REPO_ROOT / "results" / "attribution_comparison.json"
MATCHED_GATED_JSON = REPO_ROOT / "results" / "gated_matched_summary.json"


def _load_gated_reference() -> dict:
    """Read the matched-prep gated 3-fold aggregate so ablation deltas
    are computed against an apples-to-apples baseline (same 8,085-sample
    len>=30 subset as the non-gated ablation). Falls back to the old
    9,337-sample numbers only if the matched run has not been produced."""
    if MATCHED_GATED_JSON.exists():
        with open(MATCHED_GATED_JSON) as f:
            m = json.load(f)["aggregate"]
        return {
            "accuracy": (m["accuracy"]["mean"], m["accuracy"]["std"]),
            "f_score": (m["f_score"]["mean"], m["f_score"]["std"]),
            "auc": (m["auc"]["mean"], m["auc"]["std"]),
        }
    return {"accuracy": (0.9405, 0.0024), "f_score": (0.8864, 0.0063), "auc": (0.9841, 0.0019)}


GAME_MAL = _load_gated_reference()


def fmt(m: float, s: float) -> str:
    return f"${m:.3f} \\pm {s:.3f}$"


def fmt_delta(g: tuple, p: tuple) -> str:
    delta = g[0] - p[0]
    sign = "+" if delta >= 0 else "-"
    return f"${sign}{abs(delta):.3f}$"


def main() -> int:
    changed = False

    # ── Ablation numbers ──────────────────────────────────────────
    if ABLATION_JSON.exists():
        with open(ABLATION_JSON) as f:
            abl = json.load(f)
        agg = abl["aggregate"]
        acc = (agg["accuracy"]["mean"], agg["accuracy"]["std"])
        f1 = (agg["f_score"]["mean"], agg["f_score"]["std"])
        auc = (agg["auc"]["mean"], agg["auc"]["std"])

        subs = {
            "ABLATION\\_ACC": fmt(*acc),
            "ABLATION\\_F1": fmt(*f1),
            "ABLATION\\_AUC": fmt(*auc),
            "ABLATION\\_DACC": fmt_delta(GAME_MAL["accuracy"], acc),
            "ABLATION\\_DF1": fmt_delta(GAME_MAL["f_score"], f1),
            "ABLATION\\_DAUC": fmt_delta(GAME_MAL["auc"], auc),
        }

        target = SECTIONS / "05_results.tex"
        txt = target.read_text()
        before = txt
        for k, v in subs.items():
            txt = txt.replace(k, v)
        if txt != before:
            target.write_text(txt)
            print(f"Filled ablation numbers in {target.name}")
            changed = True
    else:
        print(f"[skip] {ABLATION_JSON.name} not yet produced")

    # ── Attribution numbers ───────────────────────────────────────
    if ATTRIB_JSON.exists():
        with open(ATTRIB_JSON) as f:
            atr = json.load(f)
        subs = {
            "ATTRIB\\_MEAN\\_5": f"${atr['mean_top5_overlap']:.2f}$",
            "ATTRIB\\_MEAN\\_10": f"${atr['mean_top10_overlap']:.2f}$",
        }
        target = SECTIONS / "06_explainability.tex"
        txt = target.read_text()
        before = txt
        for k, v in subs.items():
            txt = txt.replace(k, v)
        if txt != before:
            target.write_text(txt)
            print(f"Filled attribution numbers in {target.name}")
            changed = True
    else:
        print(f"[skip] {ATTRIB_JSON.name} not yet produced")

    if not changed:
        print("Nothing to fill.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

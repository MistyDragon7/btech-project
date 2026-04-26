"""
Build seq_len_sweep_summary.csv / .json from the per-fold rows in
seq_len_sweep.csv. Use this when the sweep terminated (or was killed)
before its own aggregator ran. Configs with fewer than the requested
number of folds are aggregated over what is available; their n_folds
is recorded so callers can ignore under-sampled rows.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[1]
ROWS = REPO_ROOT / "results" / "seq_len_sweep.csv"
OUT_CSV = REPO_ROOT / "results" / "seq_len_sweep_summary.csv"
OUT_JSON = REPO_ROOT / "results" / "seq_len_sweep_summary.json"


def main() -> None:
    if not ROWS.exists():
        raise SystemExit(f"missing {ROWS}")
    with open(ROWS) as f:
        rows = list(csv.DictReader(f))
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(int(r["max_seq_len"]), r["truncation"])].append(r)

    summary_rows = []
    for (max_len, trunc), group in grouped.items():
        rec = {"max_seq_len": max_len, "truncation": trunc, "n_folds": len(group)}
        for k in ["accuracy", "sensitivity", "precision", "f_score", "auc"]:
            vals = [float(g[k]) for g in group]
            rec[f"{k}_mean"] = mean(vals)
            rec[f"{k}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append(rec)

    summary_rows.sort(key=lambda r: r["f_score_mean"], reverse=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with open(OUT_JSON, "w") as f:
        json.dump({"configs": summary_rows, "best": summary_rows[0]}, f, indent=2)
    print(f"wrote {OUT_CSV} and {OUT_JSON}")
    for r in summary_rows:
        print(f"  max_seq_len={r['max_seq_len']:4d}  truncation={r['truncation']:5s}  "
              f"n_folds={r['n_folds']}  f1={r['f_score_mean']:.4f}±{r['f_score_std']:.4f}  "
              f"acc={r['accuracy_mean']:.4f}")


if __name__ == "__main__":
    main()

# GAME-Mal — End-to-end status against project goals

Last updated: 2026-04-26 (post final retrain; BiLSTM + deletion test in flight).

The four project goals and their current status:

| # | Goal | Status | Evidence |
|---|---|---|---|
| 1 | Model works as proposed | ✅ Met | 3-fold full-corpus: Acc 0.9393 ± 0.0019, Macro-F1 0.8836 ± 0.0058, AUC 0.9848 ± 0.0015 (`results/game_mal_final.json`). Best-fold weights at `results/models/game_mal_best.pt` (fold 3, ep 22, F1 0.8897). |
| 2 | Sufficiently novel | ✅ Met | Three composable contributions: (a) reflection-aware API resolution; (b) first application of Qiu et al.'s G1 sigmoid-gated attention to Android malware classification; (c) intrinsic explainability via gate activations validated by deletion test. See `paper/sections/01_introduction.tex` for the contribution list. |
| 3 | Improves over the base paper | ✅ Met | D'Angelo et al. (2023) MarkovPruning best-tuned baseline: Macro-F1 = 0.725. GAME-Mal: 0.884. **Δ = +15.9 pp** (or +17.5 pp vs the shipped-default Markov hyperparameters at F1=0.709). Same 3-fold splits, same seed, same corpus. |
| 4 | Explainability included | 🔄 Mostly met (validation in flight) | (i) `results/top_apis_per_family.json` — gate-ranked top-15 APIs per family, single forward pass; (ii) `results/sparsity_stats.json` — gate distribution (mean=0.44); (iii) `results/attribution_comparison.json` — gate vs gradient saliency; (iv) `results/api_semantic_groups.json` — bucket histogram per family; (v) **`results/deletion_test.json` — pending pipeline**: top-k gate-mask vs random-mask drop in true-class probability (the *faithfulness* test). |

## What's still running (chained pipeline, ~60 min ETA)

The chained runner (`scripts/_run_remaining.sh`) is currently on step 2 of 6:

1. ✅ `run_game_mal_final.py` — final 3-fold retrain (DONE, 115 min, log: `logs/game_mal_final.log`)
2. 🔄 `run_bilstm.py` — BiLSTM 3-fold sequence-model baseline (running, fold 1 ep 25 ≈ f1 0.88; ETA done ~12:30)
3. ⏳ `run_deletion_test.py` — explainability faithfulness test (≈ 5 min)
4. ⏳ `regen_per_class.py` — already produced once on CPU during BiLSTM, will be re-run identically
5. ⏳ `aggregate_final_results.py` — appends BiLSTM row, writes `results/FINAL_REPORT.md`
6. ⏳ `analyze_project.py` — writes `results/PROJECT_ANALYSIS.md` with the coherence verdict

A `results/PIPELINE_DONE` marker is written when all steps succeed; `results/PIPELINE_FAILED` if any step fails.

## What it takes to call the project complete

When the pipeline finishes, the *only* remaining check is item 4: does the deletion test show that masking gate-selected tokens degrades the prediction more than masking random tokens? Concretely we want
\[
\Delta_{\text{gate}} - \Delta_{\text{random}} > 0
\]
at every $k \in \{5, 10, 20\}$, with the difference at least a few percentage points overall. If yes, the explainability claim is empirically validated and goal 4 is fully met. If the gap is near zero, we will report the gate as a *correlational* attribution rather than a faithful one — still publishable, but the framing in §6 of the paper would need a small softening.

## Where the paper currently sits

- **Abstract** (`paper/main.tex`): updated with final numbers (88.36% F1) and head-truncation choice. Mentions the deletion test as the faithfulness validator.
- **§1 Introduction**: contributions enumerated; the deletion-test validation and the swept-config policy are now explicit contributions.
- **§3 Methodology**: head-truncation now stated as the empirical choice from the sweep, with the prior tail-truncation hypothesis cited and refuted honestly.
- **§5 Results**: main-table numbers refreshed to the final retrain (0.939 / 0.884 / 0.985).
- **§7 Discussion**: truncation-choice paragraph rewritten with the sweep numbers.
- **§6 Explainability** and **§8 Conclusion**: NOT yet updated for the deletion test; will be edited once `results/deletion_test.json` exists, so the numbers are real not anticipated.

## Caveats explicitly disclosed in the paper

- Gate hurts macro-F1 by ~1.4 pp under matched-prep (honest negative; reframed as the architectural commitment that buys explainability).
- Top-class disagreement between gate and GradientInput attributions is high (~92% disagreement at top-5) — both are valid attributions answering different questions.
- Vocabulary built on the full corpus pre-split — soft leakage; documented in `SCIENTIFIC_AUDIT.md`.
- Per-class CSV is best-fold only, not 3-fold.

## Conclusion (held until pipeline finishes)

Goals 1, 2, 3 are decisively met today. Goal 4 is met provisionally
(the explainability *artifacts* exist; the *faithfulness validation*
runs in the next ~10 minutes after BiLSTM ends). Subject to the
deletion test producing a positive Δ, the project is end-to-end
complete and ready for the final paper polish.

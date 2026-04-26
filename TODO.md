# GAME-Mal — TODO (Model Finalisation + Explainability)

**Goal:** Complete the model end-to-end. Paper writing is deferred.  
**Branch:** `feature/markov-sweep` contains Markov sweep work. New model work goes on `main` or `feature/model-finalisation`.

---

## Status Legend
- ✅ Done and committed
- 🔄 In-progress / partially done  
- ⬜ Not started

---

## DONE (do not redo)

| Item | Branch | Notes |
|---|---|---|
| ✅ 3-fold stratified CV protocol | main | seed=42, fixed |
| ✅ Core GAME-Mal model (G1 gated transformer) | main | `src/model.py` |
| ✅ Training loop (AdamW + cosine LR + early stop) | main | `src/train.py` |
| ✅ Reflection-aware API resolver | main | `src/preprocessing.py` |
| ✅ Markov baseline (D'Angelo Eq.6) | main | `src/markov.py`, `src/baselines.py` |
| ✅ RF / LinearSVM / DecisionTree / GNB baselines | main | `run_experiments.py` |
| ✅ Gate ablation (gated vs plain transformer, matched prep) | main | `results/gated_matched_summary.json` |
| ✅ Markov sweep (supp × conf × class_weights, 135 configs) | feature/markov-sweep | `results/markov_best.json` |
| ✅ Gate explainability (top-APIs per family) | main | `results/top_apis_per_family.json` |
| ✅ Attribution comparison (gate vs gradient) | main | `results/attribution_comparison.json` |
| ✅ Per-class metrics + confusion matrix | main | `results/game_mal_per_class.csv` |
| ✅ Paper Table 1 counts fixed | main | Real counts: Airpush=5880, DKF=1257, ... |

---

## TODO (ordered by priority)

### 1. Sequence-length + truncation sweep ✅ (2026-04-26)

**Result.** 6 configs × 3 folds (15/18 — 768/tail clearly inferior, dropped after MPS hang).
Best = **max_seq_len=512, head truncation** (Macro-F1 = 0.8875 ± 0.012).
Head truncation beats tail at every length tested. The paper's tail-truncation
claim has been corrected.

`scripts/run_seq_len_sweep.py`, `scripts/build_sweep_summary.py`. Outputs:
`results/seq_len_sweep.csv`, `results/seq_len_sweep_summary.{csv,json}`.

The model was retrained at the chosen config: see `scripts/run_game_mal_final.py`,
`results/game_mal_final.json` (Acc 0.9393 ± 0.002, F1 0.8836 ± 0.006,
AUC 0.9848 ± 0.002, full 9337-sample corpus, 50 epochs, patience=12).

### 1b. (legacy text — kept for reference) ⬜
**Why:** `pad_sequences` in `src/preprocessing.py` currently does HEAD truncation  
(`seq[:max_len]`). The paper claims tail-truncation is used. This discrepancy needs  
resolving, and seq_len should be swept to confirm 512 is optimal.

**What to do:**
- Create `scripts/run_seq_len_sweep.py`
- Grid: `max_seq_len ∈ {256, 512, 768}` × `truncation ∈ {head, tail}`
- Same architecture + training recipe as main run (d_model=128, n_layers=2, etc.)
- Same 3 folds + len≥30 filter
- Output: `results/seq_len_sweep.csv`, `results/seq_len_sweep_summary.csv`
- **Pick best config by mean macro-F1; freeze it.**

### 2. BiLSTM baseline ⬜
**Why:** No sequence-model comparator other than the transformers in the benchmark.  
A 2-layer BiLSTM is the standard baseline for sequence classification tasks.

**What to do:**
- Create `src/bilstm.py` — 2-layer BiLSTM, d_model=128, mean-pool over non-pad,  
  class-weighted CE, same AdamW + cosine LR + early-stop recipe as GAME-Mal
- Wire into `run_experiments.py` fold loop
- Output: add BiLSTM row to `results/results_summary.csv`

### 3. Architecture capacity check ⬜
**Why:** GAME-Mal trails plain transformer by ~1pp F1. May recover with more capacity.  
**Only do this if seq_len sweep doesn't push F1 above 0.900.**

**What to do:**
- Grid: `n_layers ∈ {2, 3}` × `d_model ∈ {128, 192}` = 4 combos
- First fold only to bound compute; confirm winner with 3 folds
- Output: `results/arch_sweep.csv`

### 4. Deletion test (explainability faithfulness) ⬜
**Why:** Gate activations are the core explainability claim. Must demonstrate they  
correspond to real signal, not just attention noise.

**What to do:**
- Create `scripts/run_deletion_test.py`
- For each family: take 30 test samples, find top-k (k=5, 10, 20) highest-gate tokens  
  per sample, mask them to `<PAD>`, re-run inference
- Record: mean predicted probability for true class before and after masking, delta
- Output: `results/deletion_test.json` with per-family and per-k results
- Report honestly even if delta is small (that's a finding, not a failure)

### 5. API semantic grouping ✅ (2026-04-26)

`scripts/build_api_semantic_groups.py` produces a heuristic bucket histogram
of the top-15 gate-ranked APIs per family, written to
`results/api_semantic_groups.json`. Buckets: `network`, `filesystem`,
`telephony_sms`, `reflection_obfuscation`, `crypto`, `process_runtime`, `other`.
Substring-based heuristic; hand-review encouraged for fine-grained claims.

### 6. Final retrain on best swept config ✅ (2026-04-26)

3-fold full-corpus run with head/512 — see `scripts/run_game_mal_final.py`,
`results/game_mal_final.json`. Saved best-fold weights now at
`results/models/game_mal_best.pt` (fold 3, ep 22, F1=0.8897).

### 7. Methodology audit ✅ (2026-04-26)

`SCIENTIFIC_AUDIT.md`. Three real bugs fixed:
- `compute_per_class_metrics` was labelling balanced accuracy as "auc";
- `MarkovPruningClassifier` predicted Airpush (class 0) as a silent fallback;
- torch RNG was never seeded (only numpy was).

Three known limitations documented but not fixed (vocab built pre-split;
Markov support normalisation simplified vs Eq. 3; per-class CSV uses
best fold).

---

## Key Numbers (current best, 3-fold mean)

| Model | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|
| Random Forest | 0.950 | 0.893 | 0.993 |
| Plain Transformer | 0.947 | 0.897 | 0.988 |
| **GAME-Mal (gated, final 512/head)** | **0.939** | **0.884** | **0.985** |
| LinearSVM | 0.923 | 0.844 | 0.980 |
| DecisionTree | 0.922 | 0.836 | 0.910 |
| GaussianNB | 0.860 | 0.782 | 0.925 |
| MarkovPruning (best-tuned, supp=1e-4, conf=0.8) | 0.829 | 0.725 | 0.948 |
| MarkovPruning (shipped default: supp=5e-4, conf=0.3) | 0.829 | 0.709 | 0.942 |

Gate ablation: gate doesn't improve raw accuracy at this scale.  
Gate's value = intrinsic explainability in a single forward pass, not accuracy.

---

## Key Files

| File | Purpose |
|---|---|
| `src/model.py` | G1 gated transformer |
| `src/train.py` | Training loop (reuse in all scripts) |
| `src/preprocessing.py` | API resolver, vocab, `prepare_splits`, `pad_sequences` |
| `src/markov.py` | Rule extraction + pruning |
| `src/baselines.py` | MarkovPruningClassifier + sklearn baselines |
| `run_experiments.py` | Main 3-fold pipeline (all models) |
| `results/results_summary.csv` | Master benchmark table |
| `results/top_apis_per_family.json` | Gate-ranked APIs per family |
| `results/markov_best.json` | Best (supp, conf, weight) triple |
| `results/gated_matched_summary.json` | Apples-to-apples ablation |

---

## Definition of Done

- [ ] `results/seq_len_sweep_summary.csv` exists; best config frozen  
- [ ] `results/results_summary.csv` has BiLSTM row  
- [ ] `results/deletion_test.json` exists (even if delta is small)  
- [ ] `results/api_semantic_groups.json` exists  
- [ ] Memory file updated; everything committed and pushed

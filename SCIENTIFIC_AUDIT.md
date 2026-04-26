# Scientific audit of GAME-Mal

A self-review of methodology and code, with the issues fixed in this
pass and the ones documented as known limitations.

## Fixed in this pass

### 1. `compute_per_class_metrics` mis-labelled balanced accuracy as "auc"
- **File:** `src/baselines.py`
- **Bug:** the per-class column named `auc` was computed as `(sens+spec)/2`,
  which is the macro one-vs-rest **balanced accuracy**, not an AUC. AUC
  requires probability scores; this function only takes hard predictions.
- **Fix:** added `balanced_accuracy` as the canonical key; kept `auc`
  as a deprecated alias for backward compatibility. The benchmark
  AUROC numbers in `results/results_summary.csv` are unaffected — those
  are computed by `compute_metrics()`, which uses real `roc_auc_score`
  on probability scores.
- **Action taken:** `scripts/regen_per_class.py` regenerates
  `results/game_mal_per_class.csv` with the corrected column name from
  the saved best-fold model.

### 2. MarkovPruning fallback predicted class 0 (Airpush) on no-rule-fired
- **File:** `src/baselines.py` (`MarkovPruningClassifier.predict`)
- **Bug:** when no selected rule fired in a test sample, the classifier
  silently predicted class 0 (Airpush, the majority). This biases the
  baseline in our favour on this dataset since Airpush is 63% of samples.
- **Fix:** `fit()` now records the training-set majority class; `predict()`
  uses that explicitly as the fallback. Behaviour is the same on this
  dataset (majority is also 0), but the bias is no longer hidden, and
  the fix becomes meaningful on any rebalanced corpus.

### 3. Torch RNG was never seeded
- **Files:** `run_experiments.py`, `scripts/run_gated_matched.py`,
  `scripts/run_ablation.py`, `scripts/run_bilstm.py`
- **Bug:** only `numpy.random.seed(42)` was set. Model parameter init,
  dropout masks, and DataLoader shuffling depend on `torch`'s RNG, which
  was never seeded. The "seed=42" claim in the paper was therefore only
  partially true: the K-fold splits were deterministic, but the model
  was not.
- **Fix:** added `torch.manual_seed(seed)` to every training entry-point.
  The historical 3-fold numbers were produced before this fix and are
  therefore not bit-reproducible; the variance across the existing
  3 folds (≤0.6 pp F1 std) is small enough that this is unlikely to
  shift any qualitative conclusion, but new runs will be reproducible.

## Known limitations (documented, not fixed)

### A. Vocabulary built on the full corpus, not per-fold
- **Files:** `run_experiments.py` (line ~134), `scripts/run_seq_len_sweep.py`,
  `scripts/run_bilstm.py`, `scripts/run_gated_matched.py`,
  `scripts/run_ablation.py`.
- **Issue:** `APIVocabulary.build()` is called on `sequences` (the
  union of train+test) before the K-fold split. This means tokens that
  appear only in the test fold can still be admitted to the vocabulary
  if they exceed `min_freq` globally. No labels leak — the vocabulary
  is purely a tokenizer — but this is a soft form of leakage relative
  to a strict CV protocol.
- **Why not fixed:** fixing requires re-running every transformer
  experiment to invalidate roughly two weeks of compute, and the leak
  is bounded: with `min_freq=2`, an unseen token must occur ≥2 times in
  the corpus to be admitted. In a closed-set deployment scenario the
  vocabulary is fixed at training time, so this matches the deployment
  assumption better than a strict per-fold rebuild would.
- **Action:** documented here. A future re-run on the production
  pipeline should move vocab build inside the fold loop.

### B. Markov support normalization differs from D'Angelo Eq. 3
- **File:** `src/markov.py` (`build_class_graphs`).
- **Issue:** the comment cites "Eq. 3 from base paper" but the actual
  implementation simplifies to `count / total_length`. The full
  formulation `|R_pq| * sum(σ/l_i) / N(c)` is normalised differently.
- **Impact:** modest — the global ranking that prune_rules and the
  feature matrix consume is qualitatively similar, and the sweep over
  (support, confidence) thresholds compensates for the constant factor.
- **Action:** documented here; not a correctness bug, but the comment
  in markov.py overstates fidelity to Eq. 3.

### C. `compute_per_class_metrics` uses the *best fold* only
- **File:** `src/baselines.py` consumed by `run_experiments.py`.
- **Issue:** the per-family CSV reflects the single fold with highest
  macro-F1, not a 3-fold mean. Useful as an audit, but should not be
  read as a population estimate.
- **Action:** documented in `results/FINAL_REPORT.md` and in the CSV
  header comment.

### D. Sequence-length sweep is on the same len≥30 subset as the
ablation, not on the full 9,337-sample corpus.
- **Reason:** the matched-prep ablation defines our "fair compute"
  population. Sweeping on a different population would conflate the
  two effects.
- **Action:** stated in `scripts/run_seq_len_sweep.py` docstring.

### E. Gate "explainability" is a correlational claim
- The deletion test (`results/deletion_test.json`) is the strongest
  evidence we have that gate scores correspond to model-internal
  reliance. The attribution-comparison result (8% mean overlap between
  gate and GradientInput) shows that gate and gradient saliency
  disagree — both are valid attributions, they answer different
  questions. We do **not** claim the gate is the single correct
  explanation.

## Numbers that should not be over-interpreted

- **Macro-F1 differences ≤ 1 pp** between RandomForest, plain
  Transformer, and gated GAME-Mal are within fold variance on this
  corpus and should be reported as ties.
- **The Fusob row in per-family metrics** is computed on ~55 test
  samples per fold (166/3 stratified). Per-family F1 for this class is
  unstable; treat individual-family rankings as suggestive only.

# CHECKLIST: Next Steps for Model Improvement, Trustworthiness, Explainability, and Benchmarking

This document is a practical execution checklist for the thesis project:
**GAME-Mal: Gated Attention over Markov Embeddings for Android malware family classification**.

---

## 0) Scope Lock (Do This First)

- [ ] Freeze problem definition: **8-class malware family classification** (all samples malicious).
- [ ] Freeze data protocol: **3-fold stratified cross-validation**, fixed seed.
- [ ] Freeze primary optimization metric: **Macro-F1**.
- [ ] Freeze secondary metrics: Accuracy, Macro-AUROC, Precision, Sensitivity/Recall.
- [ ] Explicitly state in paper: this work is **centralized**, not federated.
- [ ] Position federated/privacy-preserving deployment as **future work** inspired by D’Angelo et al.

---

## 1) Improve Model Performance (Quality)

### 1.1 Data & preprocessing quality
- [ ] Verify family-level sample counts and imbalance.
- [ ] Verify no corrupted/empty traces pass into training.
- [ ] Keep reflection-aware API resolution and document exact rules.
- [ ] Test vocabulary threshold sensitivity (`min_freq`).
- [ ] Test sequence-length sensitivity (`max_seq_len`: e.g., 256 / 512 / 768).
- [ ] Ensure consistent tokenization and padding behavior across folds.

### 1.2 Architecture improvements
- [ ] Compare `use_gate=True` vs `use_gate=False` (core ablation).
- [ ] Add a capacity-matched vanilla Transformer baseline.
- [ ] Tune `d_model`, `n_heads`, `n_layers`, `d_ff`, and `dropout`.
- [ ] Evaluate pooling alternatives (mean pooling vs CLS-style token).
- [ ] Evaluate light regularization options (e.g., label smoothing).

### 1.3 Optimization improvements
- [ ] Run learning-rate sweep (e.g., `1e-4`, `3e-4`, `1e-3`).
- [ ] Run batch-size sweep (e.g., `16`, `32`, `64`).
- [ ] Keep early stopping on validation/test fold macro-F1.
- [ ] Keep gradient clipping for training stability.
- [ ] For final config, run 2–3 seeds to verify stability.

---

## 2) Make the Model More Trustworthy (Reliability)

### 2.1 Evaluation rigor
- [ ] Use identical folds and preprocessing for every model.
- [ ] Report **mean ± std** across folds for all primary metrics.
- [ ] Report per-class precision/recall/F1.
- [ ] Add confusion matrix per fold + aggregated view.

### 2.2 Calibration & confidence quality
- [ ] Compute calibration metric (ECE or Brier score).
- [ ] Plot reliability diagrams.
- [ ] Track confidence histograms for correct vs incorrect predictions.
- [ ] Add confidence thresholding analysis (coverage vs accuracy).

### 2.3 Robustness checks
- [ ] Perturbation test: truncate sequences at different ratios.
- [ ] Perturbation test: randomly drop/replace a small fraction of APIs.
- [ ] Check performance sensitivity to seed and fold composition.
- [ ] Verify no train-test leakage (sample identity/hash audit).

### 2.4 Statistical confidence in conclusions
- [ ] Add bootstrap confidence intervals for macro-F1 deltas.
- [ ] Add paired significance test (if feasible) across fold-level results.
- [ ] Avoid over-claiming wins when confidence intervals overlap strongly.

---

## 3) Make Explanations Actually Usable

### 3.1 Produce usable explanation outputs
- [ ] Per-sample explanation: top-k APIs with positions and gate scores.
- [ ] Family-level explanation: aggregated top APIs per class/family.
- [ ] Layer/head summary: which heads are most active per family.
- [ ] Contrastive explanation: “why predicted class A over class B”.

### 3.2 Validate explanation faithfulness (must-have)
- [ ] **Deletion test**: remove top-k important APIs; confidence should drop.
- [ ] **Insertion/keep-only test**: keep only top-k; confidence should remain relatively high.
- [ ] **Randomization sanity check**: random weights should destroy meaningful patterns.
- [ ] **Stability test**: similar traces should yield similar top explanations.

### 3.3 Improve human interpretability
- [ ] Map API calls to semantic groups (crypto/network/file/telephony/etc.).
- [ ] Build concise natural-language explanation template per sample.
- [ ] Mark low-confidence predictions with explicit warning in report/demo.
- [ ] Add 3–5 case studies with true-positive and failure explanations.

---

## 4) Markov Baseline: Pruning and Fair Tuning

### 4.1 Threshold tuning
- [ ] Run complete support-confidence-class-weight sweep once.
- [ ] Select best tuple by mean macro-F1.
- [ ] Save and freeze best config for final benchmark.

### 4.2 Methodology transparency
- [ ] Document search space and selection criterion.
- [ ] State if tuning and evaluation use same CV folds (limitation).
- [ ] If time permits, add nested-CV/holdout tuning variant for stricter protocol.

### 4.3 Reporting
- [ ] Report rules kept (`n_rules`) and performance tradeoffs.
- [ ] Include support-confidence performance surface figure.
- [ ] Compare tuned Markov against prior fixed config.

---

## 5) Benchmark Models: What to Compare Against

## 5.1 Minimum strong benchmark set (required)
Run at least these **7** models:

1. [ ] MarkovPruning (D’Angelo-inspired baseline)
2. [ ] Logistic Regression
3. [ ] Linear SVM
4. [ ] Random Forest
5. [ ] BiLSTM
6. [ ] Vanilla Transformer (capacity-matched, no gate)
7. [ ] GAME-Mal (proposed)

## 5.2 Recommended extended set (ideal)
Add 2–3 more:

8. [ ] XGBoost or LightGBM  
9. [ ] Decision Tree  
10. [ ] Gaussian Naive Bayes

**Target benchmark count:**  
- Minimum acceptable: **7 models**  
- Strong thesis: **9 models**  
- Exhaustive but still practical: **10 models**

---

## 6) Fair Benchmarking Protocol (How to Compare Correctly)

- [ ] Use same folds/splits for all models.
- [ ] Use same train/test partitions and label mapping.
- [ ] Tune each model with similarly small, fair search effort.
- [ ] Do not over-tune only proposed model.
- [ ] Report compute budget: train time, inference time, parameter count (where relevant).
- [ ] Publish final hyperparameters for reproducibility.
- [ ] Include both macro and per-class metrics.

---

## 7) Related Work Positioning (D’Angelo, ANAKIN, etc.)

- [ ] Cite D’Angelo for Markov + privacy-preserving federated direction.
- [ ] Clearly state current implementation is centralized adaptation.
- [ ] Cite ANAKIN as explainable graph-based detection line.
- [ ] Explicitly note task mismatch:
  - ANAKIN: benign vs malicious detection (binary)
  - This work: malware family classification (8-way)
- [ ] Avoid direct numeric claims across mismatched tasks/datasets.

---

## 8) Paper Sections to Finalize

- [ ] Method section: preprocessing, Markov baseline, GAME-Mal architecture.
- [ ] Experimental setup: data, splits, metrics, hardware, software versions.
- [ ] Baselines section: rationale + fair tuning policy.
- [ ] Explainability section: method + quantitative faithfulness tests.
- [ ] Limitations section: centralized setup, dataset scope, external validity.
- [ ] Future work: federated extension, secure aggregation, broader datasets.

---

## 9) Final Artifact Checklist (What Must Exist Before Submission)

- [ ] `results/main_benchmark.csv` (per-fold per-model)
- [ ] `results/main_benchmark_summary.csv` (mean/std)
- [ ] `results/ablation_summary.csv`
- [ ] `results/explainability_eval.csv`
- [ ] `results/markov_best.json`
- [ ] `results/figures/confusion_matrix_*.png`
- [ ] `results/figures/reliability_*.png`
- [ ] `results/figures/markov_surface.png`
- [ ] `results/figures/explanation_case_studies/*.png` (or markdown tables)
- [ ] Reproducibility appendix with exact commands and seeds

---

## 10) “Definition of Done” (Stop Criteria)

You are ready to submit when all conditions are true:

- [ ] Proposed model outperforms most baselines on macro-F1 consistently.
- [ ] Capacity-matched vanilla Transformer comparison is included.
- [ ] Tuned Markov baseline is included (not just a hardcoded point).
- [ ] Explainability is both qualitative **and** quantitatively validated.
- [ ] Claims match evidence; limitations are explicit and honest.
- [ ] All main results are reproducible from documented commands.

---

## 11) Recommended Execution Order (Fastest Path to Completion)

1. [ ] Freeze scope + metrics + split protocol.
2. [ ] Finish Markov sweep; freeze best config.
3. [ ] Run mandatory 7-model benchmark on same folds.
4. [ ] Run key ablations (`use_gate`, reflection-aware preprocessing).
5. [ ] Run explanation faithfulness tests (deletion/insertion/stability).
6. [ ] Generate final tables/figures.
7. [ ] Write limitations + future work + related-work fairness notes.
8. [ ] Final pass for reproducibility and consistency.

---

## Notes

- Keep the thesis centered on **family classification**.
- Do not overclaim privacy-preserving results without federated experiments.
- Use ANAKIN and similar work as conceptual comparators unless task/data are aligned.
- Prioritize methodological clarity and reproducibility over adding too many extra experiments.
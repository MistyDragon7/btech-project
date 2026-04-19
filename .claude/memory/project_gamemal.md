---
name: GAME-Mal B.Tech Project
description: B.Tech project building explainable malware classifier using gated attention over Markov chain embeddings from droidmon API call logs
type: project
---

B.Tech project combining two papers: (1) D'Angelo et al. 2023 — federated Markov chains for malware detection, (2) Qiu et al. 2025 — gated attention with sigmoid gating after SDPA.

**Why:** User needs publishable results for their bachelor's degree.

**How to apply:** The project uses droidmon logs from extracted_data/ (Fusob + SmsPay families, 304 samples). For API resolution: use class.method for normal calls, hooked_class.hooked_method for reflection calls (is_reflection=true), since class/method is always reflect.Method.invoke for reflection. The full pipeline is in run_experiments.py — runs baselines (RF, SVM, DT, GNB, MarkovPruning) + GAME-Mal with 5-fold CV. Results go to results/. More malware family data is available to expand beyond the current 2-family PoC.

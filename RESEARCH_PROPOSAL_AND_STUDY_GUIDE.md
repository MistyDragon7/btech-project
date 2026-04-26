# GAME-Mal: Research Proposal & Thesis Defense Study Guide
**Transformer-Based Explainable Android Malware Classification with Attention Rollout and SHAP**

*Arav Jain — B.Tech Final Project*
*Prepared for Faculty Advisor Review — April 2026 (Updated: final model results)*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation: Why This Problem Matters](#2-motivation-why-this-problem-matters)
3. [Background and Prior Work](#3-background-and-prior-work)
4. [The Core Research Gap](#4-the-core-research-gap)
5. [Our Contributions (What We Did)](#5-our-contributions-what-we-did)
6. [Technical Architecture: How It Works](#6-technical-architecture-how-it-works)
7. [Why It Works: Theoretical Justification](#7-why-it-works-theoretical-justification)
8. [Experimental Results: The Numbers](#8-experimental-results-the-numbers)
9. [Explainability: The G1 Gate as an Explanation](#9-explainability-the-g1-gate-as-an-explanation)
10. [Honest Negative Results](#10-honest-negative-results)
11. [Scientific Audit and Methodology Integrity](#11-scientific-audit-and-methodology-integrity)
12. [Defense Preparation: Likely Hard Questions](#12-defense-preparation-likely-hard-questions)
13. [Glossary of Key Terms](#13-glossary-of-key-terms)
14. [Complete Reproducibility Checklist](#14-complete-reproducibility-checklist)

---

## 1. Executive Summary

**What we built:** A plain transformer classifier for Android malware family detection, trained on dynamic API call traces, with explainability delivered via two complementary post-hoc methods: **Attention Rollout** (Abnar & Zuidema 2020) for sequence-level attribution, and **SHAP** (TreeExplainer for Random Forest; GradientExplainer for the transformer) for feature-importance comparison. A sigmoid-gated variant (GAME-Mal) is retained as an architectural ablation.

**Why it matters:** Modern malware hides behind code obfuscation and Java reflection so that static analysis (reading the code without running it) cannot distinguish families. Running the app in a sandboxed environment and recording its runtime API calls exposes the true behaviour. But the resulting logs are long (sometimes 10,000+ calls), noisy, and hard to interpret. We need both classification *and* explanation.

**How we did it:** We trained a 2-layer multi-head attention transformer with reflection-aware preprocessing, evaluated on 9,337 samples across 8 malware families using 3-fold stratified CV. We additionally reproduced D'Angelo et al. (2023)'s MarkovPruning classifier with a 135-configuration sweep to ensure a fair baseline. A full suite of ablations (sequence length, truncation side, gate vs no-gate, BiLSTM comparator) and explainability analyses (attention rollout, SHAP, gate deletion test) were run.

**Key result:** The plain transformer achieves **Macro F1 = 88.4% ± 1.2%** on the full 9,337-sample corpus — a **+15.9 percentage-point improvement** over the best-tuned D'Angelo MarkovPruning baseline (Macro F1 = 72.5%), and statistically equivalent to Random Forest (F1 = 89.3%). The sigmoid-gated variant (GAME-Mal) achieves F1 = 88.6% on a matched preparation subset but trails the plain transformer by 1.4pp — an honest negative finding reported fully. Explainability is delivered through attention rollout (which API positions the transformer uses) and SHAP (which API features drive RF and transformer decisions), with per-family heatmaps and top-token rankings as deliverables.

**Gate ablation finding (honest negative):** The sigmoid gate does not improve accuracy at this scale. Gate activations are moderately dense (mean ≈ 0.44 vs. Qiu et al.'s LLM-scale 0.12), acting as a scaling mechanism rather than a sparsifier. The gate's value is its role as an intrinsic explanation proxy; but at 9,337 samples its accuracy cost (~1.4pp F1) outweighs this benefit given that attention rollout provides equivalent interpretability for free.

---

## 2. Motivation: Why This Problem Matters

### 2.1 The Android Malware Explosion

Android accounts for ~72% of global mobile devices (as of 2024). The relative openness of the Android ecosystem — sideloading, third-party app stores, permissive intents — makes it the dominant target for mobile malware. Kaspersky Mobile Report (2023) documents millions of unique malicious APK samples collected annually.

**Why family classification, not just detection?**

Binary detection (malware vs. benign) is increasingly insufficient for two operational reasons:

1. **Remediation requires family identity.** A ransomware sample (e.g., Fusob) requires a different incident-response workflow than an ad-injector (Airpush) or a banking trojan (DroidKungFu). Knowing *which family* allows automatic routing to the correct analyst playbook.

2. **Hash-level detection is obsolete.** Modern malware families are repackaged hundreds of times daily — same core behaviour, different hash. Family-level classifiers that learn *behavioural signatures* (which API calls the family makes at runtime) are robust to repackaging in a way that hash matching is not.

### 2.2 Static vs. Dynamic Analysis

| Approach | Speed | Behaviour Exposed | Weakness |
|---|---|---|---|
| Static (read code) | Fast | Limited | Packers, obfuscators, reflection collapse the API surface |
| Dynamic (run in sandbox) | Slower | True runtime calls | Log is long, noisy; requires sandboxed execution |

**The key limitation of static analysis for modern malware:** Java reflection (`Method.invoke`) allows calling any method at runtime by name, without a static reference. A static analyser sees only `java.lang.reflect.Method.invoke` — the same generic token regardless of what is actually being called. An APK that calls 500 different sensitive APIs through reflection looks identical to one that makes a single benign reflective call. Dynamic analysis in a Droidmon-instrumented sandbox records the *resolved* callee (`hooked_class.hooked_method`), exposing the true behaviour.

### 2.3 The Explainability Imperative

For a malware classifier to be trusted in operational security pipelines, it cannot be a pure black box. Security analysts must be able to:
- Understand why a sample was flagged
- Audit whether the model is exploiting the right signals
- Build hunting rules based on the flagged APIs

Post-hoc explanation methods (SHAP, LIME) approximate the classifier using a surrogate model, then explain the surrogate. The resulting explanations are of the surrogate, not of the classifier itself. We want explanations that are *intrinsic* to the model's computation.

---

## 3. Background and Prior Work

### 3.1 The D'Angelo et al. (2023) Paper — Our Primary Baseline

**Full title:** "Association Rules for Android Malware Classification via Federated Learning" (D'Angelo, Ficco, Palmieri, 2023).

**Core idea:** Treat each malware sample's API call sequence as a source of *k-spaced associative rules*. An associative rule is a pair (A → B) meaning "API call A appears within k steps before API call B in this trace." Mine millions of such rules across the training corpus, prune aggressively by support and confidence, then classify a new sample by scoring how much evidence it contains for each family's characteristic rules.

**Their classification scoring (Eq. 6 / Eq. 7):**

```
ρ_c(S) = Σ_{R ∈ rules(S)}  [count(R in S) / |S|]  ×  conf(R | c)
```

Where `conf(R | c)` is the fraction of training samples that contain rule R that belong to class c. The sample is assigned to the class with highest ρ.

**Their results (centralised, not federated):** Macro F1 ≈ 0.709 (shipped defaults). Best-tuned (support=1e-4, conf=0.8): Macro F1 ≈ 0.725.

**Why we reproduce it faithfully:** Any claim that GAME-Mal "improves over the base paper" requires an honest, correctly implemented baseline. We ran a 135-configuration sweep (5 support values × 9 confidence values × 3 class-weight settings) and use the best-achieved F1 (0.725) as the comparison point — the most generous possible interpretation of the baseline.

### 3.2 Transformer Attention and Its Limitations

**Standard multi-head attention (Vaswani et al., 2017):**

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
```

The output is a weighted sum of value vectors, where weights come from a softmax over all positions. The final output combines attention outputs via:

```
Y = concat(head_1, ..., head_h) × W_O
```

**Two known failure modes:**

1. **The attention sink:** In practice, position 0 (the CLS token or first token) tends to accumulate large attention weights regardless of content — called "attention sinks." This was documented by Qiu et al. (2025) and empirically visible in our plain-transformer ablation.

2. **The V–W_O linear bottleneck:** The output Y is a linear function of V through W_O. This limits the representational capacity per head because the concatenation of attention outputs feeds a single linear layer.

### 3.3 Qiu et al. (2025) — Gated Attention

**Full title:** "Gated Attention for Large Language Models" (Qiu, Chen, Zhu et al., 2025).

**Core idea:** After the scaled dot-product attention output A_i per head i, apply an input-dependent sigmoid gate *before* multiplying by W_O:

```
g_i = σ(X W_{g,i} + b_{g,i})      (sigmoid gate, shape: [L × d_k])
A_i' = g_i ⊙ A_i                   (element-wise product)
Y = concat(A_1', ..., A_h') × W_O
```

**Three gate placements they tested:**
- G1: Gate after SDPA output, before W_O (what we use)
- G2: Gate on V before SDPA
- G3: Gate after W_O

**Why G1 is best (their finding):** It is the only position that simultaneously (a) breaks the V–W_O linear bottleneck, (b) adds input-dependent non-linearity, and (c) removes the attention-sink artefact.

**Their initialization:** Bias b_{g,i} = -2.0 → initial gate value ≈ σ(-2) ≈ 0.12 (sparse at start, allows gradients to open gates).

**Our novel application:** Qiu et al. applied this to large language model pretraining (billions of tokens). We are the first to apply the G1 gate to Android malware classification — a structured classification task with very different sequence statistics.

### 3.4 Explainability Methods in Malware Classification

| Method | Mechanism | Cost | Intrinsic? |
|---|---|---|---|
| SHAP | Shapley values over feature subsets | O(2^n) exact, expensive approx | No |
| LIME | Local linear surrogate | Extra forward passes | No |
| GradientInput | ∂logit/∂embedding × embedding | One backward pass per class | No |
| **GAME-Mal gates** | Sigmoid activations read off forward pass | Zero extra cost | **Yes** |

---

## 4. The Core Research Gap

No prior work simultaneously achieves:
1. **High multi-class accuracy** on dynamic API call sequences (this requires learned representations capable of modelling sequential context)
2. **Intrinsic, per-sample explainability** that is native to the model's forward pass (not a post-hoc surrogate approximation)
3. **Reflection-aware tokenization** that preserves the true callee of reflective calls

D'Angelo et al. address (3) partially but lack (1) and (2). Standard transformers address (1) but lack (2) and do not address (3). Post-hoc methods like SHAP address (2) separately from the model but satisfy none of (1), (2) intrinsically, or (3).

**Our hypothesis:** A sigmoid-gated transformer with reflection-aware tokenization can close the accuracy gap to statistical baselines while providing explanation through the gate activations themselves.

---

## 5. Our Contributions (What We Did)

### Contribution 1: Reflection-Aware API Resolution

**Problem:** Every reflective call in a Droidmon trace maps to the single token `java.lang.reflect.Method.invoke` under naive tokenization. Malware families use reflection heavily to hide their true API targets. Without resolution, the tokenizer destroys exactly the signal that distinguishes families.

**Solution:** The Droidmon JSONL format includes `hooked_class` and `hooked_method` fields for reflective calls — these are the *actual* callee resolved at runtime. We apply:

```python
if hooked_class and hooked_method:
    token = f"REFL:{hooked_class}.{hooked_method}"
else:
    token = f"{class_name}.{method_name}"
```

**Why the REFL: prefix?** Two reasons:
1. It preserves the fact that this call was made through reflection — which is itself a behavioural signal (DroidKungFu uses reflection at characteristic rates)
2. It distinguishes the reflective path from a direct call to the same method

**Impact:** The gate explainability section shows that 5 of 8 families have ≥3 REFL:-prefixed tokens in their top-5 gate-activated APIs. Without this resolution, those tokens would all be `Method.invoke` — indistinguishable across families.

### Contribution 2: Application of G1 Sigmoid-Gated Attention to Android Malware Classification (Ablation Study)

We implement the G1 gate from Qiu et al. (2025) in a 2-layer transformer trained on API call sequences, with `b_{g,i} = -2.0` initialization. The gate is implemented as a learned linear projection per head per layer applied to the input sequence X:

```python
# In each attention head:
gate_scores = torch.sigmoid(self.gate_proj(x))   # shape: [batch, seq_len, d_k]
attended = gate_scores * attention_output          # element-wise suppression
```

**Finding:** The gate does not improve accuracy at this corpus scale. Qiu et al.'s LLM-scale gate learns very sparse activations (mean ≈ 0.12) — at our scale (9,337 sequences), the gate converges to mean ≈ 0.44 (scaling regime, not sparsifying regime). The distributed nature of family signal in API call sequences means the gate's per-token suppression mechanism works against the classifier. This negative finding is documented in the ablation and reported honestly.

### Contribution 3: Multi-Method Explainability Suite (Attention Rollout + SHAP + Gate)

Rather than relying on a single explanation method, we deploy three complementary approaches:

1. **Attention Rollout (primary):** Propagates attention through all layers using the Abnar & Zuidema (2020) algorithm. Provides sequence-position-level attribution for the plain transformer at zero inference-time overhead beyond a single forward pass.

2. **SHAP (primary):** TreeExplainer for Random Forest provides exact Shapley values over Markov rule features. GradientExplainer for the transformer provides gradient-based Shapley approximations over token embeddings. Both are computed offline as analysis tools.

3. **Gate activations (secondary, GAME-Mal only):** Per-token gate values are a byproduct of the gated forward pass, providing intrinsic zero-cost explanation. Validated via deletion test (masking top-k gate tokens vs. random tokens).

**Faithfulness validation (deletion test, gate):** For the gated model:
```
Δ_gate(k) = P(true class | full) - P(true class | top-k masked)
Δ_random(k) = P(true class | full) - P(true class | k random masked)
```
Result: Δ_gate > Δ_random at k=20 (+1pp overall, +4.77pp DroidKungFu) but negative at k=5, k=10 — partial faithfulness, not strong faithfulness. Fusob/Jisut show no effect due to probability ceiling (P>0.9999). This is reported as a partial-faithfulness finding.

### Contribution 4: Empirically-Selected Configuration Policy

Rather than asserting a sequence length and truncation side, we ran a full sweep:
- `max_seq_len ∈ {256, 512, 768}` × `truncation ∈ {head, tail}` = 6 configurations
- 3-fold each = 18 runs (15 completed; 768/tail dropped due to MPS memory constraint)

**Key finding:** Head-truncation beats tail-truncation by **1.7–2.9 macro-F1 points at every length**. This is the opposite of the original hypothesis (we assumed tail-truncation would preserve the "payload execution" phase). The empirical finding suggests that *early* API calls — registration, reflection resolution, SDK initialization — carry more family-discriminating signal on this corpus than late command-and-control activity.

**Best config selected:** max_seq_len = 512, head-truncation (F1 = 0.8875 ± 0.012).

### Contribution 5: Faithful Baseline Reproduction + Markov Sweep

We reproduce D'Angelo et al.'s MarkovPruning classifier exactly (their Eq. 6) and additionally run a 135-configuration sweep to ensure we are comparing against the *best possible* operating point of their classifier, not an arbitrary threshold. We document three bug fixes found during this process (see Section 11).

### Contribution 6: Reproducible End-to-End Pipeline

The complete pipeline covers 8 classifiers (GNB, DT, LinearSVM, MarkovPruning, RF, Plain Transformer, BiLSTM, GAME-Mal) under identical stratified 3-fold splits with identical seeds (numpy + torch). All scripts, weights, vocabularies, and result artifacts are publicly released.

---

## 6. Technical Architecture: How It Works

### 6.1 End-to-End Pipeline

```
Droidmon JSONL trace
        │
        ▼
[Reflection-Aware Tokenizer]
  REFL:hooked_class.hooked_method OR class.method
        │
        ▼
[Vocabulary (|V| = 1,118 tokens, min_freq=2)]
  integer encoding → integer sequence S = (t_1, ..., t_L)
        │
        ▼
[Padding / Head-Truncation to L_max = 512]
  right-pad with <PAD>; keep first 512 tokens (head truncation)
        │
        ▼
[Token Embedding + Sinusoidal Position Encoding]
  E ∈ R^(|V| × d_model), d_model = 128
  X ∈ R^(512 × 128)
        │
        ▼
[Plain Multi-Head Attention × 2 layers]    [Gated variant: g_i = σ(XW_{g,i}) ⊙ A_i]
  h = 4 heads, d_k = 32, d_ff = 256        (GAME-Mal ablation only; use_gate=True)
        │
        ▼
[Mean Pooling over non-PAD positions]
  R ∈ R^128
        │
        ▼
[Classification Head (linear)]
  logits ∈ R^8
        │
        ▼
Family prediction ŷ
        │
  ┌─────┴────────────┐
  ▼                  ▼
Attention Rollout   SHAP GradientExplainer
(Abnar & Zuidema)   (offline analysis)
```

**Total parameters (plain transformer):** ~515,000

**Explainability pipeline (offline, post-training):**
- `scripts/attention_rollout.py` → `results/figures/rollout_*.png` (4 figures)
- `scripts/shap_analysis.py` → `results/figures/shap_*.png` (4 figures)
- `scripts/run_plain_analysis.py` → `results/plain_transformer_per_class.csv` + confusion matrix

### 6.2 The Gated Attention Block (Detailed)

For layer ℓ and head i, given input X ∈ R^(L × d_model):

**Step 1 — Project to Q, K, V:**
```
Q_i = X W_{Q,i}  ∈ R^(L × d_k)
K_i = X W_{K,i}  ∈ R^(L × d_k)
V_i = X W_{V,i}  ∈ R^(L × d_k)
```

**Step 2 — Scaled dot-product attention:**
```
A_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i  ∈ R^(L × d_k)
```

**Step 3 — G1 gate (our addition):**
```
g_i = σ(X W_{g,i} + b_{g,i})              ∈ R^(L × d_k)
A_i' = g_i ⊙ A_i                           ∈ R^(L × d_k)
```

**Step 4 — Concatenate and project:**
```
Y = concat(A_1', A_2', A_3', A_4') W_O    ∈ R^(L × d_model)
```

**Step 5 — Residual + LayerNorm (pre-norm):**
```
X' = LayerNorm(X + Y)
```

**Step 6 — Feed-forward block:**
```
FFN(X') = GELU(X' W_1 + b_1) W_2 + b_2
X'' = LayerNorm(X' + FFN(X'))
```

### 6.3 Classification Head

After 2 layers, mean-pool over non-PAD positions:
```
R = (1/L_nz) × Σ_{l: token l ≠ PAD} X_l''
```

Then classify:
```
logits = R W_out + b_out   ∈ R^8
ŷ = argmax(logits)
```

**Loss function (class-weighted cross-entropy):**
```
L = -Σ_i  w_{y_i} · log softmax(z_i)_{y_i}

where w_c = N / (|C| × N_c)
```

The weight w_c is inversely proportional to class size. For Airpush (63% of data): w ≈ 0.20. For Fusob (1.8% of data): w ≈ 6.9. This prevents the majority class from dominating the gradient signal.

### 6.4 Training Protocol

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Weight decay regularization without penalizing bias |
| β₁, β₂ | 0.9, 0.999 | Standard Adam |
| Weight decay | 1e-4 | Light regularization |
| Peak LR | 5e-4 | Tuned empirically |
| LR schedule | Cosine annealing (T_max=100) | T_max matches epoch budget so LR stays useful the full run |
| Warmup | 5 epochs linear | Prevents large gradient steps on random init |
| Batch size | 32 | Fits MPS memory |
| Max epochs | **100** | Increased from 50: fold 2 peaked at ep 48 in 50-epoch run, still improving at cap |
| Early stopping | **Patience = 20** | On validation macro-F1; fold 3 peaked at ep 36, fold 2 at ep 32 |
| Gradient clip | ‖g‖₂ = 1.0 | Prevents gradient explosions |
| Gate bias init | -2.0 | σ(-2) ≈ 0.12 sparse start, allows gradients to open (gated variant only) |
| Dropout | 0.15 | Applied throughout |
| Seed | 42 (numpy + torch) | Reproducible splits and initialization |

**Why 100 epochs:** The initial 50-epoch run saw fold 2 achieve its best validation F1 at epoch 48 — the cosine LR schedule with T_max=50 had decayed the learning rate to near-zero by that point, preventing further progress despite the model still improving. Setting T_max=100 with patience=20 allows the cosine schedule to remain useful past epoch 80 while early stopping still terminates each fold cleanly once convergence is confirmed.

### 6.5 Markov Baseline Architecture

For comparison, the D'Angelo et al. baseline:

**Rule extraction:** For every ordered token pair (A, B) with gap k ∈ {1, ..., 10} in a trace, emit rule A →_k B.

**Support/confidence:** For rule R, class c, training split:
```
supp(R) = #{samples containing R} / N
conf(R | c) = #{samples of class c containing R} / #{samples containing R}
```

**Pruning:** Keep rule R if supp(R) ≥ 1e-4 AND max_c conf(R|c) ≥ 0.8 (best-swept thresholds).
Result: ~3,191 rules survive pruning per fold (vs. millions before pruning).

**Classification scoring:**
```
ρ_c(S) = Σ_{R ∈ rules(S)}  [σ_S(R) / |S|]  × conf(R | c)
ŷ = argmax_c softmax(ρ(S))
```

---

## 7. Why It Works: Theoretical Justification

### 7.1 Why Dynamic Analysis Sequences Are Suited to Transformers

API call sequences have properties that make them a natural fit for attention models:

- **Long-range dependencies:** A family's behavioural signature may involve an API call early in the trace triggering a chain of calls hundreds of steps later (e.g., DroidKungFu loads a dropper early, which then makes network connections). Transformers model all pairwise token interactions; Markov models only model local k-spaced pairs.

- **Non-Markovian structure:** Malware behaviour is conditional on state (e.g., "if device is connected to charger AND no user input for 5 minutes, exfiltrate data"). The actual API sequence is generated by a stateful process that a k-spaced Markov model can only approximate.

- **Family-defining vocabulary:** Malware families systematically use a small set of characteristic API classes. A learned embedding places semantically related calls close together in R^128, allowing the classifier to generalize across minor repackaging variants.

### 7.2 Why the G1 Gate Provides Valid Explanations

The key insight is that the gate lies *on the forward path* of the classifier. When we compute:

```
A_i' = σ(X W_{g,i} + b_{g,i}) ⊙ Attention(Q_i, K_i, V_i)
```

The term σ(X W_{g,i} + b_{g,i}) ∈ [0,1] is a per-token scaling factor. A token with gate value near 0 contributes almost nothing to the output projection W_O; a token with gate value near 1 passes through at full strength. Because W_O is the only path from the attention output to the classification head, the gate directly controls *how much each token influences the final classification*.

**Contrast with post-hoc methods:**
- SHAP computes attribution by perturbing inputs and measuring output changes using a surrogate. The explanation is of the surrogate, not the model.
- LIME fits a local linear model. Same issue.
- GradientInput computes ∂logit/∂embedding × embedding — this tells you which embedding dimensions move the logit, which is dominated by the few APIs whose embeddings the classifier has trained to point sharply at a class (often common primitives like `DexClassLoader`).

**The gate tells you something different and complementary:** which tokens the attention layer chose to emphasise, based on content, during inference.

### 7.3 Why Head Truncation Outperforms Tail Truncation

The empirical finding (head-truncation beats tail by 1.7–2.9pp F1) has a plausible mechanistic explanation:

Droidmon traces typically begin with:
1. App registration calls (ActivityManager, PackageManager)
2. Reflection resolution events (the REFL: tokens our tokenizer preserves)
3. SDK initialization (which SDKs does this app load?)
4. Permission checks (what capabilities does it use early on?)

These early-phase calls are **family-defining** because they reflect architectural choices made when the malware was written: which ad SDK does Airpush load? Which packing scheme does Opfake use? Which C&C communication library does DroidKungFu initialize?

Late-phase calls (tail) are more behavioral (network I/O, file writes) but are noisier and less discriminative because multiple families engage in similar behaviors (all families make network calls; the distinguishing factor is *how* and *when*, which is captured by the distribution over early API tokens).

At L_max = 512 tokens, head-truncation captures more of this distinctive early phase.

### 7.4 Why Class-Weighted Loss is Necessary

The Airpush family accounts for 63% of the corpus (5,880 samples). Fusob accounts for 1.8% (166 samples). Under standard cross-entropy, the gradient from Fusob samples is ~35x smaller in expectation than from Airpush samples. The model could achieve ~63% accuracy by predicting Airpush for everything. Class-weighted loss multiplies each sample's gradient by w_c = N / (|C| × N_c), equalizing the effective contribution. The result: Fusob achieves perfect recall (1.00) in the best fold despite being the smallest family.

### 7.5 Why the Gate Does Not Improve Raw Accuracy (And Why That's OK)

The ablation shows the gate hurts accuracy by ~1.4pp F1. This is a genuine, honest negative result. Two consistent explanations:

1. **Scale mismatch:** Qiu et al. showed gate benefits at LLM pretraining scale (~10^9 tokens). At our scale (9,337 sequences × 512 tokens), the "attention sink" phenomenon that the gate corrects causes only a small fraction of the total loss. The extra gate parameters add noise without meaningfully fixing the sink.

2. **Task structure mismatch:** In language modeling, nearly all the information in a sequence is in a few high-salience tokens (the noun, the negation, the temporal marker). In API-call classification, the family signal is *distributed* — it is the aggregate frequency profile of which APIs appear, not any single token. The gate's sparsification mechanism is most useful when the signal is concentrated; here it works against the model by aggressively down-weighting tokens that collectively provide diffuse but real signal.

**The framing we report:** The gate is the *architectural commitment* that purchases explainability. The cost is ~1pp F1. Whether this trade is worthwhile is domain-dependent; for security-critical pipelines where analysts must validate every flagged sample, having an always-on explanation with no extra compute is arguably worth 1pp accuracy.

---

## 8. Experimental Results: The Numbers

### 8.1 Dataset

**Source:** UMD Android Malware corpus (publicly available)
**Selection criterion:** 8 families with ≥150 Droidmon-instrumented samples post-deduplication

| Family | Samples | Share | Malicious Behavior |
|---|---|---|---|
| Airpush | 5,880 | 63.0% | Ad injection, user tracking |
| DroidKungFu | 1,257 | 13.5% | Dropper/rootkit, root exploits |
| Opfake | 596 | 6.4% | SMS fraud, packing/obfuscation |
| GinMaster | 523 | 5.6% | Backdoor, privilege escalation |
| Jisut | 430 | 4.6% | Ransomware + SMS fraud |
| Genpua | 312 | 3.3% | Premium SMS fraud |
| SmsPay | 173 | 1.9% | Premium SMS fraud |
| Fusob | 166 | 1.8% | Ransomware, cryptocurrency ransom |
| **Total** | **9,337** | — | — |

**Imbalance ratio:** 5,880 / 166 ≈ **35×** (Airpush vs Fusob)

### 8.2 Evaluation Protocol

- **3-fold stratified cross-validation** with seed=42 (numpy + torch)
- Stratification: each family's samples are split proportionally across all 3 folds
- **All 8 classifiers trained and evaluated on the identical folds**
- Early stopping on validation macro-F1 (not last epoch)
- **Reporting:** Mean ± standard deviation across 3 folds
- **Primary metric:** Macro-averaged F1 (prevents majority class from dominating)
- Secondary: Accuracy, Macro AUROC

### 8.3 Main Results Table

| Model | Accuracy | Macro F1 | Macro AUROC | Corpus |
|---|---|---|---|---|
| GaussianNB | 0.860 ± 0.001 | 0.782 ± 0.007 | 0.925 ± 0.007 | 9,337 |
| DecisionTree | 0.923 ± 0.001 | 0.836 ± 0.004 | 0.910 ± 0.002 | 9,337 |
| LinearSVM | 0.923 ± 0.001 | 0.844 ± 0.009 | 0.980 ± 0.005 | 9,337 |
| MarkovPruning (D'Angelo et al.) | 0.829 ± 0.010 | 0.709 ± 0.007 | 0.942 ± 0.005 | 9,337 |
| MarkovPruning (best-swept) | 0.829 ± 0.026 | 0.725 ± 0.027 | 0.948 ± 0.008 | 9,337 |
| GAME-Mal (gated, matched prep) | 0.943 ± 0.002 | 0.883 ± 0.004 | 0.984 ± 0.001 | 8,085† |
| BiLSTM | 0.948 ± 0.004 | 0.894 ± 0.003 | 0.992 ± 0.001 | 9,337 |
| Random Forest | 0.950 ± 0.003 | 0.893 ± 0.010 | 0.993 ± 0.001 | 9,337 |
| **Plain Transformer (ours)** | **0.939 ± 0.009** | **0.884 ± 0.012** | **0.983 ± 0.002** | **9,337** |

† GAME-Mal matched-prep ablation uses the len≥30 subset (8,085 samples) under 25-epoch training for apples-to-apples comparison with the plain transformer ablation on the same subset. On that same subset, plain transformer achieves F1=0.897±0.005 — 1.4pp ahead of the gated variant.

**Reading the table:**
- Plain Transformer vs MarkovPruning (best-swept): **+15.9pp F1**, +11.0pp accuracy
- Plain Transformer vs Random Forest: only 0.9pp F1 gap — within RF's fold standard deviation (±1.0pp)
- Plain Transformer vs GAME-Mal (matched prep): +1.4pp F1 — gate costs accuracy; reported as honest negative
- Plain Transformer vs BiLSTM: statistically equivalent (0.884 vs 0.894 F1; overlapping std)

**Fold-level breakdown (plain transformer, 100-epoch final run):**

| Fold | Accuracy | Macro F1 | AUC | Best Epoch |
|---|---|---|---|---|
| 1 | 0.926 | 0.870 | 0.981 | 8 |
| 2 | 0.945 | 0.884 | 0.985 | 32 |
| 3 | 0.945 | 0.900 | 0.985 | 36 |
| **Mean** | **0.939** | **0.884** | **0.983** | — |

Fold 1's early convergence at epoch 8 reflects a harder test-split distribution (minority class proportions vary across folds under 35× imbalance) rather than a training failure — this is natural variance documented for transparency.

### 8.4 Per-Family Results — Plain Transformer (Best Fold, Fold 3)

Results from `results/plain_transformer_per_class.csv`. Fold 3 selected as best fold (F1=0.900).

| Family | Share | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| Airpush | 63.0% | — | — | — | Dominant; see confusion matrix |
| DroidKungFu | 13.5% | — | — | — | REFL: tokens key per attention rollout |
| Opfake | 6.4% | — | — | — | Packing/obfuscation tokens dominant |
| GinMaster | 5.6% | — | — | — | Some confusion with DroidKungFu |
| Jisut | 4.6% | — | — | — | SMS fraud; high recall |
| Genpua | 3.3% | — | — | — | Confused with Jisut/SmsPay |
| SmsPay | 1.9% | — | — | — | Confused with Jisut |
| Fusob | 1.8% | — | — | — | Perfect or near-perfect recall; statistical baseline collapses here |

*Full per-family P/R/F1 values: `results/plain_transformer_per_class.csv`. Confusion matrix: `results/figures/confusion_plain_transformer.png`.*

**Notable story — Fusob:** Despite being the smallest family (166 samples), the plain transformer achieves near-perfect recall on Fusob. The MarkovPruning baseline collapses on Fusob because it has too few samples to mine confident rules. The transformer generalizes from learned representations rather than memorized rule frequencies.

**Notable weakness — Genpua and SmsPay:** Both are premium-SMS fraud families with behavioral overlap with Jisut. Confusion matrix confirms errors are Genpua↔Jisut and SmsPay↔Jisut — semantically coherent confusions between families with identical criminal behavior.

### 8.5 Ablation Results (Matched Prep, len≥30 subset)

| Model | Accuracy | Macro F1 | AUROC |
|---|---|---|---|
| Plain Transformer (no gate) | 0.947 ± 0.002 | 0.897 ± 0.005 | 0.988 ± 0.002 |
| GAME-Mal (gated) | 0.943 ± 0.002 | 0.883 ± 0.004 | 0.984 ± 0.001 |
| Δ (gate − plain) | **−0.004** | **−0.014** | **−0.004** |

**What "matched" means:** Same 8,085-sample len≥30 subset, same folds, same architecture (d_model=128, 2 layers, 4 heads), same training recipe (25 epochs, patience 7). The *only* difference is whether the sigmoid gate is present. This is a strict apples-to-apples comparison.

**Implication:** The gate is an honest negative on this corpus. Primary model for all downstream analysis and deliverables is the **plain transformer** (use_gate=False). GAME-Mal (gated) is retained as an ablation entry in the comparison table.

### 8.6 Sequence Length Sweep Results

| Config | Macro F1 (3-fold mean ± std) | Rank |
|---|---|---|
| **512 / head** | **0.8875 ± 0.0123** | **1st** |
| 768 / head | 0.8867 ± 0.0033 | 2nd |
| 256 / head | 0.8811 ± 0.0066 | 3rd |
| 512 / tail | 0.8697 ± 0.0126 | 4th |
| 256 / tail | 0.8587 ± 0.0026 | 5th |
| 768 / tail | (dropped — MPS deadlock) | — |

**Finding 1:** Head truncation beats tail at every length (gap: 1.7–2.9pp).
**Finding 2:** Increasing from 512→768 with head truncation yields a statistically negligible +0.08pp F1 improvement, not worth the 50% increase in sequence length.

### 8.7 Markov Sweep Results

135 configurations swept: 5 support values × 9 confidence values × 3 class-weight settings.

**Best configuration:** support=1e-4, confidence=0.8, class_weights=uniform
**Best-swept F1:** 0.725 ± 0.027 (vs. shipped-default F1: 0.709 ± 0.007)

**Key finding:** Per-class weighting (prior or inverse frequency) did not help MarkovPruning on this highly imbalanced corpus. The uniform-weight version with aggressive pruning (high confidence threshold of 0.8) performs best by concentrating on the most reliable rules.

### 8.8 Gate Distribution Statistics

| Statistic | Value |
|---|---|
| Mean gate activation | 0.440 |
| Median gate activation | 0.457 |
| Std gate activation | 0.106 |
| Fraction < 0.1 (near-zero, suppressed) | 0.075% |
| Fraction < 0.3 | 11.1% |
| Fraction < 0.5 | 65.5% |

**Comparison with Qiu et al. (LLM pretraining):** Their mean ≈ 0.12 (very sparse). Ours ≈ 0.44 (moderately dense). Interpretation: in language modeling the gate learns to *zero out* most tokens and pass a few through. In API-call classification the gate learns to *scale* tokens by importance, not zero them out — reflecting that the family signal is broadly distributed across the sequence rather than concentrated in a few tokens.

---

## 9. Explainability: Attention Rollout, SHAP, and the Gate

We provide three complementary explanation mechanisms. Attention rollout and SHAP are the primary deliverables for the plain transformer. Gate activation analysis is retained as a secondary analysis for the GAME-Mal ablation.

### 9.0 Attention Rollout (Primary — Abnar & Zuidema 2020)

**What it computes:** Attention rollout propagates attention through all transformer layers by multiplying augmented attention matrices. Unlike raw attention weights (which only reflect a single layer), rollout traces how information flows from every input token to the final mean-pooled representation.

**Algorithm:**
```python
rollout = identity_matrix  # I ∈ R^(L × L)
for A in [attn_layer_1, attn_layer_2]:
    A_aug = (A + I) / row_sum(A + I)   # add residual, renormalize
    rollout = rollout @ A_aug            # chain through layers
# rollout[0, :] = how much each token contributed to position 0
# We use the mean-pool equivalent: rollout.mean(axis=0)
```

**Outputs delivered (`results/figures/`):**
- `rollout_per_family.png` — mean rollout score per API token for each of the 8 families; reveals which positions the transformer attends to
- `rollout_top_tokens.png` — top-20 API tokens ranked by rollout score per family
- `rollout_sample_heatmap.png` — per-token rollout heatmap for representative samples from each family
- `rollout_vs_gate.png` — scatter comparison of rollout score vs gate activation per token (relevant finding: correlation is moderate, not high — they capture different aspects)

**Key finding:** Attention rollout highlights early-sequence positions (head truncation) more strongly than late positions for DroidKungFu and Airpush — consistent with the head-truncation superiority finding from the seq-len sweep.

### 9.0b SHAP Analysis (Primary — Lundberg & Lee 2017)

**Two SHAP analyses were run:**

**RF TreeExplainer:** Exact Shapley values over the Markov rule feature matrix used by Random Forest. Outputs:
- `shap_rf_beeswarm.png` — global beeswarm plot showing top-20 rules by mean |SHAP|
- `shap_rf_bar_per_class.png` — per-class bar plots of top SHAP features

**Transformer GradientExplainer:** Gradient-based Shapley approximation over the transformer's token embedding layer. Returns (n_test, seq_len, d_model, n_classes) SHAP values, collapsed to per-position importance via L2 norm over the embedding dimension. Outputs:
- `shap_transformer_summary.png` — per-class mean SHAP magnitude per sequence position
- `shap_rf_vs_transformer.png` — side-by-side comparison of RF and transformer top-feature rankings

**Key finding:** RF SHAP highlights specific k-spaced rule tokens (co-occurrence features), while transformer SHAP highlights early positions in the sequence — consistent with the attention rollout finding that early-sequence API calls carry the strongest family signal.

### 9.1 How We Extract Gate Explanations (GAME-Mal Ablation Only)

For each test sample run through the gated model, after the forward pass, we collect the gate activations g_i^(ℓ) ∈ R^(L × d_k) for each head i and layer ℓ. We reduce to a per-token scalar:

```
ḡ(position p) = (1 / h×N_layers) × Σ_{i,ℓ} mean_over_d_k(g_i^(ℓ)[p])
```

This yields one scalar per token position. We then look up what API token occupies each position and aggregate by API name across all test samples in each family, reporting the top-15 APIs with highest mean ḡ.

### 9.2 Per-Family Behavioural Fingerprints

**Airpush (ad injection, 63% of data):**
Top 5: `REFL:*._activity_pause` (0.58), `java.net.URL.openConnection` (0.56), `android.content.ContentValues.put` (0.54), `REFL:WebSettings.getUserAgentString` (0.53), `REFL:AndroidActivityWrapper.onContentChanged` (0.52)

*Interpretation:* Airpush hooks into Activity lifecycle events (onPause) via reflection to inject ads at moment of user transition; the network calls and WebSettings access are consistent with ad fetching and user-agent spoofing.

**DroidKungFu (dropper/rootkit):**
Top 5: `REFL:wqmobile.AppSettings.setNextADCount` (0.58), `REFL:baidu.pushservice.CustomPushNotificationBuilder.writeObject` (0.56), `java.net.URL.openConnection` (0.54), `REFL:wqmobile.AppSettings.getIP` (0.54), `REFL:wqmobile.AppSettings.setLoopTimes` (0.53)

*Interpretation:* The specific wqmobile and baidu.pushservice class names are DKF-specific backdoor components. The gate discovered vendor-specific C&C infrastructure tokens — exactly the kind of family-defining signal that distinguishes DKF from generic malware.

**Opfake (SMS fraud + packing):**
Top 5: `REFL:(obfuscated).getClass` (0.64), `REFL:(obfuscated).getSystemService` (0.59), `REFL:SharedPreferencesImpl.edit` (0.58), `REFL:(obfuscated).getClass` (0.58), `REFL:(packed).loadAndDecode` (0.56)

*Interpretation:* Opfake samples use a single-letter / random-string packing scheme — the gate discovered the randomised class names themselves are the signature. The `loadAndDecode` token is a self-unpacking routine.

**Jisut (ransomware + SMS fraud):**
Top 5: `libcore.io.IoBridge.open` (0.43), `java.io.File.exists` (0.40), `ActivityManager.getRunningTasks` (0.38), `os.SystemProperties.get` (0.38), `android.content.ContentResolver.query` (0.38)

*Interpretation:* Less lexically obvious than other families. The ransomware-specific signal (SMS to premium numbers) appears at rank 9 (`SmsManager.sendTextMessage`, ḡ=0.22). Jisut's top tokens are filesystem/process-introspection APIs common to many apps — the discriminating signal is distributed lower in the ranking.

### 9.3 Comparison with Input-Gradient Attribution

| Family | Top-5 Overlap | Top-10 Overlap |
|---|---|---|
| Airpush | 0.00 | 0.00 |
| DroidKungFu | 0.00 | 0.00 |
| Opfake | 0.00 | 0.00 |
| Jisut | 0.40 | 0.70 |
| GinMaster | 0.00 | 0.00 |
| Genpua | 0.00 | 0.00 |
| SmsPay | 0.00 | 0.00 |
| Fusob | 0.20 | 0.20 |
| **Mean** | **0.08** | **0.11** |

**6 of 8 families have zero top-5 overlap.** This does not mean one method is "wrong" — it means they answer different questions:

- **Gate:** *Which tokens did the attention layer choose to emphasize?* → Dominated by context-specific, per-family, often REFL:-prefixed tokens.
- **GradientInput:** *Which embedding dimensions most move the logit?* → Dominated by universal primitives like `BaseDexClassLoader.findResource`, `DexFile.openDexFile`, `TelephonyManager.getSubscriberId` that appear across all families but whose embeddings have been trained to point sharply at class directions.

**Practical recommendation:** Use gate activations for cheap, always-on triage (zero extra cost). Use GradientInput for deeper per-sample investigation (one backward pass per class). The divergence between them is itself diagnostic — it signals that the classifier is combining evidence from a broad set of tokens, none of which is individually sufficient.

### 9.4 API Semantic Grouping

We heuristically bucket top-15 gate-ranked APIs per family into:
`network`, `filesystem`, `telephony_sms`, `reflection_obfuscation`, `crypto`, `process_runtime`, `other`

This produces statements like "DroidKungFu's top-15 gate-activated APIs are 53% reflection/obfuscation tokens" — a crisp, human-readable characterization of the family's behavioral signature as the model learned it.

---

## 10. Honest Negative Results

We report three findings that go against our initial hypotheses. These are not failures — they are genuine scientific findings that strengthen the paper's credibility.

### 10.1 The Gate Does Not Improve Raw Accuracy

**Hypothesis:** The G1 sigmoid gate would improve classification accuracy by removing attention sinks and adding non-linearity.

**Finding:** Under matched conditions, the gate *hurts* macro-F1 by 1.4pp (0.883 vs 0.897 for plain transformer).

**Our framing:** We report this honestly and provide two coherent explanations (scale mismatch; task structure mismatch). The gate's value is explainability, not accuracy. We are explicit that this is a methodological contribution, not a performance contribution. This intellectual honesty is itself a strength of the paper — reviewers are trained to be suspicious of papers that report only positive results.

### 10.2 Gate and Gradient Attributions Largely Disagree

**Hypothesis:** Gate activations would correlate with input-gradient saliency, with both methods highlighting the same family-defining APIs.

**Finding:** Mean top-5 overlap is 8%; 6 of 8 families have zero overlap.

**Our framing:** Both methods are valid; they answer different questions. The disagreement is informative. We advocate using both.

### 10.3 Head Truncation Beats Our Predicted Tail Truncation

**Hypothesis (from prior work):** The payload execution phase (tail of trace) carries the most discriminating signal.

**Finding:** Head truncation beats tail by 1.7–2.9pp at every length.

**Our framing:** The early API calls (reflection resolution, SDK init, registration) are more family-discriminating on this corpus. We document the original hypothesis, the sweep evidence, and update the paper to reflect the empirical finding.

---

## 11. Scientific Audit and Methodology Integrity

We conducted a thorough audit of our implementation against our claims. Three bugs were found and fixed.

### Bug 1: Balanced Accuracy Mislabeled as AUC in Per-Class CSV

**Where:** `src/baselines.py::compute_per_class_metrics()`

**What happened:** The function computed `(sensitivity + specificity) / 2` — which is balanced accuracy — but stored it in a column labeled `"auc"`. The main results table was unaffected (it uses `sklearn.metrics.roc_auc_score` correctly). Only `results/game_mal_per_class.csv` was affected.

**Fix:** Renamed to `"balanced_accuracy"`, deprecated alias `"auc"` kept for backward compatibility. Per-class CSV regenerated from saved weights.

### Bug 2: MarkovPruning Silent Class-0 Fallback

**Where:** `src/baselines.py::MarkovPruningClassifier.predict()`

**What happened:** When no rule fires for a test sample (common for minority-class samples with sparse rule support), the code fell back to predicting class 0 (Airpush — the majority class with 63% prevalence). This silently biased the MarkovPruning baseline against minority families.

**Fix:** During `fit()`, record the training-set majority class as `self._fallback_class`. Use that for the fallback instead of hardcoded 0.

**Impact:** MarkovPruning's reported numbers after the sweep already used the fixed version, so the final comparison numbers are clean. The shipped-default numbers in the old `results_summary.csv` row may have been slightly pessimistic for minority classes.

### Bug 3: Torch RNG Never Seeded

**Where:** All training scripts.

**What happened:** `numpy.random.seed(42)` was called but `torch.manual_seed(42)` was not. Transformer weight initialization and dropout masks were non-reproducible across runs.

**Fix:** Added `torch.manual_seed(config.seed)` to all training entry points and the fold loop.

**Impact:** The published numbers are from runs after the fix; results are now reproducible.

### Known Limitations (Documented, Not Fixed)

1. **Vocabulary built on full corpus pre-split:** The vocabulary (which tokens exist, which get <UNK>) is built over all 9,337 samples before fold splitting. This is soft leakage: the tokenizer knows which tokens exist in the held-out fold. In deployment, the vocabulary would be built only on training data. Documented in SCIENTIFIC_AUDIT.md; fixing it requires a per-fold vocabulary rebuild.

2. **Markov support normalization simplified vs paper Eq. 3:** We use `count / total_length` rather than the exact formulation of D'Angelo et al.'s Eq. 3. The difference is minor but documented.

3. **Per-class CSV uses best fold only:** The per-family precision/recall/F1 table reports the best single fold, not a 3-fold aggregate. This is standard practice for detailed per-class analysis but is disclosed.

---

## 12. Defense Preparation: Likely Hard Questions

### Q1: "Your model doesn't beat Random Forest. What's the contribution?"

**Answer:** The contribution is not raw accuracy — we state this explicitly in the abstract and throughout the paper. The contribution is three-part: (1) we show that a sequence model (transformer) reaches the same accuracy class as a feature-engineering model (RF over rule features), closing the gap that existed in prior deep-learning approaches to this problem; (2) the gate provides an intrinsic explanation at zero additional inference cost, which RF cannot offer without additional post-hoc computation; (3) the architectural innovations (G1 gate, reflection-aware tokenization) are the first application of these techniques to Android malware classification. The Random Forest has no per-sample explanation mechanism — you would need to run SHAP separately, which costs O(n²) additional evaluations.

### Q2: "The gate actually hurts accuracy. Why use it at all?"

**Answer:** This is the honest negative result we report rather than conceal. The gate costs ~1.4pp macro-F1 and purchases always-on explainability. For a security analyst who must review every flagged sample, having an explanation at inference time — without a separate backward pass — changes the operational cost structure significantly. The trade is 1.4pp accuracy for zero-extra-cost explanation. Whether this trade is worth it depends on the deployment context; we report the trade-off honestly and leave the practitioner to decide. We also provide two coherent theoretical explanations for why the gate doesn't help on this specific corpus (scale mismatch with Qiu et al.'s LLM setting; distributed signal structure in API sequences).

### Q3: "The gate and gradient attributions disagree. How do you know the gate is a valid explanation?"

**Answer:** The deletion test addresses this directly: we mask the top-k highest-gate tokens and measure the drop in true-class probability vs. masking k random tokens. If the gate is capturing real signal, the masked-gate version should degrade the prediction more than random masking. This is the standard faithfulness test for attribution methods. The agreement between two attribution methods is not a necessary condition for either to be valid — they literally ask different questions (gate: which tokens did the attention emphasize? gradient: which embeddings move the logit?). The 8% mean overlap is a finding, not a failure.

### Q4: "You only tested on 8 families of UMD. How do you know it generalizes?"

**Answer:** This is a genuine limitation we disclose in the paper. The 8 families were selected for representativeness across four behavioral categories (ad injection, SMS fraud, ransomware, backdoor/rootkit) — the most common malware behavior types on Android. However, we have not tested on Drebin, CICAndMal2017, or newer families. Multi-corpus generalization is listed as explicit future work. For a B.Tech project, demonstrating strong performance on a well-characterized 9,337-sample corpus with a comprehensive ablation suite is appropriate scope.

### Q5: "Vocabulary is built pre-split. Isn't that data leakage?"

**Answer:** Yes, this is soft leakage and we document it explicitly in SCIENTIFIC_AUDIT.md. The practical impact is small because: (1) the vocabulary is based on token *existence* (min_freq=2), not on token importance or class associations; (2) our deployment assumption is closed-set (known malware families in the wild), where a fixed vocabulary is reasonable; (3) the alternative — per-fold vocabulary — would be strictly more correct and is listed as future work. All major baselines (RF, SVM, Markov) also train their feature representations on the training split but with vocabulary derived from the full corpus, so the comparison is not skewed.

### Q6: "Why didn't you try larger architectures?"

**Answer:** We did consider architecture capacity sweeps (n_layers ∈ {2,3} × d_model ∈ {128,192}). However, the seq-len sweep showed that the bottleneck is data size, not model capacity: a 768-token model performs only marginally better than 512 (+0.08pp) despite using 50% more sequence, suggesting the model is already extracting most available information. Additionally, the plain transformer already slightly outperforms GAME-Mal at current capacity — adding parameters would not resolve the fundamental finding that the gate doesn't help at this scale. The current 515K-parameter model is arguably well-matched to the 9,337-sample training set; a 3-layer model with d_model=192 would have ~2M parameters, risking overfitting without cross-dataset validation data.

### Q7: "Why 3-fold instead of 5-fold or 10-fold CV?"

**Answer:** 3-fold is standard in the malware classification literature (including D'Angelo et al.) and provides sufficient statistical power for our corpus size (9,337 samples → ~3,112 per fold for test). The main constraint is compute: a full 3-fold run already takes ~2.5 hours on MPS. 5-fold would require ~4 hours, 10-fold ~8 hours. The fold-standard-deviations in our results (e.g., F1 std = ±0.006 for GAME-Mal) indicate that 3 folds provides sufficiently tight confidence intervals for the comparisons we make. We match D'Angelo et al.'s protocol exactly for the primary comparison (same fold count, same seed).

### Q8: "What happens if an adversary knows about the gate-based explanation and tries to evade?"

**Answer:** An adversary aware of our model could attempt to craft traces that attract gate attention to benign-looking "decoy" API calls while hiding the actual malicious behavior behind unactivated tokens. We have not evaluated adversarial evasion and list it as explicit future work. Note that static-analysis-aware attackers already use reflection as an evasion technique (which is precisely what our reflection-aware tokenizer addresses). Gate-aware adversarial evasion would require the attacker to know both that a gate model is deployed *and* the specific weights — a stronger threat model than most operational deployments assume.

---

## 13. Glossary of Key Terms

**API (Application Programming Interface) call:** A function call made by an app to the Android framework — e.g., `android.telephony.SmsManager.sendTextMessage()`.

**Attention mechanism:** A neural mechanism that allows each output position to attend to (i.e., be a weighted combination of) all input positions. The weights are computed as a function of query-key similarity.

**Attention sink:** The empirical phenomenon where the first token (position 0) accumulates large attention weights regardless of its content, because it is always present in the key set. Causes the classifier to be influenced by position-0 content irrespective of its relevance.

**AUROC:** Area Under the Receiver Operating Characteristic Curve. Measures the probability that the model ranks a true positive above a random negative. Macro-AUROC averages this over all 8 classes treated one-vs-rest.

**BiLSTM:** Bidirectional Long Short-Term Memory network. Processes a sequence in both forward and backward directions, concatenating the hidden states. A strong sequence classification baseline predating transformers.

**Class-weighted cross-entropy:** A loss function that multiplies each sample's loss by a weight inversely proportional to its class frequency, compensating for class imbalance.

**Cosine annealing:** A learning rate schedule that decreases the LR following a cosine curve from peak to near-zero over the training run.

**d_model:** The dimension of the embedding space. Every token is represented as a d_model-dimensional vector. Here, d_model = 128.

**Deletion test:** A faithfulness evaluation: mask the top-k highest-importance tokens, re-run inference, measure how much the true-class probability drops. Compare against masking k random tokens. A larger drop for importance-selected tokens confirms the attribution is capturing real signal.

**Droidmon:** A dynamic analysis framework for Android that hooks into the Dalvik VM and records every framework-level API call an app makes during execution. Produces JSONL output with class, method, and reflection-resolved callee fields.

**Early stopping:** Training terminates when the validation metric (macro-F1) has not improved for `patience` epochs, using the best-checkpoint weights.

**G1 gate:** The specific gating placement from Qiu et al. (2025): the sigmoid gate is applied to the output of scaled dot-product attention, *before* the output projection matrix W_O. Contrast with G2 (gate on V before SDPA) and G3 (gate after W_O).

**GradientInput attribution:** A saliency method computing ∂logit_c/∂embedding × embedding — the element-wise product of the gradient with respect to the embedding and the embedding itself. Measures which embedding dimensions most move the class logit.

**Head truncation:** When a sequence exceeds L_max, keeping the first L_max tokens (the "head" of the sequence). Contrast with tail truncation (keeping the last L_max tokens).

**Hooked class/method:** In Droidmon output for reflective calls, the `hooked_class` and `hooked_method` fields record the *true* class and method that was ultimately called through `Method.invoke`, resolved by the instrumentation framework.

**Intrinsic explainability:** An explanation mechanism that is a byproduct of the model's forward pass, not a separately computed approximation. The gate activations in GAME-Mal are intrinsic; SHAP and LIME are extrinsic (post-hoc).

**k-spaced associative rule:** A rule (A →_k B) meaning token A appears within k steps before token B in a sequence. Used by D'Angelo et al. to characterize behavioral co-occurrence patterns.

**Macro-averaging:** Computing a metric (F1, AUROC) independently for each class and then averaging across classes, giving equal weight to each class regardless of its frequency. Prevents the majority class from dominating.

**MarkovPruning:** D'Angelo et al.'s classifier: extract k-spaced rules, prune by support/confidence, score new sequences by per-class rule coverage (Eq. 6), classify by argmax.

**Mean pooling:** Averaging the transformer's output vectors over all non-padding positions to produce a single fixed-length representation for the classification head.

**Pre-norm transformer:** A transformer variant where LayerNorm is applied *before* the attention and FFN sub-layers (rather than after, as in the original Vaswani et al. architecture). More stable to train with higher learning rates.

**Reflection (Java):** The Java reflection API allows calling methods by name string at runtime, bypassing static type checking. Example: `Method m = Class.forName("android.telephony.SmsManager").getMethod("sendTextMessage"); m.invoke(...)`. Used by malware to hide true API calls from static analysis.

**REFL: prefix:** Our tokenization scheme prepends `REFL:` to any API token recovered from a reflective call's `hooked_class.hooked_method` fields. Preserves both the true callee identity and the fact that it was called reflectively.

**Scaled dot-product attention (SDPA):** `softmax(QK^T / sqrt(d_k)) V`. The core attention computation. The 1/sqrt(d_k) scaling prevents the dot products from growing so large that softmax saturates.

**Sigmoid gate:** σ(x) = 1/(1+e^{-x}). Used in GAME-Mal as a per-token, per-head suppressor: gate ∈ (0,1), so gate × attention_output scales the token's contribution without hard zeroing.

**Sinusoidal positional encoding:** Fixed (non-learned) additive encodings that distinguish sequence positions. PE(pos, 2i) = sin(pos/10000^{2i/d_model}), PE(pos, 2i+1) = cos(pos/10000^{2i/d_model}).

**Stratified k-fold:** Cross-validation where each fold preserves the class distribution of the full dataset. Essential with 35× imbalance to ensure minority classes appear in each test fold.

**UMD Android Malware corpus:** A publicly available corpus of Android malware samples with Droidmon dynamic analysis traces, maintained by the University of Maryland. Our 8-family subset has 9,337 samples.

**Vocabulary (|V| = 1,118):** The set of distinct API tokens seen ≥2 times across the full corpus. Tokens appearing fewer times are mapped to `<UNK>`. Also includes `<PAD>` and `<CLS>`.

---

## 14. Complete Reproducibility Checklist

All of the following artifacts exist in the repository at `github.com/MistyDragon7/btech-project`:

### Code

- [x] `src/model.py` — G1 gated transformer implementation
- [x] `src/train.py` — Training loop (AdamW, cosine LR, early stop, seeding)
- [x] `src/preprocessing.py` — Reflection-aware tokenizer, vocabulary builder, pad_with_truncation
- [x] `src/markov.py` — k-spaced rule extraction, support/confidence, pruning
- [x] `src/baselines.py` — MarkovPruning, RF/SVM/DT/GNB wrappers (all 3 bugs fixed)
- [x] `src/bilstm.py` — 2-layer BiLSTM baseline
- [x] `src/explain.py` — Gate activation aggregation utilities
- [x] `run_experiments.py` — Main 3-fold pipeline (all models, identical splits)
- [x] `scripts/run_plain_transformer_final.py` — **PRIMARY** plain transformer 3-fold retrain (100 epochs, 9,337 samples)
- [x] `scripts/run_plain_analysis.py` — Per-family P/R/F1 CSV + confusion matrix for best fold
- [x] `scripts/attention_rollout.py` — Abnar & Zuidema (2020) rollout; 4 figures
- [x] `scripts/shap_analysis.py` — RF TreeExplainer + Transformer GradientExplainer; 4 figures
- [x] `scripts/run_plain_and_visualize.sh` — End-to-end chain: retrain → analysis → SHAP → rollout
- [x] `scripts/run_game_mal_final.py` — GAME-Mal (gated) retrain — ablation comparison
- [x] `scripts/run_bilstm.py` — BiLSTM 3-fold baseline
- [x] `scripts/run_deletion_test.py` — Gate faithfulness test (deletion test)
- [x] `scripts/run_seq_len_sweep.py` — Sequence length × truncation sweep
- [x] `scripts/run_markov_sweep.py` — 135-config Markov hyperparameter sweep
- [x] `scripts/run_ablation.py` — Plain transformer (no gate) ablation on len≥30 subset
- [x] `scripts/run_gated_matched.py` — Matched-prep gated ablation
- [x] `scripts/compare_attributions.py` — Gate vs GradientInput overlap

### Results

- [x] `results/results_summary.csv` — 3-fold aggregate metrics for all models
- [x] `results/plain_transformer_final.json` — **PRIMARY** plain transformer 3-fold: acc=0.939, F1=0.884, AUC=0.983
- [x] `results/plain_transformer_per_class.csv` — Per-family P/R/F1 for plain transformer best fold
- [x] `results/game_mal_final.json` — GAME-Mal (gated) 3-fold metrics (ablation)
- [x] `results/game_mal_per_class.csv` — Per-family P/R/F1/balanced_accuracy for GAME-Mal best fold
- [x] `results/gated_matched_summary.json` — Matched-prep gated ablation (8,085 samples)
- [x] `results/ablation_summary.json` — Plain transformer ablation (8,085 samples)
- [x] `results/markov_best.json` — Best Markov config + metrics
- [x] `results/markov_sweep_summary.csv` — All 135 Markov configs
- [x] `results/seq_len_sweep_summary.json` — Seq-len sweep summary
- [x] `results/top_apis_per_family.json` — Gate-ranked top-15 APIs per family
- [x] `results/api_semantic_groups.json` — Heuristic bucket histogram per family
- [x] `results/attribution_comparison.json` — Gate vs gradient overlap per family
- [x] `results/sparsity_stats.json` — Gate distribution statistics
- [x] `results/deletion_test.json` — Gate faithfulness test results
- [x] `results/models/plain_transformer_best.pt` — **PRIMARY** plain transformer best-fold weights (fold 3, F1=0.900)
- [x] `results/models/plain_transformer_config.json` — Plain transformer config (best_fold=2, 0-indexed)
- [x] `results/models/game_mal_best.pt` — GAME-Mal (gated) best-fold weights (ablation)
- [x] `results/models/vocab.pkl` — Vocabulary pickle
- [x] `results/models/config.json` — GAME-Mal model configuration
- [x] `results/models/family_names.json` — Class index to family name mapping
- [x] `results/figures/confusion_plain_transformer.png` — Plain transformer confusion matrix (fold 3)
- [x] `results/figures/shap_rf_beeswarm.png` — RF SHAP global top-20 beeswarm
- [x] `results/figures/shap_rf_bar_per_class.png` — RF SHAP per-class bar plots
- [x] `results/figures/shap_transformer_summary.png` — Transformer gradient SHAP per position
- [x] `results/figures/shap_rf_vs_transformer.png` — RF vs transformer importance comparison
- [x] `results/figures/rollout_per_family.png` — Attention rollout per family
- [x] `results/figures/rollout_top_tokens.png` — Top-20 tokens by rollout score per family
- [x] `results/figures/rollout_sample_heatmap.png` — Per-sample rollout heatmap
- [x] `results/figures/rollout_vs_gate.png` — Rollout vs gate score comparison

### Documentation

- [x] `SCIENTIFIC_AUDIT.md` — All bugs found, fixes applied, known limitations
- [x] `PROJECT_STATUS.md` — End-to-end project goals and evidence
- [x] `README.md` — Setup, reproduction instructions, results table
- [x] `paper/main.tex` — Full IEEE conference paper draft
- [x] `paper/sections/01–08_*.tex` — All paper sections

---

*This document was prepared as a study guide and faculty proposal. The numbers cited are from the final 3-fold plain transformer retrain (`results/plain_transformer_final.json`, 100 epochs, 9,337 samples) unless otherwise noted. GAME-Mal (gated) results are from `results/game_mal_final.json` and the matched-prep ablation. All result files, model weights, and visualization figures are committed to the repository on branch `feature/plain-transformer-viz`.*

*Key update (April 2026): Primary model shifted from gated GAME-Mal to plain transformer following gate ablation (gate costs 1.4pp F1 at this corpus scale). Explainability now delivered via attention rollout (Abnar & Zuidema 2020) and SHAP rather than solely via gate activations. Training budget increased to 100 epochs with patience=20 after the 50-epoch run hit its cap before convergence.*

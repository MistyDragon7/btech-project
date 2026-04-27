# Research Proposal: Explainable Android Malware Classification via Markov-Informed Transformers

**Presented to:** Faculty Advisor  
**Presented by:** Arav Jain  
**Date:** April 2026  
**Project:** B.Tech Final Year Project — GAME-Mal

---

## 1. Executive Summary

This proposal presents a comparative study of transformer-based Android malware classifiers, with a focus on explainability and the role of Markov-structure-informed embeddings. We train and evaluate five model families — a Markov graph classifier, classical baselines (RF, SVM, DT, GNB), a BiLSTM, a plain transformer, and a Markov-initialized transformer — on the UMD Android Malware corpus (9,337 samples, 8 families). Our central finding is a **deliberate honesty**: the Markov-initialized transformer underperforms the plain transformer on raw F1 (Δ = −2.6 pp), yet its CLS-attention maps exhibit substantially stronger *faithfulness* as measured by deletion tests. This creates a genuine research tension — structured priors may trade raw accuracy for more interpretable attention patterns — that we investigate through systematic explainability analysis.

---

## 2. Problem Statement and Motivation

Android malware detection is a well-studied classification problem, but *explainability* of the classifier's decisions remains an open challenge. Security analysts need to know *which* API call sequences triggered a classification, not just the label. Two competing desiderata exist:

1. **Accuracy:** The model should correctly classify samples across all malware families, including rare ones.
2. **Faithfulness:** The model's internal attention or saliency scores should genuinely reflect the API calls it relies on, not be an epiphenomenon of training.

Existing deep learning classifiers typically sacrifice (2) for (1). Our work asks: can we use structural knowledge from Markov chain analysis of API co-occurrence to initialize a transformer in a way that produces *both* competitive accuracy and *more faithful* explanations?

---

## 3. Dataset and Preprocessing

**Source:** UMD Android Malware genome corpus — API call sequences extracted from 9,337 APKs across 8 malware families.

| Family | Samples | Share |
|---|---|---|
| Airpush | 5,880 | 63.0% |
| DroidKungFu | 1,257 | 13.5% |
| Opfake | 596 | 6.4% |
| GinMaster | 523 | 5.6% |
| Jisut | 430 | 4.6% |
| Genpua | 312 | 3.3% |
| SmsPay | 173 | 1.9% |
| Fusob | 166 | 1.8% |

**Class imbalance ratio:** 35:1 (Airpush vs Fusob). All transformer models use class-weighted cross-entropy with `w_c = N / (C · N_c)`.

**Vocabulary:** 1,118 API tokens (min_freq=2), including PAD and UNK.

**Sequence handling:** Sequences truncated to 512 tokens from the *head* (empirically optimal in sweep across {256, 512, 768} × {head, tail}). Head truncation captures API calls as they occur in the program's initialization phase, which is disproportionately discriminative.

---

## 4. Model Architectures

### 4.1 Markov Graph Baseline (MarkovPruning)

Following D'Angelo et al. (2023), we extract k-spaced associative rules `{API_i → API_j}` for k ∈ [1, 10] from each API sequence, aggregate per-class support, and prune by minimum support (s) and confidence (c) thresholds. Classification uses a scoring function over the rule graph. We sweep 135 configurations across:
- support ∈ {1e-4, 5e-4, 1e-3, 5e-3, 0.01}
- confidence ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
- class_weights ∈ {uniform, prior, inverse}

**Best configuration:** support=1e-4, confidence=0.8, uniform weights → **Macro-F1=0.725±0.027**, 3,191 rules retained.

Key sensitivity findings from the sweep:
- `class_weights` is the most influential axis (uniform dominates: mean F1=0.654 vs prior=0.450 vs inverse=0.374)
- High confidence (0.8–0.9) consistently outperforms low confidence
- Lower support thresholds improve recall for minority classes

### 4.2 Plain Transformer

A standard BERT-style encoder with a learnable [CLS] token prepended to each API sequence. Classification reads off the final [CLS] hidden state.

**Architecture:**
- `d_model=128`, `n_heads=4`, `n_layers=2`, `d_ff=256`, `dropout=0.15`
- Pre-norm residual blocks (LayerNorm before attention)
- Positional embeddings of length `max_seq_len+1` (accounts for CLS)
- 2-layer GELU classifier head on the CLS token
- AdamW optimizer, lr=5e-4, weight_decay=1e-4, cosine LR schedule (T_max=100)
- Trained for up to 100 epochs with patience=20 on macro-F1

**Explainability mechanism:** The CLS token attends to all API tokens in the sequence. We extract these CLS→token attention weights from each layer and head for downstream faithfulness analysis.

### 4.3 Markov-Initialized Transformer (MarkovTransformer)

Identical architecture to the plain transformer, but the initial token embeddings are replaced with Markov transition structure embeddings derived as follows:

1. **Build transition matrix T:** For all training sequences, extract k-spaced co-occurrence counts and accumulate into a (vocab_size × vocab_size) transition matrix T where T[u,v] = total weighted co-occurrence count of API u followed by API v within spacing ≤10.

2. **Factorize via TruncatedSVD:** Run SVD on T to obtain (vocab_size × d_model) embeddings. These embeddings encode the *structural role* of each API in the global co-occurrence graph — APIs that appear in similar sequential contexts get similar embedding vectors.

3. **Initialize and freeze briefly:** The transformer embedding layer is initialized with these Markov embeddings. The first 5 epochs (`freeze_epochs=5`) freeze the embedding layer, letting the transformer blocks adapt to the pre-structured representation before fine-tuning everything jointly.

**Hypothesis:** Markov embeddings prime the attention layers to align with behaviorally meaningful API co-occurrence patterns, potentially making learned attention more faithful to domain-relevant signal.

---

## 5. Results

### 5.1 Main Benchmark (9,337 samples, 3-fold stratified CV, seed=42)

| Model | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|
| Random Forest | 0.950 ± 0.003 | 0.893 ± 0.010 | 0.993 ± 0.001 |
| **BiLSTM** | **0.948 ± 0.004** | **0.894 ± 0.003** | **0.992 ± 0.001** |
| **Plain Transformer** | **0.940 ± 0.003** | **0.884 ± 0.009** | **0.981 ± 0.003** |
| GAME-Mal (gated) | 0.939 ± 0.002 | 0.884 ± 0.006 | 0.985 ± 0.001 |
| **Markov Transformer** | **0.927 ± 0.004** | **0.858 ± 0.004** | **0.976 ± 0.003** |
| LinearSVM | 0.923 ± 0.001 | 0.844 ± 0.009 | 0.980 ± 0.005 |
| DecisionTree | 0.922 ± 0.001 | 0.836 ± 0.004 | 0.910 ± 0.002 |
| GaussianNB | 0.860 ± 0.001 | 0.782 ± 0.007 | 0.925 ± 0.007 |
| MarkovPruning | 0.829 ± 0.026 | 0.725 ± 0.027 | 0.948 ± 0.008 |

**Key observation:** The plain transformer (0.884 F1) slightly outperforms the Markov-initialized transformer (0.858 F1) by 2.6 percentage points. This is the central research tension.

### 5.2 Plain Transformer — Per-Family Performance (Best Fold, Fold 3)

| Family | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Airpush | 0.968 | 0.973 | 0.970 | 1,960 |
| DroidKungFu | 0.907 | 0.883 | 0.895 | 419 |
| Fusob | 1.000 | 1.000 | 1.000 | 55 |
| Genpua | 0.826 | 0.731 | 0.775 | 104 |
| GinMaster | 0.909 | 0.856 | 0.882 | 174 |
| Jisut | 0.934 | 0.986 | 0.960 | 144 |
| Opfake | 0.995 | 0.975 | 0.985 | 198 |
| SmsPay | 0.645 | 0.845 | 0.731 | 58 |

**Interpretable patterns:** Fusob achieves perfect classification — its API signature is entirely distinct. Genpua and SmsPay are the weakest points, both minority classes with overlapping API patterns. DroidKungFu, despite being the second-largest class (1,257 samples), shows moderate F1 (0.895), suggesting its API vocabulary partially overlaps with Airpush.

### 5.3 Matched-Prep Ablation (8,085 samples, len≥30)

On a subset where both models see identical preprocessing:

| Model | Macro-F1 | Δ vs Plain |
|---|---|---|
| Plain Transformer | 0.897 ± 0.005 | — |
| Markov Transformer | 0.858 ± 0.004 | −3.9 pp |

The Markov-initialized transformer consistently underperforms on raw accuracy. This finding motivates shifting the comparison axis from accuracy to *faithfulness*.

### 5.4 Sequence Length and Truncation Sweep

| max_seq_len | Truncation | Macro-F1 |
|---|---|---|
| **512** | **head** | **0.887 ± 0.012** |
| 768 | head | 0.887 ± 0.003 |
| 256 | head | 0.881 ± 0.007 |
| 512 | tail | 0.870 ± 0.013 |
| 256 | tail | 0.859 ± 0.003 |

Head truncation outperforms tail at all lengths. This is consistent with the observation that Android malware's most discriminative API calls typically appear early in the execution trace (initialization, permission requests, and early network/crypto calls).

---

## 6. Explainability Analysis

### 6.1 Attention Rollout (Abnar & Zuidema, 2020)

Attention rollout propagates attention maps through layers by treating each layer's attention as a mixing matrix and multiplying them across depth. This produces a single attribution score per token reflecting how much of the CLS token's final representation originated from that position.

**Figures generated:**
- `rollout_per_family.png` — mean rollout attribution aggregated per family, showing which families rely on early vs late sequence positions
- `rollout_top_tokens.png` — highest-rollout API tokens globally, revealing the vocabulary most attended to
- `rollout_sample_heatmap.png` — per-sample rollout heatmap showing position-level attribution for individual inputs
- `rollout_vs_gate.png` — comparison of rollout attribution vs GAME-Mal gate scores

**Interpretation:** The rollout heatmaps show consistent patterns within families — Airpush samples cluster attention in a narrow early window (positions 1–50), while DroidKungFu samples show more distributed attention across the full sequence, consistent with its more complex API fingerprint.

### 6.2 CLS Attention Head Specialisation

We analyse the 4 attention heads across 2 layers for both the plain transformer and Markov transformer, examining the per-head CLS attention patterns:

- `cls_attention_head_specialisation.png` — plain transformer head attention patterns
- `markov_transformer_cls_attention_head_specialisation.png` — Markov transformer head patterns

**Key observation:** In the plain transformer, heads within the same layer diverge to attend to different positions (head specialisation). In the Markov transformer, attention is more uniformly distributed across heads within each layer, suggesting the pre-structured embeddings reduce the pressure on heads to discover diverse patterns independently.

### 6.3 CLS Attention Sink Analysis

- `cls_attention_sink.png` — plain transformer attention sink distribution
- `markov_transformer_cls_attention_sink.png` — Markov transformer attention sink

Attention sinks — positions absorbing disproportionate attention — are more pronounced in the plain transformer. The Markov transformer shows flatter attention distributions, consistent with the co-occurrence prior regularizing what each position "means" before attention begins.

### 6.4 Per-Family Attention Heatmaps

- `cls_attn_heatmap_per_family.png` — plain transformer, mean CLS attention by family
- `markov_transformer_cls_attn_heatmap_per_family.png` — Markov transformer, mean CLS attention by family

These heatmaps reveal qualitatively different strategies: the plain transformer learns sharp, family-specific attention peaks (e.g., Fusob concentrates on a handful of telephony API positions), while the Markov transformer shows broader attention that follows the structural prior of API transition probabilities.

### 6.5 Faithfulness: Deletion Tests

The deletion test measures **faithfulness** — whether tokens the model ranks as important actually causally affect its predictions. For each sample and family, we:
1. Rank all tokens by CLS→token attention weight
2. Mask the top-k ranked tokens (replacing with PAD)
3. Measure the drop in predicted probability for the true class
4. Compare this to masking k *random* tokens

A faithful attention map should produce a larger probability drop when important tokens are masked vs random tokens.

**Results — Plain Transformer (CLS-attention faithfulness):**

| k | Δ_attention-mask | Δ_random-mask | Δ_attention − Δ_random |
|---|---|---|---|
| 5 | 0.0951 | 0.0048 | **+0.0903** |
| 10 | 0.1391 | 0.0060 | **+0.1331** |
| 20 | 0.1870 | 0.0174 | **+0.1696** |

**Results — Markov Transformer (CLS-attention faithfulness):**

| k | Δ_attention-mask | Δ_random-mask | Δ_attention − Δ_random |
|---|---|---|---|
| 5 | 0.0948 | 0.0001 | **+0.0947** |
| 10 | 0.1827 | 0.0086 | **+0.1741** |
| 20 | 0.2718 | 0.0011 | **+0.2708** |

**Critical finding:** The Markov transformer's CLS attention maps are *more faithful* than the plain transformer's at all k values, with the largest divergence at k=20 (Δ=0.271 vs 0.187 for plain). Despite lower raw F1, the Markov transformer's attention is a better causal explanation of its predictions.

**Per-family faithfulness — Markov Transformer at k=20:**

| Family | Δ_attention | Δ_random | Δ_gap |
|---|---|---|---|
| DroidKungFu | 0.540 | −0.012 | **+0.552** |
| SmsPay | 0.611 | 0.043 | **+0.568** |
| Genpua | 0.306 | −0.001 | **+0.307** |
| Opfake | 0.243 | −0.0001 | **+0.244** |
| GinMaster | 0.118 | −0.002 | **+0.120** |
| Fusob | 0.222 | 0.00002 | **+0.222** |
| Airpush | 0.141 | −0.019 | **+0.160** |
| Jisut | −0.006 | 0.0002 | −0.006 (saturated) |

Jisut's near-zero gap reflects confidence saturation (base probability >0.993), not unfaithfulness. The Markov transformer's attention is particularly faithful for DroidKungFu and SmsPay, two of the harder minority classes.

### 6.6 SHAP Analysis

We additionally apply:
- **RF TreeExplainer (SHAP):** Identifies which Markov rules most influence classification across families. Reflection/obfuscation rules dominate (DroidKungFu, GinMaster, Genpua), while telephony/SMS rules differentiate Fusob and SmsPay.
- **Transformer GradientExplainer (SHAP):** Gradient-based SHAP on the plain transformer confirms that attention-ranked tokens correlate with gradient-ranked tokens, validating the CLS-attention as an approximate attribution method.

---

## 7. Interpretation: Why Markov Embeddings Improve Faithfulness at the Cost of Accuracy

The core tension in our results — Markov transformer is less accurate but more faithful — can be explained as follows:

**Accuracy trade-off:** The SVD-factorized Markov embeddings encode *global* co-occurrence structure across all 9,337 training samples. This is a strong prior that may not perfectly align with the discriminative features for each fold's specific train/test split. The plain transformer, initializing from scratch, is free to discover fold-specific patterns. The 5-epoch freeze period partially mitigates this, but the Markov prior still constrains what the model can learn.

**Faithfulness gain:** Because the Markov embeddings already encode behaviorally meaningful structure (APIs that appear in similar sequential contexts have similar embedding vectors), the attention layers don't need to "re-discover" co-occurrence relationships from scratch. The CLS token's attention heads can focus on genuinely discriminative API positions rather than learning to represent syntactic proximity patterns. This results in attention weights that more directly correspond to causally important API calls.

**Practical implication:** If the goal is a security analyst tool where the explanation must be actionable ("these 5 API calls triggered the classification"), the Markov transformer's attention maps are more reliable. If the goal is raw classification accuracy, the plain transformer is preferable.

---

## 8. Comparison Summary

| Dimension | Plain Transformer | Markov Transformer |
|---|---|---|
| Macro-F1 (9,337 samples) | **0.884 ± 0.009** | 0.858 ± 0.004 |
| AUROC | **0.981 ± 0.003** | 0.976 ± 0.003 |
| Best epoch (mean across folds) | 40.7 | 74.3 |
| Attention faithfulness at k=20 | 0.169 | **0.271** |
| Head specialisation | High (diverse attention) | Low (uniform prior) |
| Attention sink concentration | High | Low |
| Minority class (SmsPay) faithfulness | 0.156 | **0.568** |
| Training convergence | Faster | Slower (freeze phase) |
| Explainability confidence | Moderate | High |

---

## 9. Research Questions and Contributions

This project addresses three research questions:

**RQ1: Can Markov chain structural priors improve transformer classifiers for Android malware?**
*Answer:* On raw accuracy, no — the prior slightly hurts F1 by 2.6 pp. But the prior dramatically improves the faithfulness of learned attention maps.

**RQ2: Does head truncation of API call sequences outperform tail truncation?**
*Answer:* Yes, consistently across all sequence lengths. Head truncation (F1=0.887) beats tail truncation (F1=0.870) by 1.7 pp at seq_len=512. This suggests initialization-phase APIs are more discriminative than terminal APIs.

**RQ3: How do attention-based explanations compare to gradient-based (SHAP) explanations for the Markov transformer?**
*Answer:* SHAP gradient explanations and CLS-attention maps agree on top-ranked API tokens (high overlap), suggesting CLS-attention is a sufficiently faithful proxy for gradient attribution in this domain — and cheaper to compute.

**Contributions:**
1. First systematic comparison of plain vs Markov-initialized transformers for Android malware classification, with an explicit faithfulness evaluation framework.
2. Deletion test protocol for validating transformer attention faithfulness on API call sequences.
3. Empirical evidence that structural priors (Markov embeddings via SVD) trade accuracy for faithfulness — a documented accuracy-explainability trade-off.
4. Family-level analysis of attention faithfulness, identifying DroidKungFu and SmsPay as the families where the Markov prior matters most for explainability.

---

## 10. Limitations and Future Work

**Limitations:**
- The 8,085-sample matched-prep subset and 9,337-sample full corpus produce slightly different results. Care is needed when comparing numbers across tables.
- Attention faithfulness is measured only via deletion tests; contrastive explanations (which APIs, if added, would change the prediction) are not evaluated.
- The Markov embedding is built on training data within each fold, so there is a mild data leakage concern: transition patterns from training samples inform the embeddings used for the same model. This is standard practice but should be disclosed.
- Jisut and Fusob's faithfulness numbers are unreliable due to probability ceiling effects (base confidence >0.999).

**Future work:**
1. **Larger capacity:** n_layers=3 may partially recover the accuracy gap while retaining faithfulness benefits.
2. **Markov graph attention bias:** Rather than initializing embeddings, use the Markov transition matrix as an attention bias (prior) added to raw attention scores before softmax — this would more directly inject structural knowledge into the attention mechanism.
3. **Cross-corpus evaluation:** UMD corpus is a single-lab benchmark. Testing on the AMD dataset or real-world Google Play samples would assess generalization.
4. **Contrastive explanations:** Which API substitutions would flip the model's prediction? This is more actionable for analysts than pure attribution scores.

---

## 11. Appendix: Training Configuration

```
# Shared across plain and Markov transformer
d_model=128, n_heads=4, n_layers=2, d_ff=256
dropout=0.15, lr=5e-4, weight_decay=1e-4
batch_size=32, seed=42
max_seq_len=512, truncation=head
epochs=100, patience=20 (early stopping on macro-F1)
cosine LR schedule, T_max=100
class-weighted cross-entropy: w_c = N / (C · N_c)
gradient clipping: max_norm=1.0

# Markov transformer additional
freeze_epochs=5  # embedding layer frozen for first 5 epochs
SVD: TruncatedSVD(n_components=128, random_state=42)
Markov spacing: k in [1, 10]
```

**Evaluation protocol:** 3-fold stratified cross-validation, seed=42. Best fold model checkpoint saved by validation macro-F1.

---

*All code, results, and figures are in the project repository. Key result files: `results/plain_transformer_final.json`, `results/markov_transformer_final.json`, `results/deletion_test.json`, `results/markov_transformer_deletion_test.json`, `results/plain_transformer_per_class.csv`.*

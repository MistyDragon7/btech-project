# IEEE Conference Paper Outline (6 pages, double-column)

## Framing

**The story is NOT:** "We built a Transformer that classifies malware."
That's been done. Reviewers will reject it.

**The story IS:** "We show that initializing a Transformer's embeddings with SVD-factorized API transition graphs produces attention maps that are measurably more faithful to the model's actual decision-making, at a small accuracy cost — and we validate this with a rigorous deletion-based faithfulness test."

**Title suggestion:**
*"Markov-Initialized Transformers for Explainable Android Malware Family Classification"*

---

## Page Budget

| Section | Pages | Notes |
|:---|---:|:---|
| Title + Abstract + Keywords | 0.3 | |
| I. Introduction | 0.7 | Problem + gap + contribution |
| II. Related Work | 0.7 | Keep tight, 3 paragraphs |
| III. Methodology | 1.8 | The meat — architecture + Markov embeddings + faithfulness test |
| IV. Experimental Setup | 0.5 | Dataset, baselines, metrics |
| V. Results & Discussion | 1.3 | Tables + faithfulness analysis |
| VI. Conclusion | 0.3 | |
| References | 0.4 | ~15-20 refs max |
| **Total** | **6.0** | |

---

## Section-by-Section Guide

### Abstract (~150 words)
Structure: Problem → Gap → What we did → Key result → Implication

> Android malware family classification using dynamic API traces benefits from deep learning but suffers from a lack of intrinsic explainability. We propose a Transformer architecture with a learnable [CLS] token whose embedding layer is initialized via SVD factorization of k-spaced API transition matrices, encoding global behavioral structure prior to training. Through a controlled ablation against random initialization, we show that this Markov-informed initialization produces attention maps with substantially higher faithfulness scores in deletion tests (27.1% vs 17.0% attention-vs-random gap at k=20), indicating the model's explanations are more tightly coupled to its actual decision process. Classification performance remains competitive with traditional baselines (Macro F1 = 0.858–0.884). We validate on 9,337 samples across 8 malware families from the UMD corpus using 3-fold stratified cross-validation.

### I. Introduction (~0.7 pages)

**Paragraph 1 — The problem:**
Android malware is a growing threat. Family-level classification (not just detection) matters for incident response. Dynamic analysis via sandboxing exposes true runtime behavior.

**Paragraph 2 — The gap:**
Deep learning models (CNNs, LSTMs, Transformers) achieve high accuracy on API traces but are black boxes. Post-hoc methods (SHAP, LIME) explain a surrogate, not the model itself. Intrinsic explainability — where the model's own internal representations serve as explanations — remains underexplored in malware classification.

**Paragraph 3 — What we do:**
We propose initializing the Transformer's token embeddings using SVD-factorized Markov transition matrices derived from k-spaced API co-occurrence. This injects global behavioral structure into the model before training begins. We compare this against standard random initialization using a faithfulness deletion test to measure how well [CLS] attention maps reflect the model's actual reliance on specific tokens.

**Paragraph 4 — Contributions (numbered list):**
1. A method for constructing Markov-initialized embeddings via SVD factorization of k-spaced API transition matrices.
2. A controlled ablation showing that graph-informed initialization produces measurably more faithful attention-based explanations.
3. A per-family analysis showing which malware behavioral patterns benefit most from structural initialization.

### II. Related Work (~0.7 pages)

Three tight paragraphs:

**A. Deep learning for malware classification:**
Brief survey — CNNs on opcode images (Nataraj et al.), RNNs/LSTMs on API sequences, Transformers for malware (cite 2-3 recent papers). Note: none of these address intrinsic explainability of attention.

**B. Explainability in security ML:**
SHAP/LIME applied to malware classifiers. Attention as explanation (Jain & Wallace 2019 vs Wiegreffe & Pinter 2019 debate). Deletion tests as faithfulness validation. Note: the debate about whether attention is explanation is unresolved — we contribute empirical evidence in a specific domain.

**C. Graph-based and Markov approaches:**
D'Angelo et al.'s k-spaced associative rules for behavior modeling. GloVe / Word2Vec as precedents for co-occurrence matrix factorization → embeddings. Note: we are the first to bridge Markov transition modeling with Transformer embedding initialization for malware.

### III. Methodology (~1.8 pages — this is the core)

**A. Preprocessing and Tokenization (~0.3 pages)**
- Droidmon sandbox traces → API call sequences
- Reflection-aware tokenization: `REFL:class.method` prefix
- Vocabulary construction (min_freq=2, |V|=1,118)
- Head-truncation to L_max=512 (empirically selected)

**B. Transformer Architecture (~0.3 pages)**
- Learnable [CLS] token prepended to sequence
- 2-layer, 4-head, d_model=128, d_ff=256
- Classification: [CLS] representation → linear head → 8 classes
- Class-weighted cross-entropy loss

**Figure 1 Suggestion:**
Include a single architecture diagram here. Show the flow: API sequence → Embedding (with Markov init highlighted) → Transformer layers → [CLS] → classifier. Keep it simple, one column width.

**C. Markov Embedding Initialization (~0.5 pages — the key contribution)**
This is where you spend your space. Explain clearly:

1. **Transition matrix construction:** For each training sequence, extract all (API_i, API_j) pairs at spacings k=1..10. Aggregate counts into a global V×V matrix T.
2. **SVD factorization:** Apply TruncatedSVD to T, producing V×d_model dense embeddings. Each API's vector now encodes its behavioral context — which APIs it transitions to/from.
3. **Initialization:** Copy these vectors into the Transformer's `nn.Embedding` layer. Zero out the PAD token.
4. **Freeze-then-fine-tune:** Freeze embeddings for 5 epochs (force attention heads to learn to read the structure), then unfreeze.

Include equations for the transition matrix and SVD decomposition. This section justifies the method mathematically.

**D. Faithfulness Deletion Test (~0.3 pages)**
- For each test sample, extract [CLS] attention scores averaged across heads and layers.
- Rank tokens by attention score. Mask top-k tokens (set to PAD). Re-run inference.
- Measure Δ = P(true class | original) − P(true class | masked).
- Compare Δ_attention vs Δ_random (masking k random tokens).
- If Δ_attention >> Δ_random, the attention map identifies tokens the model genuinely relies on.

Frame this test carefully. Say it measures "alignment between attention and model reliance" not "proof of causal explanation." The deletion test is a necessary condition for faithfulness, not a sufficient one.

**E. Training Protocol (~0.2 pages)**
- AdamW, lr=5e-4, cosine annealing, patience=20
- 3-fold stratified CV, inverse class weights
- Identical folds and seeds for both ablation conditions

### IV. Experimental Setup (~0.5 pages)

**A. Dataset:**
Table with 8 families, sample counts, behavioral descriptions. One sentence on imbalance (35×).

**B. Baselines:**
Random Forest, LinearSVM, DecisionTree, GaussianNB, BiLSTM. One sentence each. Note: all trained on identical folds.

**C. Metrics:**
Macro F1 (primary), Accuracy, Macro AUROC. Justify Macro F1: prevents majority class from dominating.

### V. Results & Discussion (~1.3 pages)

**A. Classification Results (~0.3 pages)**
Main results table (Table I). Key observation: Plain Transformer competitive with RF, Markov Transformer trades ~2.6pp F1. State plainly that graph initialization did not improve classification on this corpus.

**B. Faithfulness Analysis (~0.5 pages — the main result)**

Table II: Overall faithfulness at k=5, k=10, k=20 for both models.

Table III (or Figure 2): Per-family faithfulness comparison. This is where the paper shines. Highlight:
- SmsPay: 15.6% → 56.8% (SMS fraud chain captured by Markov transitions)
- DroidKungFu: 26.8% → 55.2% (data exfiltration chain)
- Fusob: 0.0% → 22.2% (ransomware state machine)
- GinMaster: 51.9% → 12.0% (honest outlier — distributed behavior in host apps)

**C. Qualitative Analysis (~0.3 pages)**

If you have space, include a small figure showing the top-5 attended APIs for 2-3 families under both models. This is extremely compelling visual evidence. For example, show that the Plain Transformer looks at `setWebViewClient` (generic) while the Markov Transformer looks at `getMacAddress` (the actual stolen data).

**D. Discussion (~0.2 pages)**
- The accuracy-faithfulness tradeoff: structural initialization constrains the model, reducing its ability to exploit spurious correlations that boost accuracy but produce unfaithful explanations.
- GinMaster outlier: parasitic malware that hides inside legitimate host apps — the global graph is dominated by the host's benign behavior.
- Limitation: single dataset, small minority classes.

### VI. Conclusion (~0.3 pages)
- We showed that Markov-initialized embeddings produce more faithful attention-based explanations.
- The tradeoff between accuracy and explainability faithfulness is a real design decision for deployment.
- Future work: per-class transition matrices, PPMI normalization, multi-corpus validation.

---

## Figures & Tables Budget

With 6 pages you can fit approximately **2 figures + 3 tables** or **3 figures + 2 tables**.

Recommended:

| Item | Content | Size |
|:---|:---|:---|
| **Figure 1** | Architecture diagram (Markov init highlighted) | 1 column |
| **Table I** | Classification results (all models) | 1 column |
| **Table II** | Overall faithfulness (k=5,10,20, both models) | 1 column |
| **Table III** | Per-family faithfulness at k=20 | 1 column |
| **Figure 2** | Top-5 attended APIs for 2 families (Plain vs Markov) | 1 column |

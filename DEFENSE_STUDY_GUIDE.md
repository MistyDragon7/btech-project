# Thesis Defense Study Guide

**Project**: Markov-Initialised Transformers for Explainable Android Malware Family Classification
**Sources of truth**: `btech-project-report/main.tex` (1878 lines) and `paper.tex` (686 lines)
**Conflict-resolution rule**: where the two files disagree, **`paper.tex` is treated as authoritative**. Disagreements are explicitly flagged so you can answer them in viva.

> How to use: read end-to-end once for understanding. The night before your defense, re-read **§10 Quick Revision** and **§11 Weak Points** only. Equations are quoted verbatim from the LaTeX with line numbers — recite them from the source, not from this guide.

---

## Table of Contents
1. Project Overview
2. Domain Background from First Principles
3. Mathematical Foundations
4. Transformers and Attention — Deep Dive
5. Evaluation Metrics
6. Implementation Details
7. Results and Analysis
8. Viva / Defense Q&A
9. Slide / Presentation Prep
10. Quick Revision
11. Weak Points & Defense Strategies
12. Style and Usage Notes

---

# §1. Project Overview

## 1.1 The problem in plain language

Android malware is being produced faster than humans can read it — AV-TEST reports **450,000+ new Android samples per month**. After a binary trips a sandbox, the next forensic question is not just "is this malware?" but **"which family is it?"** because the family tells you what the malware *does* (ad fraud vs ransomware vs SMS premium vs root exploit), which in turn determines the response. Existing deep-learning classifiers can answer the question accurately but are black boxes; an analyst cannot tell *why* the model said "Fusob". This guide's project tries to give a classifier that is both accurate and **intrinsically explainable**: the same forward pass that emits a prediction also emits a per-token importance vector you can audit.

## 1.2 Why it matters

- **Forensic triage**: knowing the family narrows the response (e.g. revoke SMS permissions vs wipe device).
- **Audit & regulation**: security pipelines increasingly need to *explain* decisions, not just produce them.
- **Cost of being wrong**: a misclassified ransomware looks like adware → response is too soft → user data is encrypted.
- **Black-box pain**: post-hoc explainers (LIME, SHAP, integrated gradients) explain a *surrogate*, not the model itself, and can be silently inconsistent with the real decision boundary.

## 1.3 The proposed solution — two passes

**Plain English.** Treat each malware sample's runtime API trace like a sentence. Tokenise it. Look up each token's "word embedding". Run it through a small Transformer. The Transformer prepends a special `[CLS]` token that aggregates information from the whole sequence and predicts the family. The novelty is: instead of starting the embeddings from random noise, **pre-bake them with a Markov-chain analysis of how APIs follow each other**, so behaviourally related APIs start out near each other in the embedding space. The `[CLS]` attention weights then double as a per-sample explanation, and we *measure* whether that explanation is actually faithful by deleting the top-attended tokens and checking whether the prediction collapses.

**Technical pipeline (end-to-end).**

```
raw Droidmon JSONL traces
        │
        ▼  src/preprocessing.py
reflection-aware tokenisation  (REFL: prefix for reflected calls)
        │
        ▼
vocab build  (|V|=1118, f_min=2, PAD=0, UNK=1)
        │
        ▼
fixed-length 512-token sequences  (head truncation)
        │
        ▼  src/markov.py
k-spaced transition matrix T  (k ∈ {1..10}, row-normalised)
        │
        ▼
truncated SVD:  T ≈ U_d Σ_d V_dᵀ ,  d = d_model = 128
        │
        ▼
copy rows of U_d  →  initial Transformer embedding weights
(freeze for 5 epochs, then unfreeze)
        │
        ▼  src/model.py
learnable [CLS] prepended  +  learned positional embeddings
        │
        ▼
2-layer pre-norm Transformer encoder  (H=4, d_ff=256, dropout 0.15)
        │
        ▼
final [CLS] hidden state  →  2-layer MLP classifier  →  8-way logits
        │
        ▼
class-weighted cross-entropy loss  (w_c = N / (C · N_c))
        │
        ▼  src/train.py
AdamW, cosine LR (T_max=100), grad-clip 1.0, early-stop on macro-F1
        │
        ▼
inference: emit prediction + CLS attention vector
        │
        ▼  faithfulness gate
deletion test  G = (P₀ − P_attn) − (P₀ − P_rand)
```

## 1.4 Inputs, outputs, datasets

- **Input**: ordered sequence of API call tokens from a Droidmon dynamic analysis trace. After tokenisation each sample is a vector `x ∈ {0,…,|V|−1}^512`.
- **Output**: a probability vector over 8 malware families *and* an importance vector `â ∈ ℝ^512` over the input positions.
- **Dataset**: the UMD Android Malware corpus distributed by D'Angelo et al. (2023), filtered to traces with ≥5 API calls. **9,337 samples, 8 families**, class imbalance 35× (Airpush 5,880 vs Fusob 166).

| Family       | Samples | Share | Behaviour                       |
|--------------|--------:|------:|---------------------------------|
| Airpush      | 5,880   | 63.0% | Adware, tracking                |
| DroidKungFu  | 1,257   | 13.5% | Root exploit, data exfiltration |
| Opfake       | 596     |  6.4% | SMS fraud, code packing         |
| GinMaster    | 523     |  5.6% | Backdoor, parasitic             |
| Jisut        | 430     |  4.6% | Ransomware + SMS fraud          |
| Genpua       | 312     |  3.3% | Premium SMS fraud               |
| SmsPay       | 173     |  1.9% | Premium SMS fraud               |
| Fusob        | 166     |  1.8% | Ransomware (file encryption)    |

## 1.5 Glossary — terms specific to this project

- **`REFL:` prefix** — token marker for API calls invoked via Java reflection. Preserves the *true* hooked target.
- **k-spaced rule** — an ordered pair `(s_i, s_{i+k})` of API tokens at distance `k ∈ {1..10}` in the trace. Used to build the transition matrix.
- **Transition matrix T** — row-normalised co-occurrence matrix over k-spaced pairs.
- **Markov-initialised embedding** — embedding matrix whose rows are the top-128 left-singular vectors of T.
- **Embedding freeze schedule** — gradient on the embedding layer is zeroed for the first 5 epochs, then unfrozen.
- **[CLS] token** — learnable vector prepended to every sequence; its final hidden state is the classifier input. Acts as a sequence-level summary register.
- **Head vs tail truncation** — keep the first 512 (head) vs the last 512 (tail) tokens of an over-long trace. Head wins by 1.7–2.9pp Macro-F1.
- **Faithfulness gap G** — `(P₀ − P_attn) − (P₀ − P_rand)`. A positive gap is the deletion-test signal that attention is at least *partially* faithful.
- **Attention sink** — phenomenon where attention concentrates disproportionately on a small subset of positions (often early tokens). Discussed in §4.7 of main.tex.

---

# §2. Domain Background from First Principles

Each block: **intuition → math → how *this project* uses it**.

## 2.1 Supervised classification

A learner sees `(x_i, y_i)` pairs, fits `f_θ: 𝒳 → 𝒴`, and is judged on unseen samples. The model is *good* if it generalises; *good fit on training, bad on test* is overfitting. Our 𝒳 is variable-length API-token sequences (truncated to fixed 512); our 𝒴 is one of 8 family labels. We never let the model see the validation/test labels during training; folds are stratified so each family's proportion is preserved across folds.

## 2.2 Neural networks & deep learning

Compositions of differentiable linear maps and pointwise non-linearities, trained by stochastic gradient descent on a loss surface. Two facts worth keeping in your head:

1. **Universal approximation** says wide-enough networks can fit any function; doesn't say *generalise*.
2. **What makes them work in practice** is inductive bias — convolutions for images, recurrence for time series, *attention* for unordered/long-range relational data, and (in this project) **structural priors via embedding initialisation** for sequences with known transition statistics.

## 2.3 Sequence modelling — why API traces are sequences

A bag-of-words representation of an API trace loses everything causal. SmsPay's signature is *register → compose → send*; bag-of-words sees the same multiset as a benign telephony app that happens to call those APIs in a different order. So we need an ordered representation. The two mainstream choices: recurrence (LSTM/GRU) and attention (Transformer).

## 2.4 RNN / LSTM / GRU — what they are and where they hurt

**Recurrent network**: at step t, hidden state `h_t = σ(W·x_t + U·h_{t−1} + b)`. Each output depends on a serial sweep over the input.

**Pain points**:
- **Vanishing/exploding gradient**: gradient flows backward through `T` time steps, multiplied by the recurrent Jacobian; singular values <1 vanish, >1 explode. LSTM/GRU mitigate via gates but don't eliminate.
- **Sequential bottleneck**: you cannot parallelise across positions during training, so wall-clock is `O(T)` even on a GPU.
- **Long-range dependency**: information from position 1 has to survive 500 multiplications by the time it reaches position 512.

We *do* use a BiLSTM in this thesis — but as a **baseline** to compare the Transformer against, not as the primary model. It achieves Macro-F1 = 0.894 (Table 3).

## 2.5 Transformers in one paragraph

A Transformer replaces recurrence with **all-pairs attention**: for each position, compute a weighted average over all other positions, where the weights come from learned dot products between queries (from this position) and keys (from every other position). No recurrence ⇒ fully parallelisable in time. The receptive field is `O(1)` hops; the cost is `O(T²)` memory in the sequence length. We pay this cost on a 513-position sequence (512 tokens + CLS).

## 2.6 Embeddings & tokenisation

An *embedding* turns a discrete symbol (here, an API token) into a continuous `d_model`-dim vector that the network can manipulate. Learned end-to-end by default. The Word2Vec/GloVe insight is that you can also **pre-train** embeddings from corpus statistics — co-occurrence counts ⇒ a matrix factorised by SVD ⇒ dense vectors where related words are geometrically close. We use exactly this trick on API tokens (§3.7).

## 2.7 Markov chains

A discrete-time Markov chain is a stochastic process where the next state depends only on the current state: `P(s_{t+1} | s_t, s_{t−1}, …) = P(s_{t+1} | s_t)`. The transition probabilities collect into a matrix `T` with `T_ij = P(j | i)`, rows summing to 1.

In this project the Markov assumption is *applied loosely*: we count k-spaced co-occurrences for `k ∈ {1..10}` (not strict 1-step transitions). This is closer to a "directed co-occurrence matrix" than a textbook 1st-order Markov chain — but the spirit (transitions encode behavioural structure) is the same. **Be ready in viva**: an examiner may push on the difference between "k-spaced co-occurrence" and a true Markov chain. The honest answer: it's a co-occurrence matrix with directionality, not a strict Markov model; the term "Markov" is used because k=1 *is* the empirical transition matrix of a 1st-order chain, and `k ∈ {2..10}` extends it to multi-step transitions of the same chain.

## 2.8 Explainable AI

Two camps:

- **Intrinsic** — the model's own internals (attention, gates, prototypes) are read directly as explanation. Cheap. But: Jain & Wallace (2019) argued attention is *not* faithful — you can permute attention and get the same output. Wiegreffe & Pinter (2019) pushed back: attention is *not not* explanation when validated empirically.
- **Post-hoc** — fit a surrogate (linear model, decision tree, SHAP, LIME, integrated gradients) to explain a fixed model. Always available, never guaranteed to match the real model.

**Where we sit**: intrinsic (CLS attention) **plus** an empirical faithfulness check (deletion test, after Samek 2017). We don't *claim* attention is fully faithful; we *measure* the gap.

## 2.9 Android malware analysis

- **Static analysis** — read the APK without running it. Resilient if you have the bytecode; defeated by packing/obfuscation. Datasets: Drebin (2014), CICAndMal2017. Not used here.
- **Dynamic analysis** — run the APK in a sandbox and record what it does. Hooks (Droidmon, idanr1986 2014) intercept Java reflection. Output: JSONL trace of API calls.
- **Reflection** — `java.lang.reflect.Method.invoke(...)`; lets malware call APIs without naming them statically. The *true* target is in `hooked_class` / `hooked_method` of the Droidmon record; raw class/method shows just `Method.invoke`.

## 2.10 Classification machinery

For an 8-class problem:
- **Precision** = TP / (TP + FP). "When I said class c, was I right?"
- **Recall** = TP / (TP + FN). "Of all true class c, how many did I catch?"
- **F1** = harmonic mean of precision and recall = `2PR/(P+R)`.
- **Macro-F1** = unweighted average of per-class F1. Treats Fusob (166 samples) the same as Airpush (5,880). This is the headline metric.
- **Micro-F1** = global F1 over all (pred, true) pairs ≈ accuracy in a balanced setting. Would let Airpush dominate.
- **ROC-AUC** = area under the (true-positive rate, false-positive rate) curve as you sweep the decision threshold. Robust to imbalance; macro-AUROC = average over classes.

## 2.11 Overfitting / underfitting / bias–variance

- **Overfit**: training F1 ≫ test F1. Common signs: tiny patience number, big model, small dataset, late-epoch test divergence.
- **Underfit**: both low. Common signs: linear model on non-linear target, learning rate too low.
- **Bias** = error from too-simple hypothesis class; **Variance** = error from sensitivity to training-set noise. Total expected error = bias² + variance + irreducible noise.

Our model lives slightly on the low-variance side: 2 layers, d_model=128, ~515K parameters, dropout 0.15, early-stop on macro-F1 patience 20. Class weighting offsets the fact that Fusob's 166 samples otherwise contribute almost no gradient signal.

---

# §3. Mathematical Foundations

Every equation below is quoted **verbatim** from the LaTeX with its source line, then unpacked.

## 3.1 Reflection-aware tokenisation (main.tex L730–734)

```latex
token = { REFL:⟨hooked_class⟩.⟨hooked_method⟩    if is_reflection=True
        { ⟨class⟩.⟨method⟩                       otherwise
```

**What it does**: turns each Droidmon record into a single string token. If the call was made via reflection, prepend `REFL:` and use the *hooked* class/method, not the raw `Method.invoke`.

**Why**: preserves semantic target, and the prefix itself is a feature (reflection-heavy families like DroidKungFu cluster on REFL: tokens).

## 3.2 Embedding lookup (main.tex L770–776)

`e_i = E[x_i] ∈ ℝ^{d_model}`, where `E ∈ ℝ^{|V|×d_model}`.

**Symbols**: `x_i` integer token id; `E` learned (or Markov-initialised) embedding matrix; `d_model = 128`.

**Why it works**: tokens are discrete; gradients only flow through continuous quantities. The lookup is a 1-hot × dense product implemented as a row index.

**Project usage**: this is where the Markov initialisation enters. `E ← U_d` (left singular vectors of `T`) before training; frozen for 5 epochs.

## 3.3 [CLS] token & input assembly (main.tex L779–790)

```
c ∈ ℝ^{d_model},  c ~ 𝒩(0, 0.02²)
H⁽⁰⁾ = [c ; e_1 ; … ; e_{L_max}]  ∈ ℝ^{(L_max+1)×d_model}
H⁽⁰⁾ ← H⁽⁰⁾ + P,    P ∈ ℝ^{(L_max+1)×d_model}  (learned positional embedding)
```

**Symbols**: `c` is the learnable summary register; `P` is a learned per-position vector (not sinusoidal). Dropout `p=0.15` is applied after the sum.

**Why CLS at position 0**: gives the classifier a single fixed-position vector to read off; the CLS is *never* masked as a key so every encoder layer's attention can write to it from anywhere.

## 3.4 Pre-norm Transformer block (main.tex L800–802)

```
H' = H^{(ℓ−1)} + MHA(LN(H^{(ℓ−1)}), mask)
H^{(ℓ)} = H' + FFN(LN(H'))
```

**Pre-norm** ⟂ post-norm: LayerNorm is applied to the *input* of the sub-layer, not its output. Empirically gives better gradient flow in small Transformers (the residual stream is never crushed by repeated post-LN scaling). Vaswani et al. 2017 used post-norm; modern small Transformers default to pre-norm.

## 3.5 Scaled dot-product attention (main.tex L807–811)

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k  +  M ) V
```

**Symbols**: `Q ∈ ℝ^{S×d_k}` query, `K ∈ ℝ^{S×d_k}` key, `V ∈ ℝ^{S×d_v}` value, `S = L_max + 1 = 513`, `d_k = d_model/H = 32`, `M` additive mask setting PAD-key positions to `-10⁹`.

**Why √d_k**: variance of `Q·K` grows linearly with `d_k`; the scaling keeps softmax in a non-saturating regime so gradients flow. (Standard derivation: `Var(q·k) = d_k · Var(q_i · k_i)` for unit-variance independent entries.)

**Multi-head**: split `d_model = 128` into `H = 4` heads of `d_k = 32`, run attention in parallel, concatenate, linear-project back to `d_model`. Lets different heads specialise (one head might track reflection, another telephony, etc.).

## 3.6 Feed-forward network (main.tex L817–819)

```
FFN(h) = W_2 · GELU(W_1·h + b_1) + b_2
```

`d_ff = 256` hidden units; dropout `p=0.15` between the two layers. GELU (Gaussian Error Linear Unit) ≈ smoothed ReLU; differentiable everywhere.

## 3.7 Classification head (main.tex L828–830)

```
ŷ = W_2 · GELU(W_1 · h_CLS + b_1) + b_2 ∈ ℝ⁸
```

Hidden = `d_model/2 = 64`. Reads only the final-layer [CLS] hidden state.

## 3.8 CLS attention aggregation (main.tex L841–847)

```
a^{(ℓ,h,b)}_CLS = A^{(ℓ)}_{b,h,0,1:} ∈ ℝ^{L_max}            (row from CLS, skipping CLS↔CLS self-position)
â^{(b)}        = (1 / (N_ℓ · H)) · Σ_ℓ Σ_h a^{(ℓ,h,b)}_CLS   (average over 2 layers × 4 heads = 8 attention rows)
```

`â^{(b)}` is the per-sample importance vector used by the deletion test.

**Defense note**: averaging across layers and heads is a choice. Alternative: attention rollout (Abnar & Zuidema 2020) propagates attention through layers multiplicatively. The thesis acknowledges rollout as future work.

## 3.9 k-spaced transition extraction (main.tex L920–922)

```
For each k ∈ {1,…,10}, for each i ∈ {1,…,n−k}:
    (s_i, s_{i+k}) is a k-spaced rule.
```

Across all training traces we accumulate counts `C_ij = #{ (s_i = i, s_{i+k} = j) at any k }`.

## 3.10 Transition matrix (main.tex L929–931, paper.tex L260–262)

```
T_ij = C_ij / Σ_{j'} C_{ij'}    ∀ i ≠ 0
T_{0,·} = 0                     (PAD row)
```

Row-normalised — each row is a probability distribution over "what comes after API i".

## 3.11 Truncated SVD (main.tex L945–950)

```
T ≈ U_d · Σ_d · V_dᵀ ,    d = d_model = 128
U_d ∈ ℝ^{|V|×128}
```

**SVD intuition**: any matrix factorises as rotation·scale·rotation. Truncating to top-`d` singular values gives the best rank-`d` approximation in Frobenius norm (Eckart–Young). `U_d` rows then sit in a 128-dim space where row similarity ≈ transition-profile similarity.

**Defense note**: this is structurally analogous to GloVe (which factorises a log co-occurrence matrix). The choice of `d = d_model` is for direct copying into the embedding layer; no ablation on `d` was done — flag in §11.

## 3.12 Class-weighted cross-entropy (main.tex L1013–1016)

```
w_c = N / (C · N_c)
L = -Σ_c w_c · y_c · log(ŷ_c)
```

`N = 9337`, `C = 8`, `N_c` = family c's sample count.

Fusob `w_Fusob = 9337 / (8 · 166) ≈ 7.03`; Airpush `w_Airpush = 9337 / (8 · 5880) ≈ 0.198`. Ratio ≈ 35×.

**Why**: without weighting, gradient from 5,880 Airpush samples drowns out the 166 Fusob samples. Without it, macro-F1 collapses on minority families.

## 3.13 Faithfulness gap (main.tex L1043–1060)

```
P_0          = P̂(y | x)                              # baseline confidence
ℐ_attn       = top-m(â)                              # top-m attended positions
P_attn       = P̂(y | x with ℐ_attn replaced by PAD)
P_rand       = P̂(y | x with m random non-PAD indices replaced by PAD)

G = (P_0 − P_attn) − (P_0 − P_rand)
  = Δ_attn − Δ_rand
```

`m ∈ {5, 10, 20}`; averaged over **30 samples per family × 8 families = 240** test samples.

**G > 0** ⇒ masking high-attention tokens hurts confidence more than masking random ones ⇒ attention is at least *aligned* with the decision.

**Subtlety the thesis flags itself** (main.tex L1067–1071): `G > 0` is **necessary** for faithful explanation but **not sufficient**. It rules out "attention is random" but not "attention is correlated with but not causally driving the decision".

## 3.14 Prerequisite formulas — quote-ready

- **Softmax**: `softmax(z)_i = exp(z_i) / Σ_j exp(z_j)`
- **Cross-entropy** (one-hot): `CE = −log p_y`
- **Gradient of softmax CE**: `∂L/∂z_i = p_i − 1{i=y}` (clean derivative; this is *why* people pair them).
- **AdamW update**: `m_t, v_t` running mean/variance of grads → `θ ← θ − η · (m̂_t / (√v̂_t + ε)) − η · λ · θ` (the `−η·λ·θ` is decoupled weight decay; this is what makes it AdamW not Adam).
- **Cosine LR schedule**: `η_t = η_min + ½(η_max − η_min)(1 + cos(π · t / T_max))`. Smoothly anneals to `η_min` at `t = T_max`.

---

# §4. Transformers and Attention — Deep Dive

## 4.1 Why Transformers replaced RNN/LSTM for this kind of task

1. **Parallelism in time**. RNN forward = serial sweep over `T`. Transformer forward = matrix multiply over `T` positions, fully parallel on GPU.
2. **Constant-hop receptive field**. Position 1 reaches position 512 in *one* attention layer, vs ~512 recurrent steps.
3. **Gradient flow**. No deep multiplicative chain in time → vanishing gradients largely solved.
4. **Inductive bias matches the data**. API traces have long-range causal patterns (initial reflection setup → much later sensitive call). All-pairs attention can directly model these.

## 4.2 Q / K / V intuition

Think of attention as a soft, learned lookup:

- **Query** (`Q_i`) = "what is position i asking about?"
- **Key** (`K_j`) = "what does position j advertise?"
- **Value** (`V_j`) = "what does position j contribute if you decide to read it?"

`softmax(Q_i · K_jᵀ / √d_k)` over `j` is the routing distribution; the output at position i is `Σ_j (routing weight) · V_j`. Library analogy: query = your search query, key = the book's title/tags, value = the book's actual content; softmax over (query·key) gives a soft probability distribution over which book you actually open.

## 4.3 Scaled dot-product attention on a toy example

Three positions, `d_k = 2`. Suppose:

```
Q = [[1, 0],     K = [[1, 0],     V = [[10, 0],
     [0, 1],          [0, 1],          [0, 10],
     [1, 1]]          [1, 1]]          [5, 5]]
```

`Q · Kᵀ = [[1,0,1],[0,1,1],[1,1,2]]`; divide by `√2`; softmax row-wise; multiply by `V`. Position 1 ends up reading mostly from row 1 of V plus a smaller amount of row 3, because its query agrees with keys 1 and 3.

## 4.4 Multi-head (this project: H=4, d_k=32)

Split `d_model = 128` into 4 heads × 32 dims. Each head has its own `W_Q, W_K, W_V`. Run attention in parallel per head; concatenate outputs; linear-project back to 128.

**Why**: a single attention map is forced to commit to one set of relationships. Multiple heads can each specialise. In §4.7 of main.tex the qualitative analysis shows different heads attending to different API clusters.

## 4.5 Pre-norm vs post-norm — *call this out in viva*

The original Vaswani (2017) Transformer used **post-norm**: `LN(x + Sublayer(x))`. We use **pre-norm**: `x + Sublayer(LN(x))` (main.tex L795–796).

**Why**: post-norm sits LN *outside* the residual; the residual stream is repeatedly normalised, which crushes gradients in small / shallow Transformers. Pre-norm keeps the residual stream un-normalised, so gradients flow back without compression. This is now the default in modern small Transformers (and in GPT-style models since GPT-2).

Examiners often expect you to mention this distinction.

## 4.6 Learned positional embeddings — *also viva-bait*

We use **learned positional embeddings** (a table of `(L_max+1) × d_model` parameters indexed by position) **not** sinusoidal. Examiners often expect "Attention is All You Need" sinusoidal positional encodings.

**Justification**:
- For a fixed `L_max = 512` the model never sees positions outside this range, so the extrapolation argument for sinusoids doesn't apply.
- Learned embeddings give slightly better fit on bounded-length data.
- Same choice as BERT, GPT-style models in this size range.

## 4.7 Residuals and LayerNorm

- **Residual `x + Sublayer(x)`**: identity shortcut means the *worst case* a sublayer can do is "do nothing" — the gradient still flows. Enables training very deep stacks.
- **LayerNorm**: normalises across the feature dim per position (mean 0, var 1, learnable scale/bias). Stabilises activations against drift; pairs with pre-norm to keep the residual stream well-conditioned.

## 4.8 Your specific instantiation (memorise these)

| Spec               | Value           |
|--------------------|-----------------|
| Total parameters   | ≈515K           |
| `d_model`          | 128             |
| Layers `N_ℓ`       | 2               |
| Heads `H`          | 4               |
| `d_k = d_v`        | 32              |
| `d_ff`             | 256             |
| Dropout            | 0.15            |
| `|V|`              | 1,118           |
| `L_max`            | 512             |
| CLS init           | `𝒩(0, 0.02²)`   |
| Positional emb     | learned         |
| Norm position      | pre-norm        |
| Activation         | GELU            |
| Mask value         | `−10⁹`          |

## 4.9 Markov-initialised embedding — the project's architectural novelty

The embedding matrix `E` is **not** initialised from `𝒩(0, σ²)`. It is initialised from `U_d` — the top-128 left singular vectors of the row-normalised k-spaced transition matrix `T`. Then frozen for 5 epochs (so the attention has to *learn to read* the pre-encoded structure before gradient can perturb it), then unfrozen.

**Hypothesis** (main.tex L909–912): structurally-similar APIs start out near each other → attention heads don't have to waste capacity discovering that, e.g., `TelephonyManager.getDeviceId` and `SmsManager.sendTextMessage` are behaviourally linked → attention can concentrate on a smaller behavioural cluster → **more faithful** explanations.

**The empirical result** (Tables 3 and 5): classification accuracy *drops* by 2.6pp Macro-F1, but the faithfulness gap *rises* by 10.1pp at `m=20`. That is the central tradeoff of the project.

---

# §5. Evaluation Metrics

For each metric: **formula → intuition → tiny worked example → why this project uses it → caveats**.

## 5.1 Accuracy

`Acc = (TP + TN) / N`.

Intuition: fraction of predictions that are right.

Toy: 10 samples, 9 Airpush + 1 Fusob, predict "Airpush" always → Acc = 0.9. Looks great. Macro-F1 = (F1_Airpush + F1_Fusob)/2 = (0.947 + 0)/2 = 0.47. Looks bad.

This is why **accuracy is misleading on imbalanced data**, and why the thesis demotes it to a secondary metric.

## 5.2 Precision, Recall, F1

Per class c:
- `Precision_c = TP_c / (TP_c + FP_c)`
- `Recall_c = TP_c / (TP_c + FN_c)`
- `F1_c = 2 · P_c · R_c / (P_c + R_c)`

**Precision** = "of the things I called c, what fraction really were c?"
**Recall** = "of the things that really were c, what fraction did I catch?"
**F1** = harmonic mean — penalises imbalance between P and R.

Worked example for Fusob in Table 4 of main.tex: TP = 55, FP = 0, FN = 0 → P = R = F1 = 1.0. For SmsPay: TP = 49, FP = 27, FN = 9 → P = 0.6447, R = 0.8448, F1 = 0.7313. Matches the table.

## 5.3 Macro-F1 — the headline metric

`Macro-F1 = (1/C) · Σ_c F1_c`.

Treats every family equally regardless of support. This is why we report it as primary: a model that gets Airpush perfect but misses Fusob entirely is *not acceptable* for malware triage.

**Defense answer to "why macro-F1?"**: because in malware, false negatives on rare families (Fusob, SmsPay) are operationally the most costly — they're the ones an analyst hasn't seen often and the model is supposed to help catch.

## 5.4 Macro vs micro

- **Macro-F1** averages F1 per class. Equal weight per class.
- **Micro-F1** computes one global F1 over the pooled (pred, true) pairs. In multiclass with one prediction per sample, micro-F1 = accuracy. Weighted by support.

For our 35× imbalance, micro-F1 would essentially track Airpush accuracy. Macro-F1 surfaces minority-family failures.

## 5.5 Confusion matrix

`C_{ij}` = # samples truly class i predicted class j. Diagonal = correct. Off-diagonal = errors, and *which* errors. Read by row to find recall failures; by column to find precision failures.

The thesis's Plain Transformer confusion matrix (`results/figures/confusion_plain_transformer.png`, Fig 4.2 in main.tex) shows the dominant off-diagonal is SmsPay ↔ Genpua — both are premium-SMS-fraud families with overlapping telephony APIs.

## 5.6 ROC-AUC / Macro-AUROC

For each class one-vs-rest, sweep the decision threshold; plot TPR vs FPR; area under that curve is AUROC. Macro-AUROC averages across classes.

- AUROC = 1.0 perfect; 0.5 random.
- Robust to class imbalance (doesn't depend on prior).
- Reported as a third number per model.

Our best models hit 0.99+ macro-AUROC, which mainly says "the model can rank-order samples well" — even when it sometimes picks the wrong threshold.

## 5.7 False positives / false negatives in malware context

- **False positive** = benign-or-other-family flagged as family c. Cost: wasted analyst time.
- **False negative** = family c sample missed. Cost: device stays compromised.

For ransomware (Fusob, Jisut) FN is operationally catastrophic; for adware (Airpush) FP costs are tolerable. Macro-F1 indirectly balances both, but a deployment would tune thresholds per family.

## 5.8 Faithfulness metrics — the project's own evaluation contribution

`Δ_attn`, `Δ_rand`, `G = Δ_attn − Δ_rand`. Defined in §3.13. Reported at `m ∈ {5, 10, 20}`. The deletion test is *the* contribution beyond classification: it is what lets you claim something about *explanation quality*, not just prediction quality.

**Limitations (be ready to volunteer these)**:
- `G > 0` is necessary but not sufficient — the thesis itself says so.
- Saturates at `P_0 → 1` (you can't drop a probability that's already at 1). Affects Jisut and the plain model on Fusob.
- 30 samples/family is small; CI not reported.

---

# §6. Implementation Details

> When the two source files disagree, `paper.tex` is treated as truth; the disagreement is flagged in §11.

## 6.1 Dataset — exactly what we used

- **Source**: Droidmon dynamic-analysis traces, originally collected by the authors of D'Angelo et al. 2023; obtained directly from them (paper.tex L310).
- **Filter**: traces with at least 5 API calls retained.
- **Final corpus**: **9,337 samples** across 8 families (counts in §1.4).
- **Raw format**: JSONL, one event per line with fields `timestamp, class, method, hooked_class, hooked_method, is_reflection`.

## 6.2 Preprocessing pipeline (src/preprocessing.py)

1. **JSON sanitisation** — recover from trailing commas; normalise missing fields to `"UNK"`.
2. **Timestamp ordering** — sort events chronologically.
3. **Reflection-aware tokenisation** — apply the rule in §3.1.
4. **Vocabulary** — build on training folds; `f_min = 2`; PAD = 0, UNK = 1; final `|V| = 1,118`.
5. **Sequence truncation** — pad/truncate to `L_max = 512`; **head truncation** (keep first 512 tokens).

## 6.3 Feature engineering = none for the Transformer

The Transformer eats integer token IDs directly. No TF-IDF, no n-grams, no hand-crafted features. Classical baselines (RF, SVM, DT, GaussianNB) *do* use TF-IDF-weighted API-gram features (main.tex L972–974) — that's the only place hand engineering enters.

## 6.4 Markov pre-training stage (src/markov.py)

- Extract all `(s_i, s_{i+k})` for `k ∈ {1..10}` from **training** traces.
- Accumulate into `C ∈ ℝ^{|V|×|V|}`.
- Row-normalise to `T` (zero out PAD row).
- Truncated SVD: `T ≈ U_d Σ_d V_dᵀ`, keep `d = 128`.
- Copy `U_d` into the Transformer's embedding weight.

## 6.5 Architecture — see §4.8 table for the spec.

## 6.6 Training (src/train.py)

| Setting              | Value                                                                                                |
|----------------------|------------------------------------------------------------------------------------------------------|
| Optimiser            | AdamW                                                                                                |
| Learning rate        | `5×10⁻⁴`                                                                                             |
| Weight decay         | `10⁻⁴`                                                                                               |
| LR schedule          | Cosine annealing, `T_max = 100`                                                                      |
| Grad clip            | 1.0 (global norm)                                                                                    |
| Loss                 | Class-weighted CE, `w_c = N/(C·N_c)`                                                                 |
| Batch size           | **64** (per `paper.tex` L299 — the authoritative file; `main.tex` L1020 says 32; see §11.1)          |
| Epochs               | Up to 100                                                                                            |
| Early stopping       | Patience 20 on validation macro-F1                                                                   |
| Embedding freeze     | First 5 epochs (Markov variant only); plain variant trains all params from epoch 0                   |
| CV protocol          | 3-fold stratified, seed 42                                                                           |
| Same folds for all   | Yes (so model comparison is paired)                                                                  |

## 6.7 Hardware / software stack

- **Hardware**: Apple Silicon Mac; PyTorch MPS backend for training. CPU fallback for SHAP gradient passes because MPS has occasional deadlocks at long sequences.
- **Software**: Python 3, PyTorch ≥2.0, scikit-learn ≥1.3, numpy, pandas, matplotlib, seaborn. (See `requirements.txt`.)

## 6.8 Data splitting / validation strategy

- 3-fold stratified CV, seed 42.
- For every model: train on 2 folds, validate on 1, rotate.
- Best epoch chosen by validation macro-F1.
- Results reported as **mean ± std across the 3 folds**.

## 6.9 Inference pipeline

For a new APK:
1. Run in Droidmon sandbox → JSONL trace.
2. `resolve_api()` per event → token string.
3. `APIVocabulary` map → integer IDs.
4. `pad_with_truncation(side='head')` → length 512.
5. Forward pass through `MalwareTransformer` with `return_attention=True`.
6. Read `ŷ.argmax()` for the family, and `â` for per-token importance.
7. Optional: run the deletion test for an audit certificate `G`.

## 6.10 Why each choice, with the alternative

| Choice                            | Alternative                | Why we picked ours                                                                  |
|-----------------------------------|----------------------------|-------------------------------------------------------------------------------------|
| Transformer over LSTM             | LSTM/GRU                   | Parallelism + constant-hop receptive field + we want attention as explanation       |
| `L_max=512` head trunc            | 256, 768, tail             | Direct sweep (Table 2); head ≫ tail at every length                                 |
| 2 layers / 4 heads                | Larger                     | 9.3k samples; deeper Transformers overfit; 515K params already plenty               |
| Pre-norm                          | Post-norm                  | Better gradient flow in small Transformers                                          |
| Learned positional emb            | Sinusoidal                 | Fixed-bounded length; learned ≈ slightly better; same choice as BERT                |
| Markov init                       | Random init                | Encodes prior structure; we measure cost (2.6pp F1) and benefit (10.1pp G@m=20)     |
| Truncated SVD d=128               | `d=64, d=256, PMI/PPMI`    | Match `d_model` for direct copy; PPMI listed as future work                         |
| k=1..10 spacing                   | k=1 only, k=20             | "Captures local + medium-range" (paper.tex L263); not ablated — flag as weak point  |
| Embedding freeze 5 epochs         | 0 or 10                    | Lets attention adapt to fixed embeddings first; not ablated — flag as weak point    |
| Class-weighted CE                 | Focal loss, oversampling   | Simplest fix for 35× imbalance; matches sklearn convention                          |
| AdamW + cosine                    | SGD, plateau LR            | Standard for Transformers; cosine keeps useful LR through epoch 80                  |
| 3-fold (not 5/10)                 | 5-fold, 10-fold            | Compute budget; 3 folds reuse fold splits across all 7 models for fair comparison   |
| Deletion test                     | LIME, SHAP, attn rollout   | Tests *the model's own attention*, no surrogate                                     |

## 6.11 Computational complexity

- Attention is `O(S²)` in sequence length per layer, `S = 513`. Two layers, four heads → ≈ 2 × 4 × 513² ≈ 2.1M attention operations per sample, trivially fast.
- Markov pre-training: `O(N · L · k_max)` to count co-occurrences (`N` traces, `L` avg length, `k_max=10`); SVD on a 1118×1118 matrix is ≈ ms on CPU.

## 6.12 Scalability / deployment considerations

- Current size (~515K params) fits comfortably on-device for an Android security app.
- Sequence length cap is the binding constraint for very long traces; head truncation is conservative but blind to late-stage behaviour.
- Inference is single forward pass + optional deletion test (which is ~3 extra forward passes per `m` value, per sample).

## 6.13 Debugging notes & known engineering hazards

- **MPS deadlocks** at `seq_len ≥ 768` on Apple Silicon — that's why `L_max = 768` configuration is excluded for the tail variant in the sequence-length sweep.
- **Padding mask in SHAP wrapper** — earlier bug: if positional embeddings are added *before* the SHAP gradient wrapper, PAD positions stop being zero, and the attention mask breaks. Fix: wrapper returns token-only embeddings; positional addition happens inside `forward()`.
- **Vocab-on-full-corpus vs per-fold** — the thesis contradicts itself on this (§11.2).

---

# §7. Results and Analysis

> Every number quoted below is verbatim from the LaTeX. Round only when explicitly written rounded.

## 7.1 Sequence-length / truncation sweep (main.tex Table 2)

Subset: length-≥30, 8,085 samples, 3-fold CV. 5 configs run (768/tail dropped for hardware-memory reasons).

| `L_max` | Trunc | Macro-F1                  | Accuracy                 | AUROC              |
|---------|-------|---------------------------|--------------------------|--------------------|
| **512** | **head** | **0.8875 ± 0.0123**   | **0.9445 ± 0.0053**      | 0.9834 ± 0.0016    |
| 768     | head  | 0.8867 ± 0.0033           | 0.9459 ± 0.0059          | 0.9843 ± 0.0016    |
| 256     | head  | 0.8811 ± 0.0066           | 0.9410 ± 0.0031          | 0.9865 ± 0.0012    |
| 512     | tail  | 0.8697 ± 0.0126           | 0.9352 ± 0.0031          | 0.9805 ± 0.0034    |
| 256     | tail  | 0.8587 ± 0.0026           | 0.9307 ± 0.0065          | 0.9810 ± 0.0024    |

**Headline**: 512/head wins (or ties 768/head within noise). Head beats tail by 1.7–2.9pp at every length. Justifies the canonical choice.

## 7.2 Overall classification benchmark (main.tex Table 3, paper.tex Table 2 — agree)

3-fold stratified CV, 9,337 samples, mean ± std.

| Model              | Accuracy           | Macro-F1           | AUROC              |
|--------------------|--------------------|--------------------|--------------------|
| GaussianNB         | 0.860 ± 0.001      | 0.782 ± 0.007      | 0.925 ± 0.007      |
| Decision Tree      | 0.922 ± 0.001      | 0.836 ± 0.004      | 0.910 ± 0.002      |
| LinearSVM          | 0.923 ± 0.001      | 0.844 ± 0.009      | 0.980 ± 0.005      |
| Markov Transformer | 0.927 ± 0.004      | 0.858 ± 0.004      | 0.976 ± 0.003      |
| Plain Transformer  | 0.940 ± 0.003      | 0.884 ± 0.009      | 0.981 ± 0.003      |
| BiLSTM             | 0.948 ± 0.004      | **0.894 ± 0.003**  | **0.992 ± 0.001**  |
| Random Forest      | **0.950 ± 0.003**  | 0.893 ± 0.010      | **0.993 ± 0.001**  |

**Honest reading**:
- BiLSTM and RF are the *best classifiers*. We are not SOTA on raw F1.
- Plain Transformer (0.884) is within ≈1pp of BiLSTM (0.894); within noise on AUROC.
- Markov Transformer pays a **2.6pp Macro-F1 cost** vs the plain Transformer.

That 2.6pp is **not** the headline number to apologise for — it is the price of the structural prior, paid in exchange for faithfulness (§7.4). The defense framing in paper.tex L375–380 is explicit: *"the Markov variant is not a better classifier, but we show below that its attention tells a more faithful story."*

## 7.3 Per-family performance, Plain Transformer best fold (Table 4)

| Family       | Precision | Recall | F1     | Support |
|--------------|----------:|-------:|-------:|--------:|
| Airpush      | 0.9675    | 0.9730 | 0.9702 | 1,960   |
| DroidKungFu  | 0.9069    | 0.8831 | 0.8948 | 419     |
| **Fusob**    | 1.0000    | 1.0000 | 1.0000 | 55      |
| Genpua       | 0.8261    | 0.7308 | 0.7755 | 104     |
| GinMaster    | 0.9085    | 0.8563 | 0.8817 | 174     |
| Jisut        | 0.9342    | 0.9861 | 0.9595 | 144     |
| Opfake       | 0.9948    | 0.9747 | 0.9847 | 198     |
| SmsPay       | 0.6447    | 0.8448 | 0.7313 | 58      |

**Reading**:
- Fusob/Opfake/Jisut/Airpush are easy — distinctive behavioural signatures.
- DroidKungFu/GinMaster are mid-tier — both use reflection-heavy root exploits, some mutual confusion.
- **SmsPay and Genpua are the hard pair** — both are premium-SMS-fraud with shared telephony APIs. Confusion matrix shows the dominant off-diagonal is SmsPay ↔ Genpua.

## 7.4 Deletion-test results (Table 5)

240 samples (30/family); `Δ` in percentage-points of `P̂(y|x)`.

|         | metric         | m=5      | m=10     | m=20            |
|---------|----------------|----------|----------|-----------------|
| Plain   | Δ_attn         | 9.5%     | 13.9%    | 18.7%           |
|         | Δ_rand         | 0.5%     | 0.6%     | 1.7%            |
|         | **G**          | 9.0%     | 13.3%    | 17.0%           |
| Markov  | Δ_attn         | 9.5%     | 18.3%    | 27.2%           |
|         | Δ_rand         | 0.01%    | 0.9%     | 0.1%            |
|         | **G**          | **9.5%** | **17.4%**| **27.1%**       |

**Key findings (memorise this)**:
- Both models pass the basic faithfulness criterion `G > 0` at every `m`.
- At `m=5`, the models tie within noise.
- At `m=10` the Markov gap is 4.1pp higher; at `m=20`, **10.1pp higher**.
- Markov's `Δ_rand` stays near zero ⇒ random tokens carry almost no decision weight ⇒ attention has concentrated decision-relevant signal in a small token subset.

This is the **central empirical claim of the project**: Markov initialisation buys faithfulness at the cost of accuracy.

## 7.5 Per-family faithfulness at m=20 (Table 6)

| Family       | Plain G | Markov G | ΔG          | Direction |
|--------------|--------:|---------:|-------------|-----------|
| Airpush      |  0.4%   | 16.0%    | +15.6 pp    | ↑         |
| DroidKungFu  | 26.8%   | 55.2%    | +28.4 pp    | ↑         |
| Fusob        |  0.0%   | 22.2%    | +22.2 pp    | ↑         |
| Genpua       | 19.7%   | 30.7%    | +11.0 pp    | ↑         |
| **GinMaster**| 51.9%   | 12.0%    | **−39.9 pp**| **↓**     |
| Jisut        |  0.1%   |  0.0%    |  −0.1 pp    | ≈         |
| Opfake       | 21.1%   | 24.3%    | +3.2 pp     | ↑         |
| SmsPay       | 15.6%   | 56.8%    | **+41.2 pp**| ↑         |

**6 of 8 families improve**. The two exceptions are diagnostically interesting, not embarrassing:

- **SmsPay (+41.2pp)** — the biggest win. SmsPay's signature is a stereotyped three-step chain `registerTelephonyManager → composeMessage → sendSms`. K-spaced transitions in `T` capture this exact chain; SVD places these APIs close in embedding space; attention picks them up early in training. Masking them collapses confidence by 56pp.
- **DroidKungFu (+28.4pp)** — multi-hop root-exploit-then-exfiltrate chain captured well by k=1..10 transitions.
- **Fusob (+22.2pp)** — plain model "classifies by exclusion" (P₀>0.999 regardless of tokens), so its deletion test floors out. Markov model retains sensitivity to the lock-screen/SMS-demand APIs.
- **GinMaster (−39.9pp)** — the exception worth its own paragraph. GinMaster is **parasitic malware**: it embeds in host apps, so traces are dominated by host-app behaviour. The global transition matrix `T`, built over *all* training samples, has high weights for host-app patterns (because they dominate the corpus). Markov-init thus assigns high salience to non-discriminative tokens for GinMaster specifically. The plain Transformer, unconstrained by any prior, finds the small set of privilege-escalation APIs that actually discriminate GinMaster, and its faithfulness gap is correspondingly large. **This is the cleanest illustration of the double-edged nature of structural priors**: when global structure aligns with discriminative signal, it sharpens attention; when it doesn't, it actively misleads.
- **Jisut (~0)** — both models hit P₀ > 0.999, so the deletion test floors. *This is a measurement-ceiling effect, not evidence of unfaithfulness*.

## 7.6 Discussion — accuracy / explainability tradeoff

Three deployment scenarios (main.tex L1487–1503) clarify when each model wins:

| Scenario             | Recommended model | Reasoning                                                                            |
|----------------------|-------------------|--------------------------------------------------------------------------------------|
| Forensic analysis    | Markov Transformer| Analyst is auditing decisions; explanation quality > 2.6pp F1                        |
| Automated triage     | BiLSTM or RF      | Top accuracy is what matters; explanation is optional                                |
| On-device IoT classifier | Plain Transformer | Best balance of accuracy + footprint + intrinsic attribution                     |

## 7.7 Limitations the thesis itself names (main.tex §4.9)

1. **Single dataset** — generalisation to Drebin / CICAndMal2017 / unseen families untested.
2. **Vocabulary leakage** — vocab built on full corpus before fold split (soft leakage; see §11.2).
3. **Small minority classes** — Fusob/SmsPay have ≤173 samples; per-class metrics noisy.
4. **`G > 0` necessary, not sufficient** — see §3.13 and §11.5.
5. **Global transition matrix** — per-class matrices listed as future work (would address GinMaster).
6. **Sandbox evasion** — malware that detects the sandbox produces atypical traces; not modelled.

---

# §8. Viva / Defense Q&A

For each question: **what the examiner is testing → strong model answer → common mistakes → likely follow-up**.

## 8.1 Basic conceptual

**Q1. What is attention?**
*Testing*: do you understand the mechanism, not just the marketing.
*Answer*: A learned, soft, content-based lookup. For each query position, compute a similarity (scaled dot product) with every key position, softmax to get a probability distribution, take a weighted sum of values. The output at position i is a re-weighted aggregation of the whole sequence, where the weights depend on what position i is "asking about".
*Common mistakes*: saying "attention is interpretable by default" — Jain 2019 says no.
*Follow-up*: "Why divide by √d_k?" → variance control on softmax inputs.

**Q2. What is a Markov chain?**
*Testing*: do you know what you actually used.
*Answer*: A stochastic process where the next state depends only on the current state. Characterised by a transition matrix `T_ij = P(j|i)`, rows summing to 1. We use a *generalisation*: we count k-spaced co-occurrences for k=1..10, which gives a directed co-occurrence matrix that strictly contains the 1st-order Markov transition matrix (k=1 row) as a special case.
*Common mistakes*: claiming we use a strict Markov chain.
*Follow-up*: "Is that really a Markov chain then?" → no, it's a directed co-occurrence matrix; we call it Markov-initialised because the spirit is the same.

**Q3. What is the role of `[CLS]`?**
*Answer*: Learnable summary register. Prepended at position 0; never masked as a key; final-layer hidden state at position 0 is the only thing the classifier reads. Same role as BERT's `[CLS]`.

**Q4. Why a Transformer here and not a CNN?**
*Answer*: 1D CNNs would impose a *local* receptive field; malware traces have long-range causal patterns (reflection setup at position 5, sensitive call at position 400). Attention has constant-hop receptive field.

## 8.2 Project-specific

**Q5. Why factorise T with SVD? Why not just use T directly as embedding?**
*Testing*: do you understand dimensionality and rank.
*Answer*: T is `|V|×|V|` = 1118×1118. Embedding needs to be `|V|×d_model = 1118×128`. SVD gives the optimal (in Frobenius norm) rank-128 approximation of T, and `U_d` gives a dense 128-dim representation per token. Using T directly would force `d_model = |V|`, which is absurd for the model size.
*Follow-up*: "Why not PPMI?" → listed as future work; PMI/PPMI typically gives slightly better semantic embeddings (GloVe-style), but SVD on raw T is the simplest direct analogue and matches D'Angelo's k-spaced framework.

**Q6. Why d = 128 specifically?**
*Answer*: To match `d_model = 128` so we can copy `U_d` directly into the embedding layer. Not ablated — flag as future work.

**Q7. Why freeze for 5 epochs?**
*Answer*: To force the attention heads to *learn to read* the Markov-encoded structure before gradient signal can warp the embeddings. If we leave embeddings trainable from epoch 0, the gradient might immediately push them away from the Markov geometry, defeating the prior. 5 was chosen as a small constant; not ablated.
*Common mistakes*: claiming 5 was selected by validation.

**Q8. Why head truncation?**
*Answer*: Direct empirical sweep, Table 2: head beats tail by 1.7–2.9pp at all lengths. Mechanistically, the first 512 API calls of a malware trace contain SDK initialisation, permission requests, reflection setup, and initial network probes — these are family-discriminative. Tail-end calls tend to be more generic system calls shared across families.
*Follow-up*: "Would this generalise to corpora where malicious behaviour is *late* in the trace?" → No, this is a corpus-specific finding; flag as a limitation.

**Q9. Why class-weighted CE not focal loss?**
*Answer*: Simpler; one hyperparameter (the class weights, fixed by frequency); standard sklearn convention. Focal loss adds a second hyperparameter (γ) and didn't improve in pilot. Not centrally ablated.

**Q10. Why average attention across heads and layers?**
*Answer*: Gives a single interpretable scalar per token for the deletion test. Alternative: attention rollout (Abnar & Zuidema 2020), which propagates attention through layers multiplicatively. Listed as future work. Per-head analysis is shown qualitatively in §4.7 of main.tex but not used for the deletion test.

## 8.3 Mathematical

**Q11. Derive the gradient of softmax cross-entropy.**
*Answer*: Let `p = softmax(z)`, target one-hot y. `L = −Σ_i y_i log p_i = −log p_y`. Then `∂L/∂z_i = p_i − y_i`. *That clean form is why softmax+CE are paired*: the gradient is simply (predicted minus target), no awkward Jacobian.

**Q12. Why √d_k scaling?**
*Answer*: If q, k have unit-variance independent components in d_k dims, `Var(q·k) = d_k`. Without scaling, large d_k makes the softmax inputs huge → softmax saturates to one-hot → gradient vanishes. Dividing by √d_k restores `Var ≈ 1` and keeps softmax in a useful regime.

**Q13. Why SVD gives the best rank-d approximation?**
*Answer*: Eckart–Young theorem. Any matrix M with SVD `UΣVᵀ` has best rank-d approximation in Frobenius norm = `U_dΣ_dV_dᵀ` (keep top-d singular values, zero the rest). The approximation error is `Σ_{i>d} σ_i²`.

**Q14. Derive AdamW's update.**
*Answer*: Adam maintains `m_t = β_1 m_{t-1} + (1-β_1) g_t`, `v_t = β_2 v_{t-1} + (1-β_2) g_t²`, bias-corrected `m̂_t, v̂_t`, then `θ_t = θ_{t-1} − η m̂_t / (√v̂_t + ε)`. AdamW *decouples weight decay*: `θ_t = θ_{t-1} − η m̂_t/(√v̂_t + ε) − η λ θ_{t-1}`. The decoupling matters because L2 regularisation interacts badly with Adam's per-parameter LR scaling.

## 8.4 Architecture

**Q15. Pre-norm vs post-norm.**
See §4.5. Examiners frequently test this.

**Q16. Why learned positional embeddings, not sinusoidal?**
See §4.6. Fixed `L_max`, no extrapolation needed.

**Q17. Why only 2 layers?**
*Answer*: 9,337 samples is small. Deeper Transformers overfit. 2 layers / 4 heads / d_model=128 already give 515K parameters, ~55× the training-set size. Architecture-capacity check (3 layers) is listed in our internal notes as "optional, may not justify cost".

## 8.5 Evaluation

**Q18. Why macro-F1 not accuracy?**
*Answer*: 35× class imbalance. Accuracy with always-predict-Airpush = 63%. Macro-F1 forces minority families to count equally. Forensic priorities (catching rare families) demand it.

**Q19. What does G > 0 actually prove?**
*Answer*: That the attention-ranked top-m tokens are more decision-relevant than random top-m tokens. It rules out "attention is random noise" and "attention is uniformly distributed". It does **not** prove the attention is causally responsible for the decision (correlation vs causation), nor that it identifies *all* relevant tokens (completeness). The thesis says this explicitly (main.tex L1067–1071).

**Q20. Why 30 samples per family in the deletion test?**
*Answer*: Compute budget — each deletion test requires 3 extra forward passes per sample per `m` value × 3 values = 9 forward passes. 240 samples × 9 = 2,160 forward passes per model. We didn't compute CIs; this is a limitation. For SmsPay (only 173 total samples), 30 per fold is nearly the whole test partition.

## 8.6 Tradeoffs — "why this not that"

**Q21. Why not a GNN on the API call graph?**
*Answer*: Two reasons. (i) Loses temporal order — a GNN over a graph aggregated from the trace is permutation-invariant and discards the sequence structure that distinguishes SmsPay's `register → compose → send`. (ii) Higher model and inference complexity; we wanted something deployable. We do *use* graph statistics (the transition matrix) — just to initialise embeddings, not as the inference model.

**Q22. Why not SHAP or LIME for explanation?**
*Answer*: They explain *surrogates*, not the model itself. Our claim is about the model's own attention, validated by a model-native deletion test. SHAP/LIME would also need surrogate fitting per sample (slow). We do run SHAP for RF baseline analysis (results/figures/shap_rf_*) but not as the primary explanation channel.

**Q23. Why not pretrain a BERT-style model on API traces first?**
*Answer*: 9,337 samples is far too small for self-supervised pretraining. Markov initialisation is a *cheap* alternative pretrain step: count, factorise, copy. No labels needed, runs in seconds, encodes the same kind of distributional structure pretraining would.

**Q24. Why not just use Random Forest? It wins on accuracy.**
*Answer*: Two reasons. (i) RF on TF-IDF features destroys order — `register, compose, send` and `send, compose, register` look identical. (ii) RF's feature importance is global; it can't give a *per-sample* attribution over the input positions. The Transformer can.

## 8.7 Limitations and future work

**Q25. Biggest limitation?**
*Answer*: Single corpus. We've only shown this on UMD via D'Angelo et al. 2023. Drebin or CICAndMal2017 are static-feature datasets we can't directly evaluate. Cross-corpus generalisation is the most important next step.

**Q26. Future work?**
*Answer*: Per-class transition matrices during training (would address the GinMaster failure mode); PPMI weighting before SVD; attention rollout instead of average; causal-counterfactual testing for sufficiency; larger corpora; federated training.

## 8.8 Real-world deployment

**Q27. How would you ship this?**
*Answer*: On-device ML kit model (515K params; ~2MB quantised). Input pipeline = sandboxed dynamic analysis or Frida-style hooking. Per detection: forward pass + top-m attention output. For audit/forensic mode, run deletion test → emit `G` per sample as a confidence-of-explanation flag.

**Q28. Cold start on a new family?**
*Answer*: Current model is 8-way; new families need at least a few-shot retrain. Markov initialisation helps: even with few samples of family 9, k-spaced rules from those samples enter the global `T` and shift the embedding geometry.

**Q29. How would adversarial malware evade you?**
*Answer*: Two attack surfaces. (i) Sandbox detection → atypical traces. (ii) Inject decoy API calls to dilute attention. Defence: integrate with non-trace-based features; ensemble with static analysis; monitor for trace anomalies.

## 8.9 Professor-style cross-questions

The hardest questions you'll actually face. These map directly to the weak-points list in §11.

**Q30. "Your thesis says vocabulary is built on training folds only on line 745, then in the limitations section on line 1526 it says it's built on the full corpus. Which is right?"**
*Answer*: The Limitations section is correct. The vocabulary was built on the full corpus before fold splitting. This is a soft leakage of *token existence* (which APIs appear at all) — not statistics, not labels. It affects which test tokens get mapped to UNK. We flag this explicitly as a limitation. The earlier wording on line 745 is from a draft that wasn't updated. Honest framing wins here.

**Q31. "Your batch size is 32 in the thesis and 64 in the paper. Which?"**
*Answer*: The paper number, 64, is the final configuration. The 32 in the thesis is an early draft value. Both were tried during development; final runs used 64.

**Q32. "D'Angelo et al. get ~99% on the same corpus. You get 95% with Random Forest. Why?"**
*Answer*: Three independent differences. (i) **Corpus size**: D'Angelo report ~3,500 samples; we use 9,337 — larger, harder, more class overlap. (ii) **Evaluation protocol**: they report on a single held-out test split; we use 3-fold CV with mean ± std, which is a stricter, lower-mean estimator. (iii) **Feature representation**: their best result uses k-spaced rules as classifier *features*; we use them as Transformer *embedding initialisation*. The fairest single-number comparison is our Markov-Transformer (0.927 accuracy) at the same family granularity — about 6pp below their best.

**Q33. "G > 0 is necessary but not sufficient. So what *would* be sufficient?"**
*Answer*: Causal counterfactual testing — show that for each top-attended token, if you replace it with a behaviourally distinct token while holding all others constant, the prediction changes in the predicted direction. We don't do this; it's listed as future work. A second route: completeness testing — show that masking *any* non-attended token doesn't change the prediction. Combined necessity + sufficiency would constitute a stronger faithfulness claim.

**Q34. "Did you ablate k=1..10?"**
*Answer*: No. We chose k=1..10 by analogy to D'Angelo et al. who used the same range. Sensitivity analysis is an explicit gap in our ablation study and listed as future work. The motivation (line 936 of main.tex) is qualitative: malware behavioural patterns span multiple intermediate generic calls, so multi-hop transitions matter.

**Q35. "Did you ablate SVD dim = 128?"**
*Answer*: No. Chosen to match `d_model`. Same gap; same answer.

**Q36. "Did you ablate the 5-epoch freeze?"**
*Answer*: No. Chosen as a small constant. Same gap; same answer. *Strategy in viva*: be honest about the three missing ablations. The defender's framing: "our contribution is the *concept* (Markov-init for attention faithfulness) and the *measurement methodology* (deletion test). Hyperparameter sensitivity is a separate study and a clean future-work item."

**Q37. "GinMaster fails. Why call it 'parasitic' instead of admitting the method fails on hard cases?"**
*Answer*: It's both — and the thesis says so. "Parasitic" is the diagnostic mechanism: GinMaster embeds in host apps, so its trace is dominated by host-app behaviour; the global transition matrix encodes the host-app pattern as high-weight; Markov-init thus highlights *non*-discriminative tokens. The framing isn't "we failed; here's an excuse" but "we found a clean boundary condition: when global structural prior aligns with discriminative signal, it sharpens attention; when it doesn't, it misleads." Future work directly addresses this via per-class transition matrices.

**Q38. "Your Jisut faithfulness gap is 0%. So your model's attention is meaningless for Jisut?"**
*Answer*: No — it's a measurement ceiling. Jisut samples are classified with `P₀ > 0.999` by both models. The deletion test measures `(P₀ − P_masked)`; if `P₀` is already at the softmax ceiling, the gap *cannot* be large no matter how important the masked tokens are. This is a *floor effect on the metric*, not evidence of unfaithfulness. Mitigations would be: temperature scaling, larger `m`, or logit-space deletion tests instead of probability-space.

**Q39. "Why not just train longer? Maybe 200 epochs would close the gap with BiLSTM."**
*Answer*: Cosine annealing with `T_max=100` makes the effective LR negligible past epoch 80. Early stopping (patience 20) caps wasted compute. Best folds reached their best macro-F1 at epochs 8–36; extending the budget wouldn't help.

**Q40. "What's the single most important contribution of this thesis?"**
*Answer*: The *measurement* — applying a deletion test to a Markov-initialised Transformer and showing a clean accuracy-vs-faithfulness tradeoff (2.6pp F1 cost for 10.1pp G gain at m=20). The Markov initialisation itself is a small architectural tweak; the contribution is using a controlled comparison to *quantify* the cost and benefit of structural priors on explanation quality.

---

# §9. Slide / Presentation Prep

## 9.1 The 2-minute pitch

> "Android malware family classification is a forensic problem: AV-TEST sees 450k new samples a month, and after a sample trips the sandbox, the analyst needs to know *which* family it is and *why* the model thinks so. Existing classifiers give the *what* but not the *why* — they're black boxes. We trained a small Transformer (515K params, 2 layers) on the Droidmon API traces from D'Angelo's UMD corpus, 9,337 samples across 8 families. The novelty is the embedding layer: instead of random initialisation, we factorise a k-spaced API-transition matrix with SVD and copy the top-128 left singular vectors into the embedding weights, then freeze for 5 epochs. This pre-bakes behavioural structure into the embedding geometry. We then use the [CLS] attention as a per-sample explanation, and we *measure* whether it's faithful by deleting the top-attended tokens and seeing whether confidence collapses. Markov initialisation costs 2.6pp Macro-F1 versus the plain Transformer but buys 10.1pp more faithfulness at m=20 tokens deleted — that's the central tradeoff."

## 9.2 The 10-minute presentation flow

| Slide | Topic                                      | What to emphasise                                            | Likely interruption                          |
|-------|--------------------------------------------|--------------------------------------------------------------|----------------------------------------------|
| 1     | Title + author                             | Move fast                                                    | —                                            |
| 2     | Problem & motivation                       | 450k samples/month; family → response decision               | "Why family, not just malicious/benign?"    |
| 3     | Dataset (the 8-family table)               | 35× imbalance; that's why macro-F1                           | "How was it collected?"                      |
| 4     | Pipeline diagram                           | Walk slowly: trace → tokens → vocab → embedding → Transformer| —                                            |
| 5     | Tokenisation (reflection-aware)            | The REFL: trick                                              | "Why not just use the raw class?"            |
| 6     | Transformer architecture diagram           | 2 layers, 4 heads, pre-norm, learned positional, CLS         | "Why pre-norm?" "Why learned positional?"    |
| 7     | Markov pipeline (T → SVD → U_d → freeze 5) | This is the contribution; show the matrix → SVD → embed flow| **VERY LIKELY:** "Did you ablate k? d? 5?"   |
| 8     | Training protocol & metrics                | AdamW, cosine, class-weighted CE, 3-fold seed 42             | "Batch size?" (answer: 64 per paper)         |
| 9     | Results — main benchmark table             | Acknowledge BiLSTM/RF lead; frame Markov honestly            | "You're behind BiLSTM. Why bother?"          |
| 10    | Sequence-length sweep                      | Justifies 512/head                                           | —                                            |
| 11    | Per-family breakdown                       | SmsPay/Genpua confusion as honest weakness                   | —                                            |
| 12    | Deletion test definition + results         | Show the formula AND the 10.1pp number                       | **VERY LIKELY:** "Is G > 0 enough?"          |
| 13    | Per-family faithfulness                    | SmsPay win, GinMaster failure as boundary condition          | "Why is GinMaster the exception?"            |
| 14    | Qualitative attention heatmaps             | Markov attention is sharper for stereotyped families         | "How did you pick the visualised samples?"   |
| 15    | Limitations & future work                  | Be the first to name vocabulary leakage, no ablations        | —                                            |
| 16    | Conclusions                                | Tradeoff is the contribution; *measurement* is the point     | —                                            |

## 9.3 Things to emphasise

- The contribution is the **measurement methodology**, not raw accuracy.
- Markov-init **costs accuracy**; you knew this and report it honestly.
- The GinMaster failure is **diagnostically interesting**, not a hidden bug.
- Saturation (Jisut, Fusob/plain) is a **metric limitation**, not a model failure.

## 9.4 Things to avoid

- Don't claim SOTA. You're behind BiLSTM/RF and they'll know.
- Don't claim attention is *fully* faithful. Necessary, not sufficient — say it before they do.
- Don't oversell Markov as a *Markov chain*. It's a directed co-occurrence matrix; the *k=1* slice is a Markov transition matrix.
- Don't generalise to other corpora. You only tested one.

## 9.5 If interrupted mid-slide

Strategy: **acknowledge → answer briefly → offer to expand at the end**.
> "Good question — short answer is [one sentence]. I have a slide on that coming up / I can expand at the end if you'd like more detail."

This buys you back the thread without sounding evasive.

## 9.6 Diagram explanations

- **Architecture diagram**: walk top to bottom. "Input is integer token sequence → embedding layer (Markov-initialised or random) → CLS prepended → positional addition → 2 pre-norm Transformer blocks → final LN → MLP classifier."
- **SVD pipeline figure**: "Count k-spaced co-occurrences → row-normalise → factorise; the rows of U_d become the embedding."
- **Per-family faithfulness bar chart**: "Each bar is `Markov G − Plain G`. Six bars positive, one approximately zero (Jisut, saturation), one strongly negative (GinMaster, parasitic). The signed bar pattern *is* the central empirical claim."

## 9.7 Common defense traps

- **Over-claiming explainability**: don't say "we explain the model"; say "we measure partial faithfulness".
- **Hiding the accuracy gap**: Acknowledge it on slide 9 before they raise it on slide 12.
- **Cross-corpus claims**: avoid "this would also work on Drebin" — you didn't test it.
- **Hyperparameter ablations**: don't pretend k=1..10 was chosen scientifically. Say "by analogy to prior work".

---

# §10. Quick Revision

## 10.1 One-page architecture cheat sheet

```
Dataset:  9337 samples, 8 families, 35× imbalance (Airpush 5880 vs Fusob 166)
Source:   Droidmon traces from D'Angelo et al. 2023 (UMD corpus)
Vocab:    |V|=1118  (f_min=2; PAD=0, UNK=1)
Seq:      L_max=512, head truncation

Architecture (≈515K params):
  d_model=128, N_layers=2, H=4, d_k=32, d_ff=256, dropout=0.15
  Pre-norm, learned positional embeddings, GELU, learnable [CLS]
  Mask value = −10⁹

Markov init:
  k ∈ {1..10}    →    C  →  row-normalise → T  →  truncated SVD →  E ← U_d
  Freeze E for 5 epochs, then unfreeze

Training:
  AdamW lr=5e-4 wd=1e-4
  Cosine LR T_max=100
  Grad clip 1.0
  Class-weighted CE: w_c = N/(C·N_c)
  Batch=64 (paper.tex; main.tex says 32 — paper wins)
  Early stop: patience 20 on macro-F1
  3-fold stratified CV, seed 42

Results headline:
  Plain Transformer: 0.940 acc / 0.884 macro-F1 / 0.981 AUROC
  Markov Transformer: 0.927 / 0.858 / 0.976  (−2.6pp F1)
  BiLSTM: 0.948 / 0.894 / 0.992   (best F1)
  RF: 0.950 / 0.893 / 0.993    (best accuracy)

Deletion test (G = Δ_attn − Δ_rand, m=20):
  Plain G = 17.0%
  Markov G = 27.1%   (+10.1pp)

Per-family m=20 G winners:
  Markov ↑ in 6/8 (SmsPay +41.2, DroidKungFu +28.4, Fusob +22.2, Airpush +15.6, Genpua +11.0, Opfake +3.2)
  Markov ↓ GinMaster (−39.9, parasitic — global T highlights wrong APIs)
  Saturated:  Jisut (P₀>0.999)
```

## 10.2 Formula summary card

```
Attention:     softmax(QKᵀ/√d_k + M) · V
Pre-norm:      H' = H + MHA(LN(H));  H_next = H' + FFN(LN(H'))
FFN:           W₂·GELU(W₁·h + b₁) + b₂,    d_ff=256
Loss:          w_c = N/(C·N_c);   CE = −Σ w_c y_c log ŷ_c
T:             T_ij = C_ij / Σ_j' C_ij'    (PAD row zero)
SVD:           T ≈ U_d Σ_d V_dᵀ,    d=128
CLS attn:      â^b = (1/(N_ℓ·H)) Σ_ℓ Σ_h A^ℓ_{b,h,0,1:}
Faithfulness:  G = (P₀ − P_attn) − (P₀ − P_rand) = Δ_attn − Δ_rand
Softmax grad:  ∂L/∂z_i = p_i − y_i
AdamW:         θ ← θ − η m̂/(√v̂+ε) − η λ θ
```

## 10.3 Ten facts you must not forget

1. **9,337 samples, 8 families, 35× imbalance.**
2. **Macro-F1 is the headline metric. Plain Transformer = 0.884; Markov = 0.858; BiLSTM = 0.894; RF = 0.893.**
3. **The contribution is the deletion-test tradeoff: −2.6pp F1 for +10.1pp G at m=20.**
4. **G > 0 is necessary but NOT sufficient for faithfulness.**
5. **Pre-norm, learned positional embeddings — both non-default choices, both deliberate.**
6. **k=1..10 was not ablated, freeze=5 was not ablated, d=128 was not ablated.**
7. **GinMaster fails because global T encodes host-app patterns (parasitic malware).**
8. **Jisut G ≈ 0 is a saturation floor, not model failure.**
9. **Batch size: 64 (paper) — main.tex's 32 is a stale draft number.**
10. **Vocabulary leakage exists (built on full corpus, not per-fold) — owned in limitations.**

## 10.4 Glossary

- **REFL:** prefix on tokens for reflected calls.
- **Droidmon**: dynamic-analysis hook framework.
- **k-spaced rule**: ordered token pair at distance k ∈ {1..10}.
- **Transition matrix T**: row-normalised count matrix.
- **U_d**: top-d left singular vectors of T.
- **Embedding freeze schedule**: 5 epochs frozen, then unfrozen.
- **Pre-norm**: LN inside the residual, before sublayer.
- **Faithfulness gap G**: deletion-test signal.
- **Saturation**: P_0 > 0.999 → deletion test cannot drop.
- **Parasitic malware**: GinMaster — embeds in host apps.

## 10.5 Commonly confused

| Concept              | What it is                          | What it is *not*                  |
|----------------------|-------------------------------------|-----------------------------------|
| Macro-F1             | Unweighted per-class F1 average     | Sample-weighted F1                |
| Micro-F1 (multiclass)| ≈ accuracy                          | Same as macro-F1                  |
| Attention            | Soft, learned, all-pairs lookup     | Guaranteed explanation            |
| Markov initialisation| Embedding init from SVD of T        | A real Markov chain inference     |
| G > 0                | Necessary for faithfulness          | Sufficient for faithfulness       |
| Pre-norm             | LN before sublayer                  | The original Vaswani convention   |
| Learned positional   | Indexed table of position vectors   | Sinusoidal encoding               |
| Head truncation      | Keep first L_max tokens             | Random subsampling                |

---

# §11. Weak Points & Defense Strategies

Listed in order of *severity*. Each item: **issue → where it appears in the LaTeX → what examiner will ask → honest model answer → what to be ready for as the follow-up**. `paper.tex` is treated as authoritative throughout.

## 11.1 Batch size: 32 vs 64

**Issue**. `main.tex` L1020 says batch size = 32. `paper.tex` L299 says batch size = 64. Same project, two numbers, both published.
**Likely question**. "What batch size did you actually use?"
**Model answer**.
> "The authoritative number is 64, in the conference paper. The thesis draft contains an older 32 that was not propagated when batch size was changed to 64 for the final runs. Macro-F1 differences between batch sizes of 32 and 64 on a model this size were within fold-noise, but 64 is what the reported results were trained at."
**Follow-up to expect**. "Are your reported results reproducible at batch 32?" → "Within ±std reported in the tables, yes."

## 11.2 Vocabulary leakage — the thesis contradicts itself

**Issue**. `main.tex` L745–748 says vocab is "built on the training folds only". `main.tex` L1526–1529 (Limitations) says vocab is "built on the full corpus before fold splitting, creating a soft leakage". `paper.tex` L549–550 agrees with the Limitations section. So *paper.tex + main Limitations* both say "full corpus", while *main.tex Methodology* says "training folds only". Authoritative reading: **full corpus, soft leakage**.
**Likely question**. "Your methodology and limitations contradict each other. Which is right?"
**Model answer**.
> "The Limitations section is correct. The vocabulary was constructed on the full 9,337-sample corpus before fold splitting. This is a soft leakage — leakage of *token existence* (which APIs appear at all in the dataset), not of token statistics and not of labels. It affects which test-set tokens get mapped to UNK; tokens absent from training would otherwise become UNK in any production setting. We acknowledged this explicitly. The wording in the methodology section is an artefact of an earlier draft and is overridden by the Limitations section."
**Follow-up**. "How big is the effect?" → "A handful of tokens per fold. Did not re-run with per-fold vocab; will do for a journal version."

## 11.3 D'Angelo et al. ~99% vs our ~95% RF

**Issue**. D'Angelo et al. 2023 report ~99% accuracy on the *same corpus*; our RF reaches 95% on a 35×-imbalanced 9,337-sample split with 3-fold CV. The numbers don't reproduce.
**Likely question**. "Why can't you reproduce D'Angelo's accuracy?"
**Model answer**.
> "Three differences make the comparison non-direct. First, **corpus size**: D'Angelo report results on roughly 3,500 samples; we use the full 9,337-sample release, which is larger and includes harder, more class-overlapping samples — particularly in the minority families. Second, **evaluation protocol**: they report on a single held-out test split; we report mean ± standard deviation over 3-fold CV, which is a stricter, lower-mean estimator. Third, **feature representation**: their best result uses k-spaced rules directly as classifier features; we use them as Transformer embedding initialisation, which is a fundamentally different architectural choice. Our Markov Transformer reaches 0.927 accuracy on the larger corpus — about 6pp below their best, and we believe that gap is mostly explained by the three differences above."
**Follow-up**. "Did you try replicating their exact protocol?" → "No; reproducing prior work was not in scope. It would be a clean validation step for future work."

## 11.4 No ablation on k=1..10, SVD d=128, freeze=5

**Issue**. Three core hyperparameters of the Markov-init pipeline are not ablated.
**Likely question**. "How do you know those hyperparameters are optimal?"
**Model answer**.
> "We don't claim they are optimal. We chose `k ∈ {1..10}` by analogy to D'Angelo et al. — they used the same range and we wanted comparability. We chose `d = 128` to match `d_model` so the SVD output drops directly into the embedding layer with no projection. We chose freeze = 5 epochs as a small constant — long enough to let attention adapt to a fixed embedding geometry, short enough that the model still has ~95 epochs to fine-tune end-to-end. None of these were swept. The contribution of this thesis is the *methodology* — applying a deletion test to a Markov-initialised Transformer and quantifying the accuracy-vs-faithfulness tradeoff. Sensitivity analysis on these three hyperparameters is the cleanest natural extension and is in the future-work section."
**Follow-up**. "What would you do first?" → "k-spacing sweep on a subset; that's the cheapest to run and the most likely to move the result."

## 11.5 "G > 0 is necessary but not sufficient"

**Issue**. The thesis itself says (main.tex L1067–1071) that `G > 0` is necessary but not sufficient for faithful explanation. It then *uses* `G > 0` as the main evidence of faithfulness. Sharp examiners notice the gap.
**Likely question**. "If G > 0 isn't sufficient, what evidence do you have that the Markov model's attention is actually faithful — rather than just more concentrated?"
**Model answer**.
> "Two-part answer. First, what `G > 0` *does* rule out: random attention, uniform attention, and reverse-attention (attention concentrated on irrelevant tokens). It is a non-trivial necessary check. Second, what *would* close the gap to sufficiency: causal-counterfactual testing — for each top-attended token, replace it with a behaviourally distinct token and verify the prediction shifts in the predicted direction. That's listed as future work. We're explicit in the thesis that we make a partial-faithfulness claim — concentration plus alignment — not a full mechanistic-faithfulness claim. The 10.1pp gap in G at m=20 is consistent with the Markov model relying on a more compact, identifiable token subset; we don't claim that subset is the *causal* mechanism of the decision."
**Follow-up**. "Could attention be high on a token that the model doesn't actually use?" → "Yes — that's the Jain 2019 attack. The deletion test addresses the *opposite* direction (if attention is high, masking should hurt), which is what we measure."

## 11.6 GinMaster "parasitic" — qualitative claim, no quantitative breakdown

**Issue**. The thesis labels GinMaster "parasitic" (main.tex L1364) to explain its -39.9pp G. There's no quantitative breakdown of host-app vs malicious tokens per trace.
**Likely question**. "How much of a GinMaster trace is host-app noise versus malicious signal? Can you quantify it?"
**Model answer**.
> "We don't have a per-trace quantitative breakdown, and you're right that it would strengthen the claim. Qualitatively: GinMaster injects itself into legitimate apps, so its trace is dominated by whatever the host app does — UI lifecycle, generic I/O, network calls — with a small set of characteristic privilege-escalation and backdoor APIs interspersed. The global transition matrix T is shaped by the *bulk* of the corpus, which is not GinMaster; so the high-weight transitions in T encode patterns from other families. When we copy U_d into the embedding, GinMaster's discriminative tokens are *not* the ones with large embedding norms. Per-class transition matrices would directly address this — they're listed as the first future-work item."
**Follow-up**. "Would you call this a failure of the method?" → "A boundary condition. The method works when global behavioural structure aligns with discriminative signal. It fails when discriminative signal is a *deviation from* the global structure. That's a clean, interpretable boundary."

## 11.7 Jisut G ≈ 0 — saturation vs unfaithfulness

**Issue**. Jisut's deletion-test gap is 0% in both models. Doesn't *look* like a faithfulness result.
**Likely question**. "If your model's attention on Jisut has G = 0, doesn't that mean the attention is meaningless for Jisut?"
**Model answer**.
> "No. It's a measurement-ceiling effect. Both models predict Jisut with `P₀ > 0.999`. The deletion test measures `(P₀ − P_masked)`. If `P₀` is at the softmax ceiling, the gap can't be large no matter how important the masked tokens are. The metric saturates. Two mitigations would distinguish saturation from unfaithfulness: (i) temperature-scale the logits to push P₀ away from the ceiling, or (ii) measure deletion in logit space rather than probability space. Both are clean future-work items. This is also what we mean when we say `G > 0` is necessary but not sufficient — and *separately*, `G = 0` is not even necessarily evidence of failure."
**Follow-up**. "Did you try logit-space deletion?" → "No; flagged as future work."

## 11.8 Small per-family test sets (especially SmsPay)

**Issue**. Deletion test uses 30 samples per family. SmsPay has 173 total samples; 30 per evaluation is most of one fold's test partition. No confidence intervals reported.
**Likely question**. "Are your per-family gaps stable? You're reporting +41.2pp on what — 30 samples?"
**Model answer**.
> "The point estimate is on 30 samples; we did not compute CIs and we acknowledge that as a limitation. The qualitative direction is robust: SmsPay's 41.2pp gap is large enough that any reasonable bootstrap CI would still place the lower bound well above zero. The same is not true of small gaps like Opfake +3.2pp — that one we wouldn't defend as significant. For SmsPay, the size of the effect plus the mechanistic story (clear three-step API chain captured by k-spaced transitions) means even if the point estimate moves ±10pp under bootstrapping, the conclusion holds."
**Follow-up**. "Will you compute CIs?" → "Yes; cheap to add — pure post-hoc bootstrap over the existing deletion-test outputs."

## 11.9 Averaging attention across layers and heads

**Issue**. CLS attention is averaged across 2 layers × 4 heads = 8 attention rows. Different heads may specialise on different patterns; averaging can obscure this.
**Likely question**. "Why average? Don't you lose information about head specialisation?"
**Model answer**.
> "Averaging gives a single per-token importance scalar that is the input to the deletion test. Per-head analysis is shown qualitatively in §4.7 of the thesis, where different heads attend to different API clusters — so we don't deny that information exists. The choice to *average* for the deletion-test input is methodological simplicity: it gives a model-agnostic ranking. The proper alternative is attention rollout (Abnar & Zuidema 2020), which propagates attention through layers multiplicatively rather than averaging. We list rollout as future work."
**Follow-up**. "Why not rollout already?" → "Standard scope reasons; rollout would change the deletion-test ranking but not the test methodology."

## 11.10 Head truncation generalisability

**Issue**. Head-truncation superiority is a Droidmon-corpus-specific result; the justification (SDK init, reflection setup) is *why* it works for this corpus, but assumes malicious behaviour is concentrated early in the trace.
**Likely question**. "Would head truncation work on a corpus where the malicious behaviour appears late?"
**Model answer**.
> "Probably not, and we don't claim it would. For Droidmon traces of these eight families, the early-trace API calls (registration, SDK init, reflection setup, initial permission requests) are highly family-discriminative — we see this in both the sweep result (1.7-2.9pp head over tail) and in the qualitative attention heatmaps, where the model attends strongly to early positions. For a corpus where exfiltration or detection-evasion happens late (e.g. a slow-burn C&C), tail truncation or sliding-window approaches would be more appropriate. This is corpus-conditional, not a universal claim."
**Follow-up**. "What would you do for a generic corpus?" → "Sliding-window inference or learned dynamic-truncation rather than fixed head/tail."

## 11.11 Plain Transformer > Markov Transformer on accuracy

**Issue**. The Markov initialisation costs 2.6pp Macro-F1. The plain Transformer is the *better classifier*. Examiners may interpret this as "the contribution doesn't help".
**Likely question**. "Your Markov initialisation makes the classifier worse. Why is it a contribution?"
**Model answer**.
> "The Markov initialisation is not a contribution to *classification accuracy* — and we don't claim it is. The contribution is a controlled experiment: by holding everything else constant and varying only the embedding initialisation, we get a clean measurement of the accuracy-vs-faithfulness tradeoff that structural priors induce. The plain Transformer wins on accuracy by 2.6pp Macro-F1; the Markov Transformer wins on faithfulness by 10.1pp at m=20. Which model you deploy depends on the application: forensic / audit settings should pick Markov; pure-accuracy triage settings should pick the plain Transformer or BiLSTM. The honest framing is exactly what we report: a measurable tradeoff between two desirable properties."
**Follow-up**. "Is 2.6pp F1 too high a price for explanation?" → "Depends on the deployment. For forensic mode, where the analyst will *act* on the explanation, 2.6pp F1 in exchange for 10.1pp more concentrated, audit-able attention is a good trade. For automated triage where nobody reads explanations, you wouldn't pay it."

---

# §12. Style and Usage Notes

## 12.1 How to use this guide

- **First read**: cover to cover, slowly. Mark sections you struggle with.
- **Second read**: §4 (Transformers), §3 (math), and §11 (weak points). These are where the hardest questions come from.
- **Night before**: §10 (cheat sheet) + §11 (weak points). Just these two.

## 12.2 How to study the equations

Block + recall, not re-reading. Cover the right-hand side; write the equation; check. Equations to drill: (i) scaled attention; (ii) pre-norm block; (iii) faithfulness gap; (iv) class-weighted loss; (v) transition-matrix construction; (vi) SVD of T.

## 12.3 If asked something you genuinely don't know

- **Acknowledge** ("good question, I don't have a direct answer for that").
- **Restate** what you do understand ("what I can say is …").
- **Offer the nearest related thing** you do know ("the related result is …").
- **Volunteer follow-up** ("I'd want to test this by …").

Don't bluff. Examiners catch it; honest admission with structured fallback is the strongest move.

## 12.4 Tone

Be specific, be quantitative, be honest. The thesis already names its limitations — your job in viva is to *anticipate the question*, *open with the honest framing*, and *finish with the future-work direction*. The strongest defense is one that pre-empts critique, not one that resists it.

---

*End of guide.*

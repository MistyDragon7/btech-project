# Deep Learning Concepts — In Depth (for GAME-Mal)

This is the companion to `STUDY_GUIDE.md`. The study guide covers **what GAME-Mal does**; this document covers **why the DL machinery works** and gives you the vocabulary to defend every choice.

Everything here is written to be read *in order*. Later sections assume earlier ones.

---

## Table of Contents

1. [Mental Model: What is a neural network actually doing?](#1-mental-model-what-is-a-neural-network-actually-doing)
2. [Embeddings — Turning Discrete Tokens Into Vectors](#2-embeddings--turning-discrete-tokens-into-vectors)
3. [Linear Layers and Why They're Everywhere](#3-linear-layers-and-why-theyre-everywhere)
4. [Activation Functions](#4-activation-functions)
5. [Softmax, Sigmoid, and Probabilities](#5-softmax-sigmoid-and-probabilities)
6. [Cross-Entropy Loss](#6-cross-entropy-loss)
7. [Backpropagation and Gradients (Intuition, Not Calculus)](#7-backpropagation-and-gradients-intuition-not-calculus)
8. [Optimizers: SGD → Adam → AdamW](#8-optimizers-sgd--adam--adamw)
9. [Learning Rate and Schedulers](#9-learning-rate-and-schedulers)
10. [Regularization: Dropout, Weight Decay, Layer Norm](#10-regularization-dropout-weight-decay-layer-norm)
11. [Attention — Derived From Scratch](#11-attention--derived-from-scratch)
12. [Multi-Head Attention](#12-multi-head-attention)
13. [Positional Encodings](#13-positional-encodings)
14. [The Full Transformer Block](#14-the-full-transformer-block)
15. [Pre-Norm vs Post-Norm](#15-pre-norm-vs-post-norm)
16. [Padding, Masking, and Why It Matters](#16-padding-masking-and-why-it-matters)
17. [Pooling Strategies for Classification](#17-pooling-strategies-for-classification)
18. [The Gated Attention Mechanism (G1) — Math Deep Dive](#18-the-gated-attention-mechanism-g1--math-deep-dive)
19. [Class Imbalance Handling](#19-class-imbalance-handling)
20. [Gradient Clipping and Why Attention Needs It](#20-gradient-clipping-and-why-attention-needs-it)
21. [Early Stopping Explained](#21-early-stopping-explained)
22. [Evaluation Metrics Deep Dive](#22-evaluation-metrics-deep-dive)
23. [Cross-Validation — Why K-Fold?](#23-cross-validation--why-k-fold)
24. [Numerical Stability Tricks You Should Know](#24-numerical-stability-tricks-you-should-know)
25. [PyTorch-Specific Concepts That Often Trip People Up](#25-pytorch-specific-concepts-that-often-trip-people-up)
26. [Glossary](#26-glossary)

---

## 1. Mental Model: What is a neural network actually doing?

A neural network is a **parameterized function** `f_θ(x) = ŷ`.

- `x` is your input (for us: an encoded API sequence, shape `(batch, 512)`).
- `ŷ` is your output (logits over 8 malware families, shape `(batch, 8)`).
- `θ` are all the *learnable parameters* (weights and biases — millions of numbers).

Training = searching for the `θ` that makes `f_θ(x) ≈ y_true` for the training data, while generalizing to unseen data.

Three things you tune:
1. **Architecture** — the shape of `f` (layers, attention, etc.).
2. **Loss** — how you measure "f_θ(x) ≠ y_true".
3. **Optimizer** — how you update θ using the loss gradient.

That's it. Everything else is a variant.

---

## 2. Embeddings — Turning Discrete Tokens Into Vectors

### Why embeddings?
API call names like `java.io.File.exists` are discrete categories. Deep networks want continuous vectors. The naive approach (one-hot of vocab_size=1118 dims) wastes space and encodes no similarity (every pair is equidistant).

### What an embedding layer does
An embedding is literally a **lookup table**:
```python
self.api_embedding = nn.Embedding(vocab_size=1118, d_model=128)
```
- Parameter matrix: shape `(1118, 128)`.
- Forward pass: `embedded[i] = table[token_id[i]]`.
- `table` is learned by backprop — semantically similar API calls end up close in the 128-dim space.

### Why d_model = 128?
- Too small → can't separate 1118 APIs clearly.
- Too large → overfits and wastes compute.
- Rule of thumb: `d_model ≈ √vocab_size × 4` ≈ 133. We chose 128 (power of 2, GPU-friendly).

### `padding_idx=0`
Setting `padding_idx=0` tells PyTorch: "the gradient of index 0 is always zero — don't update it." We use index 0 for `<PAD>` tokens that fill out sequences shorter than 512. Without this flag, pad embeddings would absorb gradient noise.

---

## 3. Linear Layers and Why They're Everywhere

```python
nn.Linear(in_features, out_features)
# computes y = x @ W.T + b
# W has shape (out_features, in_features)
# b has shape (out_features,)
```

Almost every computation in a transformer is a linear layer. QKV projections? Linear. Output projection? Linear. FFN? Two linears with a non-linearity between them. Classifier head? Linear.

Why so many? Because linear layers are where the **learning** happens — non-linearities shape the function but don't have many parameters. Stacking linears with non-linearities in between lets the network approximate arbitrary functions (Universal Approximation Theorem).

### A subtle point: `nn.Linear` projections are parameterized matrix multiplies
When you see `self.w_q = nn.Linear(d_model, d_model)`, you should read: "a learnable `d_model × d_model` matrix that projects tokens into query space." The layer's *weights* are the matrix.

---

## 4. Activation Functions

You need non-linearities between linear layers — otherwise stacking linears is still linear (collapses to one big matmul).

| Name | Formula | Where used | Why |
|---|---|---|---|
| **ReLU** | `max(0, x)` | Older nets | Simple, fast, sparse |
| **GELU** | `x · Φ(x)` (Φ = Gaussian CDF) | Transformers (us, BERT, GPT) | Smoother than ReLU; better gradient flow |
| **Sigmoid** | `1 / (1 + e^-x)` | Gating, binary output | Squashes to (0, 1) |
| **Tanh** | `(e^x - e^-x) / (e^x + e^-x)` | RNNs | Squashes to (-1, 1), zero-centered |
| **Softmax** | `e^x_i / Σ e^x_j` | Final layer for classification | Turns logits into a probability distribution |

### GELU vs ReLU (why we use GELU in the FFN)
- ReLU kills half the inputs (negative → 0). Those dead neurons can stay dead forever.
- GELU smoothly lets small negatives through (`GELU(-1) ≈ -0.16`). Smoother loss surface, better training dynamics in transformers.

---

## 5. Softmax, Sigmoid, and Probabilities

Both produce numbers in [0, 1], but they mean very different things.

### Sigmoid: per-unit independent probability
```
σ(x_i) = 1 / (1 + e^-x_i)    # each element computed independently
```
Output is **not a distribution** — sum can be anything. Use when each output is an independent yes/no (e.g., gate activation, multi-label classification).

### Softmax: normalized joint distribution
```
softmax(x)_i = e^x_i / Σ_j e^x_j
```
Output **always sums to 1**. Use when outputs are mutually exclusive (one class wins).

### Why `x - max(x)` before softmax? (numerical stability)
`e^1000 = overflow`. Subtracting the max makes the largest exponent 0:
```
softmax(x) = e^(x_i - max(x)) / Σ e^(x_j - max(x))   # mathematically identical
```
All exponents are now ≤ 0, so all `e^...` are ≤ 1 — no overflow.

---

## 6. Cross-Entropy Loss

### Formula
For one sample with true class `c` and predicted logits `z`:
```
CE(z, c) = -log(softmax(z)_c)
         = -z_c + log(Σ_j e^z_j)
```

### Intuition
- If the model is very confident in class c (z_c much larger than others), `softmax(z)_c → 1`, `log(1) = 0`, **loss → 0**.
- If it's confident in the wrong class, `softmax(z)_c → 0`, `log(0) → -∞`, **loss → ∞**.

Cross-entropy is the **negative log-likelihood** of the true class under the model's distribution. Minimizing it = maximizing the probability assigned to the correct class.

### Why not use accuracy as the loss directly?
Accuracy is discrete (right or wrong) — its gradient is zero almost everywhere. You can't do gradient descent on it. Cross-entropy is a smooth surrogate that moves in the "more-correct" direction continuously.

### Class-weighted CE (what we use)
```python
weight_c = 1 / count_c                      # inverse frequency
weight = weight / weight.sum() * num_classes  # normalize to sum to num_classes
criterion = nn.CrossEntropyLoss(weight=weight)
```
This multiplies the loss for class `c` by `weight_c`. Airpush (63% of data) gets small weight; Fusob (2%) gets large weight. The gradient signal for a rare class is amplified so the model can't win just by predicting Airpush every time.

---

## 7. Backpropagation and Gradients (Intuition, Not Calculus)

You don't need to do the derivatives by hand — PyTorch's autograd does it. But you need the mental picture.

### What backprop does
For every parameter `θ_i`, compute `∂Loss / ∂θ_i` — "if I nudge this parameter by ε, how much does the loss change?"

### How
Chain rule applied backwards through the computation graph:
- During forward pass, PyTorch records every operation.
- During `loss.backward()`, it walks the graph *backwards* from loss to inputs, multiplying partial derivatives.

### Update rule (simplest: SGD)
```
θ ← θ - learning_rate · ∂Loss/∂θ
```

Repeat for every batch, every epoch. If the loss surface is well-behaved, θ converges to a minimum.

### Why gradients can vanish or explode
- **Vanish**: gradients get multiplied through many layers, each time by a number < 1 → exponentially small → no learning in early layers.
- **Explode**: same but × numbers > 1 → NaN losses.
- Fixes: ReLU/GELU (don't saturate), residual connections, layer norm, gradient clipping.

---

## 8. Optimizers: SGD → Adam → AdamW

### SGD (Stochastic Gradient Descent)
```
θ ← θ - lr · g                   # g = gradient
```
Simple, but sensitive to lr. Makes you pick a single lr for every parameter — bad when some weights need to move fast, others slow.

### SGD with momentum
```
v ← β · v + g
θ ← θ - lr · v
```
Velocity accumulates like a ball rolling down a hill — smooths out noisy gradients and helps escape shallow local minima.

### Adam
Adam maintains **per-parameter** running averages:
```
m ← β1 · m + (1 - β1) · g          # 1st moment (mean of grad)
v ← β2 · v + (1 - β2) · g²         # 2nd moment (mean of grad²)
m̂ ← m / (1 - β1^t)                 # bias-corrected
v̂ ← v / (1 - β2^t)
θ ← θ - lr · m̂ / (√v̂ + ε)
```
Parameters with historically small gradients get boosted (divide by small √v). Parameters with historically large gradients get dampened. Works well out of the box on most problems.

### AdamW (what we use)
Adam has a subtle bug: when combined with L2 weight decay, the decay gets mixed into the gradient and scaled by `1/√v`. AdamW **decouples weight decay** from the gradient step:
```
θ ← θ - lr · (m̂ / (√v̂ + ε) + weight_decay · θ)
```
Cleaner theoretically, empirically better on transformers. Almost every modern transformer uses AdamW.

### Our config
```python
torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
```

---

## 9. Learning Rate and Schedulers

### Why lr is the most important hyperparameter
- lr too high → training diverges (loss goes to NaN).
- lr too low → training is slow / gets stuck in bad minima.
- "Perfect" lr changes over training: you want big steps early, small steps late.

### Cosine annealing (what we use)
```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
```
Follows a cosine curve from initial lr → 0 over `T_max` epochs:
```
lr_t = lr_initial · (1 + cos(π · t / T_max)) / 2
```
- Big steps at the start (explores).
- Small steps near the end (fine-tunes).
- Smooth (no abrupt drops like step-decay).

### Warmup (not used here but common)
Linearly ramp lr from 0 → lr_peak over the first ~1000 steps, *then* start cosine. Prevents early instability in transformers.

---

## 10. Regularization: Dropout, Weight Decay, Layer Norm

Regularization = anything that helps generalization (keeps test loss low when train loss is low).

### 10.1 Dropout
During training, randomly zero out each activation with probability `p`:
```python
self.dropout = nn.Dropout(0.15)  # we use p=0.15
```
- Prevents neurons from co-adapting (relying on specific other neurons).
- At inference, dropout is off (all neurons active, outputs scaled by `1/(1-p)` to compensate).

### 10.2 Weight Decay
Add a penalty proportional to parameter magnitude to the loss:
```
L_total = L_CE + λ · ||θ||²     # λ = weight_decay = 1e-4
```
Pushes parameters toward zero unless they're earning their keep. Prevents overfitting by penalizing overly complex models.

### 10.3 Layer Normalization
For each sample, normalize activations across the feature dimension:
```
LN(x) = γ · (x - mean) / √(var + ε) + β
```
- `γ`, `β` are learnable (per-feature).
- Keeps activations well-scaled throughout the network.
- Essential for deep transformers — without it, training is wildly unstable.

### LayerNorm vs BatchNorm
- BatchNorm normalizes across the batch dimension. Works poorly with small batches or variable-length sequences. Not used in transformers.
- LayerNorm normalizes per sample. Batch-size independent. Standard in transformers.

---

## 11. Attention — Derived From Scratch

### The problem attention solves
A CNN sees local patterns. An RNN processes sequentially but forgets. To model **any-range dependencies** between tokens, you need each token to look at every other token and decide who matters.

### The formula (and what each piece does)
```
Q = X · W_Q       # queries:  "what am I looking for?"
K = X · W_K       # keys:     "what do I offer?"
V = X · W_V       # values:   "what's my content?"

Attention(X) = softmax(Q · K.T / √d_k) · V
```

### Walking through it
1. Project the input into three spaces: Q, K, V. These are learned.
2. **`Q · K.T`** — dot product of every query with every key. Result: `(N, N)` matrix of compatibility scores. `scores[i, j]` = "how much does token i want to look at token j?"
3. **`/ √d_k`** — scale down. Without this, for large `d_k`, the dot products get huge, softmax saturates, and gradients vanish.
4. **`softmax(..., dim=-1)`** — convert each row to a probability distribution over keys. Now each token has attention weights summing to 1.
5. **`... · V`** — weighted sum of value vectors. Each output position is a blend of all value vectors, weighted by the attention weights.

### Why it's called "attention"
Token i *attends to* token j proportionally to the attention weight `α[i, j]`. Attention weights are the interpretable intermediate — you can see what each token looks at.

---

## 12. Multi-Head Attention

### Why one attention is not enough
A single attention head has to learn one way of relating tokens. In practice, different kinds of relations matter simultaneously:
- Position relation (near vs far)
- Semantic relation (crypto APIs cluster)
- Temporal relation (before vs after)

### Split the d_model space into heads
With `d_model = 128` and `n_heads = 4`:
- Each head gets `d_k = d_model / n_heads = 32` dimensions.
- Each head does its own attention: Q_h, K_h, V_h are projections into 32-dim space.
- After attention, concatenate all heads back to 128 dims.

### Code sketch
```python
Q = self.w_q(x).view(B, N, n_heads, d_k).transpose(1, 2)
# shape: (B, n_heads, N, d_k)
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
attn = softmax(scores)
out = attn @ V
out = out.transpose(1, 2).contiguous().view(B, N, d_model)
out = self.w_o(out)
```

### The `view(...).transpose(...)` trick
PyTorch stores tensors as a single flat buffer with a *shape* descriptor. `view` reshapes without moving data; `transpose` permutes dimensions without copying. Together they let you go from `(B, N, d_model)` to `(B, n_heads, N, d_k)` cheaply.

---

## 13. Positional Encodings

### Why attention is position-blind
`softmax(QK.T/√d)V` is a **permutation-equivariant** operation. Shuffle the input tokens and the outputs shuffle the same way — but the *relationships* computed are identical. This is bad: API call **order** matters a lot (a `Cipher.init` before `Cipher.doFinal` is a crypto op; reversed it's meaningless).

### Fix: inject position information
Before the first attention layer, add a per-position vector:
```python
h = api_embedding + pos_embedding
```

Two flavors:
- **Learned** (what we use): `nn.Embedding(max_seq_len, d_model)` — one vector per position, trained like any other weights.
- **Sinusoidal** (original Transformer): fixed sinusoids at different frequencies — generalizes to longer sequences at inference than training.

For our case, sequences are capped at 512 and learned positions work fine.

### Rotary/relative positions
Modern LLMs use rotary or relative positional encodings — they're better for generalization to longer contexts. Overkill here.

---

## 14. The Full Transformer Block

Our `GAMEMalBlock` (pre-norm):

```
x_in
 ├─ LN → GatedAttention → residual add
 │
 └─ LN → FFN (Linear → GELU → Linear) → residual add
 │
x_out
```

### Residual (skip) connections
`x_out = x_in + f(LN(x_in))`. Critical for deep networks — gradients can flow around `f` during backprop, fighting vanishing gradients. If f is a no-op (zero), identity is preserved. So the worst-case is "layer does nothing," never "layer destroys information."

### FFN (Feed-Forward Network)
Every position gets processed independently by a two-linear MLP:
```
FFN(x) = Linear₂(GELU(Linear₁(x)))
```
`Linear₁` expands d_model (128) → d_ff (256). `Linear₂` contracts back to 128. The expansion-and-contraction lets the FFN do computation at higher dimension, then compress.

Think of attention as **mixing across positions** and FFN as **mixing across features at each position**. Stacking them alternately builds up representations.

---

## 15. Pre-Norm vs Post-Norm

The original Transformer (2017) used post-norm:
```
x_out = LN(x_in + f(x_in))       # post-norm
```
Problem: gradients through `f` get squished by the final LN, causing training instability for deep models.

Modern transformers (ours included) use pre-norm:
```
x_out = x_in + f(LN(x_in))       # pre-norm
```
The residual path is a clean identity — gradients flow unchanged. Much more stable, can be trained to deeper stacks.

---

## 16. Padding, Masking, and Why It Matters

### The padding problem
We pad sequences to length 512. If we don't tell the model those are pads:
- Attention would attend to pad positions.
- The global-pool would average real and pad embeddings together.
- The model would learn to rely on padding, which defeats the purpose.

### Two masks
1. **Attention mask**: in `scores.masked_fill(mask, -inf)`, set score at pad positions to -∞ so softmax gives them 0 weight.
2. **Pool mask**: in pooling, divide by the number of real tokens, not the padded length.

### Code pattern (from our model)
```python
pad_mask = (x == 0)                              # True where padded
# attention:
scores = scores.masked_fill(pad_mask[:, None, None, :], -inf)
# pooling:
mask_expanded = (~pad_mask).unsqueeze(-1).float()
h_sum = (h * mask_expanded).sum(dim=1)
lengths = mask_expanded.sum(dim=1).clamp(min=1)
h_pooled = h_sum / lengths
```

Forgetting masks is one of the most common causes of "my model gets 100% train accuracy but 50% test accuracy."

---

## 17. Pooling Strategies for Classification

After the transformer, you have shape `(B, N, d_model)`. For classification, you need `(B, d_model)`. How to collapse N?

| Method | Formula | Pros / Cons |
|---|---|---|
| **[CLS] token** | Prepend a learnable token, use its final embedding | BERT standard; needs extra token |
| **Mean pool** (ours) | Average all non-pad positions | Simple, symmetric |
| **Max pool** | Take max across positions per feature | Captures strongest signal |
| **Attention pool** | Learn a small attention layer over positions | Most expressive, more params |

We use masked mean pool: it has no extra params, works with sequences of any length, and in our experiments with gated attention, the gate already provides per-position weighting inside the blocks.

---

## 18. The Gated Attention Mechanism (G1) — Math Deep Dive

This is the core innovation. Fully derive it so you can defend it.

### Standard MHA
```
O = (softmax(Q K.T / √d_k) · V) · W_O        # (B, N, d_model)
```
Between V and W_O, the composition is **linear**. You have:
```
O = attn(Q, K) · V · W_O
```
where `attn` is a fixed matrix (given Q, K). V and W_O are both learned linears — their product is also a linear. So the model can merge them into one matrix during training. **This is a rank bottleneck**: the effective parameter count is `d_model × d_model`, not `2 × d_model × d_model`.

### Adding the gate
```
gate = sigmoid(X · W_gate)                   # (B, N, d_model)
O = ((attn · V) ⊙ gate) · W_O                # ⊙ = element-wise
```

### What changes
- `gate` depends on the **input** X (query-dependent).
- `sigmoid` is **non-linear** — it can't be merged into W_O.
- The gate **breaks the rank bottleneck** — V · W_O is no longer linearly equivalent because of the non-linear ⊙.

### Head-specific gate
In our implementation, `W_gate` has output dim `d_model`, which we reshape to `(n_heads, d_k)`. This means each head has its own gating pattern — head 0 might learn to gate on "API name type," head 1 on "position," etc.

### Sparsity from sigmoid
Sigmoid saturates:
- `σ(−∞) = 0`, `σ(∞) = 1`.
- If the pre-activation is large negative, gate ≈ 0, **activation is suppressed**.
- Empirically, ~80% of gates settle near 0 after training. That's the "sparsity" property from Qiu et al.

### The bias = -2.0 initialization
```python
nn.init.constant_(self.w_gate.bias, -2.0)
```
- Without this, bias = 0, sigmoid(0) = 0.5 — every gate is "half on" at init.
- With bias = -2.0, sigmoid(-2) ≈ 0.12 — gates start *sparse*. Training only needs to carve out the few tokens to promote, not suppress the noisy majority.
- Saves ~20% of training epochs to reach the same sparsity level.

### Why not tanh instead of sigmoid?
Tanh ranges [-1, 1]. If gate could go negative, it would flip the sign of attention outputs, creating instability and ambiguous semantics. Sigmoid's [0, 1] range is a clean "scale down" interpretation.

---

## 19. Class Imbalance Handling

### The raw problem
If 63% of data is Airpush, a dumb classifier that always predicts "Airpush" gets 63% accuracy. Cross-entropy loss doesn't care — predicting Airpush is the "path of least resistance" for the optimizer.

### Three knobs to turn

1. **Resample the data** — oversample minorities or undersample the majority. Downside: oversampling risks overfitting to small classes; undersampling throws away data.
2. **Loss weighting** (we use this) — upweight loss contributions from minority samples so the gradient signal is balanced.
3. **Use the right metric** — macro-F1 punishes ignoring minorities. Micro-F1 does not.

### Our exact formula
```python
class_counts = [5880, 1257, 166, 312, 523, 430, 596, 173]
weights = 1 / counts                        # inverse frequency
weights = weights / sum(weights) * num_classes   # normalize so sum = num_classes
# now Fusob gets weight ~5, Airpush gets weight ~0.05
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

Normalizing so the weights sum to `num_classes` keeps the overall loss scale the same as unweighted — makes it easier to compare training curves across runs.

---

## 20. Gradient Clipping and Why Attention Needs It

Attention involves dot products of potentially large vectors; gradients through softmax can occasionally spike. A single batch with a bad spike can send weights into NaN land permanently.

### Gradient clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
If the total gradient norm exceeds 1.0, scale all gradients down so the norm is exactly 1.0. Caps the per-step damage of any single bad batch.

This is a standard stabilizer for transformers. You will almost never train a transformer without it.

---

## 21. Early Stopping Explained

### Why
Training loss monotonically decreases (or should). Test/validation loss has a U-shape: first it decreases (model is learning), then increases (model is overfitting). You want to stop at the minimum of the validation curve.

### Our implementation
```python
if metrics["f_score"] > best_f1:
    best_f1 = metrics["f_score"]
    best_state = { ... }    # save weights
    no_improve = 0
else:
    no_improve += 1
    if no_improve >= patience:
        break
```

- Track the best F1 so far.
- If the current epoch beats it, reset patience counter.
- If patience (10 epochs) runs out without improvement, stop and restore the best weights.

### Why F1, not accuracy?
Same reason we macro-average metrics: accuracy is biased toward the majority class. If Airpush accuracy is fine but Fusob accuracy is 0, overall accuracy barely moves but F1 drops sharply. F1 is a faithful objective for imbalanced multi-class.

---

## 22. Evaluation Metrics Deep Dive

For a **binary** confusion matrix with classes {positive, negative}:
- **TP** — true positives (correctly predicted positive)
- **FP** — false positives (wrongly predicted positive)
- **FN** — false negatives (positive missed)
- **TN** — true negatives

Metrics:
| Metric | Formula | Question answered |
|---|---|---|
| Accuracy | (TP+TN)/Total | Overall correctness |
| Sensitivity (Recall) | TP/(TP+FN) | "Of actual positives, how many did I catch?" |
| Specificity | TN/(TN+FP) | "Of actual negatives, how many did I spare?" |
| Precision | TP/(TP+FP) | "Of my positive predictions, how many are real?" |
| F1 | 2·P·R/(P+R) | Harmonic mean of P and R |
| AUC-ROC | area under ROC curve | Threshold-independent ranking quality |

### Multi-class macro averaging
For K classes, compute each metric **per class** (one-vs-rest), then **average unweighted**:
```
macro-F1 = (F1_class_0 + F1_class_1 + ... + F1_class_{K-1}) / K
```
Each class counts equally regardless of frequency.

### AUC via one-vs-rest
For each class c: treat "c vs all others" as binary, compute AUC, then average. We use `sklearn`'s `roc_auc_score(..., multi_class='ovr')`.

### Why F1 uses harmonic mean instead of arithmetic
Arithmetic mean (P+R)/2 gives a "decent" score when one is very high and the other very low — misleading. Harmonic mean `2PR/(P+R)` is dominated by the smaller of the two. A classifier with 100% precision and 5% recall has arithmetic mean 52.5% but harmonic mean 9.5%. F1 tells the truth.

---

## 23. Cross-Validation — Why K-Fold?

### The single-split problem
If you just split 80/20 once, your performance estimate depends on which 20% was unlucky. Variance can swing results by several percentage points.

### K-Fold CV
- Split data into K folds.
- For each fold k: train on the other K-1 folds, evaluate on fold k.
- Average the K scores → a lower-variance performance estimate.

### Stratified K-Fold (what we use)
Standard K-Fold can produce folds with zero Fusob samples. Stratified K-Fold **preserves class ratios** in each fold. Essential for imbalanced data.

### Our choice (K=3)
Compute budget on MPS. K=3 gives an honest variance estimate at roughly 1/3 the cost of K=10. For the final paper run you'd bump to K=5 or K=10.

---

## 24. Numerical Stability Tricks You Should Know

| Trick | Where | Why |
|---|---|---|
| **Subtract max before exp** | `softmax`, `logsumexp` | Prevent `e^big = overflow` |
| **Log-softmax instead of log(softmax)** | Computing CE loss | Single-pass, stable |
| **ε in divisions** | `1 / (std + 1e-5)` in LayerNorm | Avoid div-by-zero |
| **Clamp logits** | Before softmax in extreme cases | Cap runaway values |
| **Check for NaN** | During long training | Catch corruption early |
| **Mixed precision (fp16)** | Large models | Halves memory at some precision cost |

---

## 25. PyTorch-Specific Concepts That Often Trip People Up

### `model.train()` vs `model.eval()`
- **`train()`** enables dropout and batchnorm running stats.
- **`eval()`** disables dropout; batchnorm uses accumulated stats.
- Forgetting `eval()` at test time gives noisy results.

### `@torch.no_grad()`
Disables gradient tracking in a block. Use at inference — saves memory and speeds things up:
```python
@torch.no_grad()
def evaluate(...):
    ...
```

### `optimizer.zero_grad()`
Gradients **accumulate** by default in PyTorch. You must zero them before each backward pass:
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### `.to(device)`
Tensors and models must be on the same device. For MPS (Apple Silicon):
```python
device = torch.device("mps")
model.to(device)
X = X.to(device)
```

### `.contiguous()`
After a `transpose` or `permute`, the tensor's memory layout may not be contiguous. `view` requires contiguous memory — use `.contiguous()` before `view` or use `.reshape()` which handles this automatically.

### `DataLoader` and `TensorDataset`
`DataLoader` wraps a dataset and handles batching, shuffling, and (optionally) parallel workers:
```python
ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=32, shuffle=True)
for X_batch, y_batch in loader:
    ...
```

### Model state dict
`model.state_dict()` returns a dict of all parameter tensors. Save/load for checkpointing:
```python
torch.save(model.state_dict(), "best.pt")
model.load_state_dict(torch.load("best.pt"))
```

---

## 26. Glossary

| Term | Plain meaning |
|---|---|
| **Tensor** | Multi-dimensional array (PyTorch's fundamental data type) |
| **Batch** | A group of samples processed together for efficiency |
| **Epoch** | One full pass through the training data |
| **Iteration / step** | One batch's forward + backward + update |
| **Parameter** | A single learnable number (weights and biases) |
| **Hyperparameter** | Something you set before training (lr, batch_size, num_layers) |
| **Embedding** | A learned dense vector for a discrete token |
| **Logits** | Raw unnormalized scores before softmax |
| **Softmax** | Converts logits into a probability distribution |
| **Cross-entropy** | Standard classification loss; negative log-likelihood of true class |
| **Gradient** | Vector of partial derivatives of loss w.r.t. parameters |
| **Backpropagation** | Algorithm that efficiently computes all gradients via chain rule |
| **Autograd** | PyTorch's automatic differentiation engine |
| **Optimizer** | Algorithm that uses gradients to update parameters |
| **Learning rate (lr)** | How big the parameter update step is |
| **Weight decay** | L2 regularization; shrinks parameters toward zero |
| **Dropout** | Randomly zeroing activations during training for regularization |
| **Layer norm** | Per-sample activation normalization across features |
| **Attention** | Weighted sum over positions; each output attends to all inputs |
| **Multi-head** | Run several independent attentions in parallel |
| **Positional encoding** | Per-position vector added to embeddings to inject order info |
| **Residual connection** | `x + f(x)` — lets gradient flow around a layer |
| **FFN** | Feed-forward network inside a transformer block |
| **Transformer block** | Attention + FFN + residuals + norms |
| **Pre-norm** | LayerNorm applied before the sub-layer (more stable) |
| **Mask** | Boolean tensor used to ignore certain positions (padding, future tokens) |
| **Pooling** | Collapsing a sequence dimension into one vector |
| **Early stopping** | Halt training when validation metric stops improving |
| **Overfitting** | Model memorizes training data, fails on unseen data |
| **Underfitting** | Model too simple to capture patterns |
| **Generalization** | How well train-time knowledge transfers to test data |
| **Class imbalance** | Training data has much more of some classes than others |
| **Macro metric** | Per-class metric averaged unweighted across classes |
| **K-fold CV** | Cross-validation across K train/test splits for robust estimates |
| **MPS** | Apple Silicon GPU backend in PyTorch |
| **State dict** | PyTorch's serializable model parameters dictionary |
| **Logit gate** | A pre-sigmoid value; sigmoid of it is the gate |
| **Sparsity** | How many activations are close to zero |
| **Gate saturation** | Sigmoid outputs very close to 0 or 1 — effectively on/off |

---

## Final Study Tips for the Defense

1. **Don't try to memorize math — memorize the role of each piece.** You'll be asked "why attention?" before "what's the exact formula?"
2. **Be able to draw the architecture on a whiteboard.** Embedding → +pos → N×(LN, gated attention, residual, LN, FFN, residual) → LN → pool → MLP → softmax. Practice until it's automatic.
3. **Have a ready two-sentence answer for every design choice.** "Why GELU not ReLU?" → "GELU is smoother, lets small negatives pass through, and is empirically better in transformers." Done.
4. **Know the failure modes.** Committee members love asking "when does this break?" Answers you have: class imbalance without weights, reflection calls without resolution, very long sequences beyond 512, adversarial API insertion.
5. **Tie everything back to the data.** "This matters because malware uses reflection." "Class weights matter because Airpush is 63% of our data." Concrete > abstract.

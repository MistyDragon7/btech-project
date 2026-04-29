# Project Summary

**Explainable Android Malware Classification via Transformer with Markov-Initialized Embeddings**

*Arav Jain — B.Tech Final Project*

---

We classify Android malware into 8 families using dynamic API call traces from the Droidmon sandbox. We compare a standard Transformer architecture against traditional ML baselines, and introduce a variant where the Transformer's embedding layer is initialized using SVD-factorized Markov transition matrices.

The Markov embedding approach is inspired by D'Angelo et al.'s (2023) work on k-spaced associative rules for malware behavior modeling. We use the same UMD corpus and their k-spaced transition methodology as the basis for constructing our global API transition matrix.

The primary focus is **intrinsic explainability**: using the Transformer's own `[CLS]` attention maps to identify which API calls drive each classification decision, without relying on external post-hoc methods like SHAP or LIME.

---

## Dataset

**Source:** UMD Android Malware corpus (8 families, 9,337 samples)

| Family | Samples | Behavior |
| :--- | ---: | :--- |
| Airpush | 5,880 | Aggressive adware, user tracking |
| DroidKungFu | 1,257 | Root exploit, data theft |
| Opfake | 596 | SMS fraud, code packing |
| GinMaster | 523 | Backdoor, privilege escalation |
| Jisut | 430 | Ransomware + SMS fraud |
| Genpua | 312 | Premium SMS fraud |
| SmsPay | 173 | Premium SMS fraud |
| Fusob | 166 | Ransomware |

Imbalance ratio: ~35× (Airpush vs Fusob). Handled via inverse class-weighted cross-entropy loss and stratified 3-fold CV.

---

## Classification Results (3-fold Stratified CV)

| Model | Accuracy | Macro F1 | AUC |
| :--- | ---: | ---: | ---: |
| GaussianNB | 0.860 | 0.782 | 0.925 |
| DecisionTree | 0.923 | 0.836 | 0.910 |
| LinearSVM | 0.923 | 0.844 | 0.980 |
| **Plain Transformer** | **0.940** | **0.884** | **0.981** |
| **Markov Transformer** | **0.927** | **0.858** | **0.976** |
| BiLSTM | 0.948 | 0.894 | 0.992 |
| Random Forest | 0.950 | 0.893 | 0.993 |

**Observations:**
- The Plain Transformer is competitive with Random Forest (within ~1pp F1) and outperforms all other traditional ML baselines except BiLSTM.
- The Markov Transformer trades ~2.6pp F1 relative to the Plain Transformer. It does not improve classification accuracy on this corpus.
- Random Forest and BiLSTM remain the top performers on raw accuracy and F1 on this dataset.

---

## The Markov Embedding Experiment

### What We Did
1. Built a global API transition matrix using k-spaced rules (k=1..10), inspired by D'Angelo et al.'s associative rule methodology, across all training sequences.
2. Applied Truncated SVD to compress the transition matrix into 128-dimensional dense vectors.
3. Injected these vectors as the initial weights of the Transformer's `api_embedding` layer.
4. Froze the embeddings for the first 5 epochs, then unfroze for fine-tuning.

### Classification Outcome
The Markov Transformer achieved lower classification metrics than the Plain Transformer (F1: 0.858 vs 0.884). The inductive bias from the global transition graph appears to constrain the model in ways that reduce its ability to discriminate between families on this corpus and at this scale.

### Explainability Outcome
Where the Markov initialization shows a more interesting effect is in the **Faithfulness Deletion Test**. This test measures how much the model's confidence drops when its top-attended tokens are removed.

**Overall Faithfulness (Δ attention-masked − Δ random-masked, k=20):**

| Model | Δ (attention) | Δ (random) | Faithfulness Gap |
| :--- | ---: | ---: | ---: |
| Plain Transformer | 18.7% | 1.7% | 17.0% |
| Markov Transformer | 27.2% | 0.1% | 27.1% |

The Markov Transformer's attention maps appear to be more tightly coupled to the tokens the model actually relies on for its predictions. When its top-attended tokens are removed, the confidence drop is ~10pp larger than the Plain Transformer's.

**Per-family breakdown (k=20):**

| Family | Plain Δ | Markov Δ | Direction |
| :--- | ---: | ---: | :--- |
| Airpush | 0.4% | 16.0% | Markov higher |
| DroidKungFu | 26.8% | 55.2% | Markov higher |
| Fusob | ~0.0% | 22.2% | Markov higher |
| Genpua | 19.7% | 30.7% | Markov higher |
| GinMaster | 51.9% | 12.0% | Plain higher |
| Jisut | ~0.1% | ~0.0% | Neither |
| Opfake | 21.1% | 24.3% | Markov higher |
| SmsPay | 15.6% | 56.8% | Markov higher |

In 6 of 8 families, the Markov Transformer's attention-based deletion causes a larger confidence drop. The exceptions are GinMaster (where the Plain Transformer is substantially more faithful) and Jisut (where neither model shows meaningful faithfulness at k=20).

**Interpretation:** These results are suggestive that Markov-initialized embeddings may encourage attention heads to focus on more structurally relevant tokens. However, this is a single-dataset observation on 240 test samples, and the faithfulness improvement comes at the cost of lower classification accuracy. We do not claim this as definitive evidence of a general principle.

---

## Explainability Pipeline

### Method
We use the `[CLS]` token's attention weights as a per-sample importance ranking over input API calls. For each sample:
1. Extract attention from `[CLS]` (position 0) to all other tokens.
2. Average across heads and layers.
3. Rank tokens by attention score.

### Faithfulness Validation
The **Deletion Test** validates whether the attention map is identifying tokens the model genuinely depends on:
- Mask the top-k highest-attention tokens (set to `<PAD>`), re-run inference, measure confidence drop.
- Compare against masking k random tokens.
- If Δ(attention) > Δ(random), the attention map has some predictive alignment with model behavior.

Both the Plain and Markov Transformers pass this test (Δ attention > Δ random at all tested values of k). The gap is larger for the Markov variant.

### Visualizations Generated
- `cls_attn_heatmap_per_family.png` — Per-family CLS attention heatmaps across layers
- `cls_attention_head_specialisation.png` — Per-head attention distribution
- `cls_attention_sink.png` — Positional attention distribution
- `markov_embeddings_similarity.png` — Cosine similarity of SVD-initialized embeddings

All visualizations are generated for both models independently (prefixed `plain_transformer_*` and `markov_transformer_*`).

---

## Architecture

- **Model:** 2-layer Transformer with learnable `[CLS]` token
- **Parameters:** ~515K
- **Embedding:** d_model=128, vocabulary=1,118 APIs (min_freq=2)
- **Attention:** 4 heads, d_k=32, d_ff=256
- **Sequence:** max_len=512, head-truncation (empirically selected via sweep)
- **Training:** AdamW, lr=5e-4, cosine annealing, patience=20, class-weighted CE loss
- **Preprocessing:** Reflection-aware tokenization (`REFL:class.method` for reflected calls)

---

## Key Design Decisions

1. **Head truncation over tail:** Empirically, keeping the first 512 tokens outperforms keeping the last 512 by 1.7–2.9pp F1 across all tested lengths. Early API calls (app registration, SDK init, reflection resolution) appear more family-discriminating on this corpus.

2. **`[CLS]` token over mean pooling for explainability:** The `[CLS]` token provides a natural anchor point for extracting per-token importance via attention weights, without requiring any post-hoc method.

3. **Global Markov graph (not per-class):** For the Markov embedding experiment, we build a single global transition matrix rather than per-class graphs. Per-class graphs cannot be used for embedding initialization because the class is unknown at inference time.

4. **Gated attention (GAME-Mal) is an honest negative:** The sigmoid gate from Qiu et al. (2025) does not improve accuracy at this corpus scale. Reported as an ablation.

---

## Reproducing

```bash
pip install -r requirements.txt

# Plain Transformer (baseline)
python scripts/run_plain_transformer_final.py

# Markov Transformer (ablation)
python scripts/run_markov_embedding_experiment.py

# Deletion test (explainability validation)
python scripts/run_deletion_test.py --model_prefix plain_transformer
python scripts/run_deletion_test.py --model_prefix markov_transformer

# Attention visualizations
python scripts/visualize_attention.py --model_prefix plain_transformer
python scripts/visualize_attention.py --model_prefix markov_transformer
```

Or on Google Colab:
```bash
!bash colab_markov_pipeline.sh
```

---

## Limitations

1. **Single dataset.** All results are on the UMD corpus (8 families). Generalization to other corpora (Drebin, CICAndMal2017) or to unseen families is untested.
2. **Vocabulary leakage.** The vocabulary is built on the full corpus before fold splitting. This is soft leakage (token existence, not label information). Documented but not fixed.
3. **Markov embeddings did not improve classification.** The hypothesized benefit of structural initialization did not materialize as higher F1 on this dataset. The faithfulness improvement is an interesting observation but not a substitute for classification performance.
4. **Faithfulness test is necessary but not sufficient.** Passing the deletion test indicates some alignment between attention and model behavior, but does not guarantee that the attention maps are complete or causally faithful explanations.
5. **Small minority classes.** Fusob (166 samples) and SmsPay (173 samples) have very few test samples per fold, making per-class metrics for these families noisy.

---

## Repo Layout

```
src/
  preprocessing.py      # Reflection-aware parsing + vocabulary
  markov.py             # k-spaced rule mining + SVD embedding generation
  model.py              # Transformer with [CLS] token
  train.py              # Training loop
  baselines.py          # sklearn classifiers + MarkovPruning

scripts/
  run_plain_transformer_final.py     # Plain Transformer 3-fold CV
  run_markov_embedding_experiment.py # Markov Transformer ablation
  run_deletion_test.py               # Faithfulness validation
  visualize_attention.py             # CLS attention heatmaps
  visualize_markov_embeddings.py     # Cosine similarity plot

results/
  results_summary.csv                # All models, 3-fold aggregates
  plain_transformer_final.json       # Plain Transformer fold details
  markov_transformer_final.json      # Markov Transformer fold details
  deletion_test.json                 # Plain Transformer faithfulness
  markov_transformer_deletion_test.json  # Markov faithfulness
  figures/                           # All generated plots
  models/                            # Saved weights + configs
```

## Author

Arav Jain

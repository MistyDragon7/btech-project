# GAME-Mal — Implementation Plan (Model Finalisation + Explainability)

> Paper writing is deferred. Focus: get the model performing at its best and
> make the explainability claims defensible.

---

## 1. Sequence-length + Truncation Sweep

### Motivation
`src/preprocessing.py::pad_sequences` takes `seq[:max_len]` — that is **head truncation**
(earliest API calls). The paper argues tail-truncation (retaining the *most recent* calls)
is better because malware payload executes late in the trace. This discrepancy must be
resolved empirically.

### Implementation
**New file:** `scripts/run_seq_len_sweep.py`

```
load dataset → len≥30 filter → build vocab (same as main)
for fold in [0,1,2]:
    for max_len in [256, 512, 768]:
        for trunc in ["head", "tail"]:
            X_train = pad(train, max_len, trunc)
            X_test  = pad(test,  max_len, trunc)
            model, metrics = train_game_mal(...)   # reuse src/train.py
            save row to results/seq_len_sweep.csv
aggregate → results/seq_len_sweep_summary.csv
```

Pad helper (add alongside existing `pad_sequences`):
```python
def pad_with_truncation(seqs, max_len, truncation="head"):
    result = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[-length:] if truncation == "tail" else seq[:length]
    return result
```

Architecture: identical to main run (`d_model=128`, `n_layers=2`, `n_heads=4`,
`d_ff=256`, `dropout=0.15`, `lr=5e-4`, `epochs=40`, `patience=10`, `batch=32`).

**Pick best (max_len, truncation) by mean macro-F1 across folds. Freeze it.**

---

## 2. BiLSTM Baseline

### Motivation
The benchmark has RF / LinearSVM / DecisionTree / GNB / MarkovPruning / Transformers.
No sequence model other than the transformers. BiLSTM is the standard comparator.

### Implementation
**New file:** `src/bilstm.py`

```python
class BiLSTMClassifier(nn.Module):
    # 2-layer BiLSTM, hidden=128 per direction (total=256)
    # mean-pool over non-pad positions
    # linear head → num_classes logits
```

Training: reuse `src/train.py::train_game_mal` pattern — same AdamW + cosine LR +
class-weighted CE + early-stop on macro-F1. Match parameter count to GAME-Mal (~515K).

**Wire into `run_experiments.py`** — add to the fold loop so it runs under the same
stratified splits and produces a row in `results/results_summary.csv`.

---

## 3. Architecture Capacity Check (Conditional)

**Only run if seq_len sweep doesn't push GAME-Mal macro-F1 above 0.900.**

Grid (4 combos, first fold only for speed, confirm best with 3 folds):
- `n_layers ∈ {2, 3}` × `d_model ∈ {128, 192}`

Output: `results/arch_sweep.csv`

---

## 4. Deletion Test (Explainability Faithfulness)

### Motivation
The gate-activation rankings in `results/top_apis_per_family.json` are the central
explainability claim. A deletion test is the minimal falsifiable check: if masking the
top-k gate-ranked tokens doesn't reduce the model's confidence in the true class,
the gate scores don't actually mark important tokens.

### Implementation
**New file:** `scripts/run_deletion_test.py`

```
load model from results/models/game_mal_best.pt
load gate rankings from results/top_apis_per_family.json
for each family:
    sample 30 test examples (by true label)
    for k in [5, 10, 20]:
        mask top-k gate tokens → replace with <PAD> token id (0)
        run inference on original + masked
        record: prob_before, prob_after, delta
save results/deletion_test.json
```

Report honestly. A large drop (>10 pp) confirms signal. A small drop is a real finding
and should be disclosed as a limitation of the gate attribution method.

---

## 5. API Semantic Grouping

No code needed. Hand-label `results/top_apis_per_family.json` (top-15 APIs per family)
into buckets: `network`, `filesystem`, `telephony_sms`, `reflection_obfuscation`,
`crypto`, `other`.

Store as `results/api_semantic_groups.json`:
```json
{
  "Airpush": {
    "REFL:UNK._activity_pause": "reflection_obfuscation",
    "java.net.URL.openConnection": "network",
    ...
  },
  ...
}
```

---

## Execution Order

```
1.  git checkout main   (or feature/model-finalisation)
2.  python3 scripts/run_seq_len_sweep.py        # ~2.5h
3.  python3 scripts/run_seq_len_sweep.py --quick-check   # verify wiring first
    → inspect results/seq_len_sweep_summary.csv; pick best (max_len, trunc)
4.  Add src/bilstm.py; update run_experiments.py
5.  python3 run_experiments.py --skip-game-mal  # only train BiLSTM + update CSV
    (or add a --only-bilstm flag)
6.  python3 scripts/run_deletion_test.py        # ~10 min, uses saved checkpoint
7.  Hand-label results/api_semantic_groups.json
8.  Update memory/MEMORY.md with all final numbers
9.  git add -p; git commit; git push
```

---

## Critical Files

| File | Role | Status |
|---|---|---|
| `src/preprocessing.py::pad_sequences` | current head-truncation | needs `pad_with_truncation` variant |
| `src/train.py::train_game_mal` | training loop | reuse as-is |
| `src/model.py::GAMEMal` | main model | reuse as-is |
| `run_experiments.py` | fold loop | add BiLSTM here |
| `results/models/game_mal_best.pt` | saved checkpoint | used by deletion test |
| `results/top_apis_per_family.json` | gate rankings | used by deletion test |
| `results/results_summary.csv` | master benchmark | add BiLSTM row |

---

## Constraints

- No paper edits until model work is complete.
- All new results must reproduce under `python3 scripts/<script>.py` with fixed seed.
- Commit results CSVs + scripts together; never commit partial/overwritten results.
- Update `memory/MEMORY.md` before ending any session.

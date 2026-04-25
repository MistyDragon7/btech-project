# Continuation Prompt — GAME-Mal Project

Paste the text below into a new Claude Code session opened in the project directory
(`/Users/aravjain/PERSONAL/btech-project`).

---

## Prompt to paste

You are continuing a B.Tech research project called **GAME-Mal** — a gated-attention
transformer for Android malware family classification. The full project state is
documented in the repo. Read these files first before doing anything:

1. `MEMORY.md` (or `.claude/projects/…/memory/MEMORY.md`) — full project state, all results, all outstanding TODOs
2. `TODO.md` — prioritised task list with definition of done
3. `PLAN.md` — implementation plan with code structure and execution order

**Goal:** Complete the project end-to-end. Paper writing is deferred.  
**Do NOT edit any paper files** (`paper/` directory) until explicitly asked.

### What has already been done (do not redo)
- Core model (`src/model.py` — G1 gated transformer, `use_gate` flag)
- Training loop (`src/train.py` — AdamW + cosine LR + early-stop on macro-F1)
- Reflection-aware API preprocessing (`src/preprocessing.py`)
- Markov baseline + sweep (`src/markov.py`, `src/baselines.py`, `scripts/run_markov_sweep.py`)
- 3-fold benchmark: RF / LinearSVM / DecisionTree / GNB / MarkovPruning / GAME-Mal
- Gate vs plain-transformer ablation (matched prep, results in `results/gated_matched_summary.json`)
- Gate explainability (`results/top_apis_per_family.json`)
- Attribution comparison (`results/attribution_comparison.json`)

### What still needs to be done (in order)

**1. Seq-len + truncation sweep** — create `scripts/run_seq_len_sweep.py`  
- IMPORTANT: `src/preprocessing.py::pad_sequences` does HEAD truncation (`seq[:max_len]`).  
  The paper claims TAIL truncation (last N calls). This discrepancy must be resolved.  
- Grid: `max_seq_len ∈ {256, 512, 768}` × `truncation ∈ {head, tail}` = 6 configs  
- Same architecture + training recipe as main run (d_model=128, n_layers=2, etc.)  
- Same 3 folds + len≥30 filter (8,085 samples)  
- Output: `results/seq_len_sweep.csv`, `results/seq_len_sweep_summary.csv`  
- Pick best config by mean macro-F1; freeze it  

**2. BiLSTM baseline** — create `src/bilstm.py`, wire into `run_experiments.py`  
- 2-layer BiLSTM, d_model=128 per direction, mean-pool over non-pad, class-weighted CE  
- Same AdamW + cosine LR + early-stop recipe  
- Add row to `results/results_summary.csv`  

**3. Architecture capacity check** (only if F1 < 0.900 after step 1)  
- `n_layers ∈ {2,3}` × `d_model ∈ {128,192}` = 4 combos  

**4. Deletion test** — create `scripts/run_deletion_test.py`  
- Load checkpoint from `results/models/game_mal_best.pt`  
- Mask top-k gate tokens (k=5,10,20) per sample → measure prob drop for true class  
- Output: `results/deletion_test.json`  
- Report honestly even if delta is small  

**5. API semantic groups** — hand-label `results/api_semantic_groups.json`  
- Label top-15 APIs from `results/top_apis_per_family.json`  
- Buckets: network / filesystem / telephony_sms / reflection_obfuscation / crypto / other  

### Important notes
- Always use `python3` (not `python` — that's Python 2.7 on this machine)
- PyTorch MPS backend available (Apple Silicon); DEVICE auto-selects mps→cuda→cpu
- All scripts must be runnable with `python3 scripts/<name>.py`
- Current branch: `feature/markov-sweep`; new model work goes on `main` or a new `feature/model-finalisation` branch
- Commit + push results CSVs + scripts after each completed step
- Update `memory/MEMORY.md` (in `.claude/projects/…/memory/`) before ending the session

### Key result files to check on startup
```
results/results_summary.csv          ← master benchmark table
results/gated_matched_summary.json   ← matched ablation numbers
results/markov_best.json             ← best Markov config
results/top_apis_per_family.json     ← gate rankings for deletion test
results/models/game_mal_best.pt      ← saved checkpoint
```

Start by reading the three files listed above, then execute TODO items 1–5 in order.

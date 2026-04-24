# Markov baseline sweep — report

## Best operating point
- `min_support = 0.0001`
- `min_confidence = 0.8`
- `class_weights = uniform`
- Accuracy: **0.8290 ± 0.0257**
- Macro F1: **0.7250 ± 0.0272**
- Macro AUROC: **0.9477 ± 0.0080**
- Rules kept (fold mean): 3191

## Baseline as currently shipped
- Current config `(0.0005, 0.30, uniform)`: acc=0.8289 F1=0.7092 AUC=0.9421

## Reference points on the same 3 folds
- GAME-Mal (gated, main run):   acc=0.9405  F1=0.8864  AUC=0.9841
- Random Forest (rule feats):   acc=0.9500  F1=0.8930  AUC=0.9930

## Sensitivity of macro-F1 by axis

### class_weights
| class_weights | min | max | mean |
|---|---|---|---|
| inverse | 0.2638 | 0.6147 | 0.3739 |
| prior | 0.2845 | 0.7207 | 0.4497 |
| uniform | 0.5249 | 0.7250 | 0.6539 |

### min_support
| min_support | min | max | mean |
|---|---|---|---|
| 0.0001 | 0.2969 | 0.7250 | 0.5331 |
| 0.0005 | 0.2763 | 0.7092 | 0.5198 |
| 0.001 | 0.2743 | 0.6886 | 0.5054 |
| 0.005 | 0.2683 | 0.6410 | 0.4637 |
| 0.01 | 0.2638 | 0.6191 | 0.4406 |

### min_confidence
| min_confidence | min | max | mean |
|---|---|---|---|
| 0.1 | 0.2638 | 0.7024 | 0.4117 |
| 0.2 | 0.2638 | 0.7024 | 0.4117 |
| 0.3 | 0.2729 | 0.7169 | 0.4242 |
| 0.4 | 0.2891 | 0.6965 | 0.4354 |
| 0.5 | 0.3093 | 0.7106 | 0.4688 |
| 0.6 | 0.3446 | 0.6968 | 0.4957 |
| 0.7 | 0.3639 | 0.7235 | 0.5717 |
| 0.8 | 0.4375 | 0.7250 | 0.5916 |
| 0.9 | 0.5249 | 0.7210 | 0.6218 |

## Artifacts
- `results/markov_sweep.csv` — per-fold grid
- `results/markov_sweep_summary.csv` — fold-aggregate grid
- `results/markov_best.json` — best triple
- `results/figures/markov_surface.png` — F1 surface (supp × conf, one panel per weight)
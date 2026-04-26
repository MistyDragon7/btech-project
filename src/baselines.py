"""
Baseline classifiers for comparison:
1. Random Forest on Markov rule features
2. Linear SVM on Markov rule features
3. Decision Trees on Markov rule features
4. Gaussian Naive Bayes on Markov rule features
5. Markov + Pruning classifier (reproducing base paper approach)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


# ── Metrics computation ─────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """Compute all evaluation metrics matching the base paper."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "f_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # AUC (needs probability scores)
    if y_score is not None and num_classes >= 2:
        try:
            if num_classes == 2:
                auc = roc_auc_score(y_true, y_score[:, 1])
            else:
                y_bin = label_binarize(y_true, classes=list(range(num_classes)))
                auc = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")
            metrics["auc"] = auc
        except ValueError:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    family_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-class metrics for the paper's per-family table."""
    result = {}
    for i, name in enumerate(family_names):
        tp = np.sum((y_true == i) & (y_pred == i))
        tn = np.sum((y_true != i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))

        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        # NOTE: this is the macro-averaged balanced accuracy of a one-vs-rest
        # split, NOT a true AUC. Earlier versions of this function mis-labelled
        # the field as "auc"; we keep that key for backward-compatibility but
        # also expose the correct name "balanced_accuracy".
        bal_acc = (sens + spec) / 2

        result[name] = {
            "accuracy": acc, "sensitivity": sens, "specificity": spec,
            "precision": prec,
            "balanced_accuracy": bal_acc,
            "auc": bal_acc,  # deprecated alias — see note above
            "f_score": f1,
        }
    return result


# ── Baseline models ─────────────────────────────────────────────────────────

BASELINE_MODELS = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
    "LinearSVM": lambda: LinearSVC(
        max_iter=5000, random_state=42, dual=False
    ),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
    "GaussianNB": lambda: GaussianNB(),
}


def train_evaluate_baseline(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Train and evaluate a single baseline model."""
    model_fn = BASELINE_MODELS[name]
    model = model_fn()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get probability scores if available
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_test)
        if dec.ndim == 1:
            y_score = np.column_stack([-dec, dec])
        else:
            y_score = dec

    metrics = compute_metrics(y_true=y_test, y_pred=y_pred,
                              y_score=y_score, num_classes=num_classes)
    logger.info("%s: Acc=%.4f F1=%.4f AUC=%.4f",
                name, metrics["accuracy"], metrics["f_score"], metrics["auc"])

    return metrics, y_pred


# ── Markov + Pruning classifier (base paper) ───────────────────────────────

class MarkovPruningClassifier:
    """
    Reproduces the associative-rules classifier from D'Angelo et al. (2023).
    Uses support/confidence-based pruning and softmax ranking.
    """

    def __init__(self, min_support=0.001, min_confidence=0.3, class_weights=None):
        """
        class_weights: one of
            None / "uniform"  -> w_c = 1 for every c (current behaviour)
            "prior"           -> w_c = N_c / N   (class frequency; rewards majority)
            "inverse"         -> w_c = N / (|C| * N_c) (inverse frequency; rewards minority)
            np.ndarray of shape (num_classes,) -> used as-is.
        Broadcast element-wise into rho before softmax.
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.class_weights_spec = class_weights
        self.class_support = None
        self.class_confidence = None
        self.class_weights = None  # resolved at fit() time
        self.selected_rules = None
        self.num_classes = None

    def _resolve_class_weights(self, labels, num_classes):
        import numpy as _np
        spec = self.class_weights_spec
        if spec is None or spec == "uniform":
            return _np.ones(num_classes, dtype=_np.float64)
        if isinstance(spec, str):
            y = _np.asarray(labels)
            counts = _np.array([max((y == c).sum(), 1) for c in range(num_classes)],
                               dtype=_np.float64)
            N = counts.sum()
            if spec == "prior":
                return counts / N
            if spec == "inverse":
                return N / (num_classes * counts)
            raise ValueError(f"Unknown class_weights spec: {spec!r}")
        # Array-like
        arr = _np.asarray(spec, dtype=_np.float64)
        if arr.shape != (num_classes,):
            raise ValueError(
                f"class_weights array must be shape ({num_classes},), got {arr.shape}"
            )
        return arr

    def fit(self, class_graphs, num_classes, labels=None):
        from src.markov import compute_support_confidence, prune_rules

        self.num_classes = num_classes
        support, confidence = compute_support_confidence(class_graphs, num_classes)
        self.selected_rules = prune_rules(support, confidence,
                                          self.min_support, self.min_confidence)
        self.class_support = {r: support[r] for r in self.selected_rules}
        self.class_confidence = {r: confidence[r] for r in self.selected_rules}
        # Resolve class weights. "prior"/"inverse" need label counts.
        if labels is None and isinstance(self.class_weights_spec, str) \
                and self.class_weights_spec in ("prior", "inverse"):
            raise ValueError(
                f"class_weights={self.class_weights_spec!r} requires labels= in fit()"
            )
        self.class_weights = self._resolve_class_weights(labels, num_classes)
        # Majority-class fallback: when a test sample contains no selected
        # rule, predict() previously returned 0 (Airpush), silently biasing
        # the baseline toward the majority. We now record the training-set
        # majority class and use it as a principled fallback.
        if labels is not None:
            counts = np.bincount(np.asarray(labels), minlength=num_classes)
            self._fallback_class = int(counts.argmax())
        else:
            self._fallback_class = 0

    def predict(self, encoded_sequences, max_spacing=10):
        from src.markov import extract_rules

        # Build a fast lookup: rule -> (class_confidence vector)
        # Only selected rules contribute.
        rule_set = set(self.selected_rules)
        predictions = []
        for seq in encoded_sequences:
            sample_rules = extract_rules(seq, max_spacing)
            seq_len = max(len(seq), 1)
            # Compute rank rho for each class (Eq. 6 from base paper)
            # rho[c] = sum over matching rules: (count/seq_len) * confidence(rule, c)
            rho = np.zeros(self.num_classes)
            for rule, count in sample_rules.items():
                if rule not in rule_set:
                    continue
                sigma_norm = count / seq_len
                rho += sigma_norm * self.class_confidence[rule] * self.class_weights

            # Softmax classification (Eq. 7)
            if rho.sum() == 0:
                # No selected rule fired in this sample — fall back to the
                # training-set majority class (set in fit()).
                predictions.append(getattr(self, "_fallback_class", 0))
                continue
            rho_exp = np.exp(rho - rho.max())  # numerical stability
            probs = rho_exp / rho_exp.sum()
            predictions.append(int(np.argmax(probs)))

        return np.array(predictions)

    def predict_proba(self, encoded_sequences, max_spacing=10):
        from src.markov import extract_rules

        rule_set = set(self.selected_rules)
        all_probs = []
        for seq in encoded_sequences:
            sample_rules = extract_rules(seq, max_spacing)
            seq_len = max(len(seq), 1)
            rho = np.zeros(self.num_classes)
            for rule, count in sample_rules.items():
                if rule not in rule_set:
                    continue
                sigma_norm = count / seq_len
                rho += sigma_norm * self.class_confidence[rule] * self.class_weights

            if rho.sum() == 0:
                probs = np.ones(self.num_classes) / self.num_classes
            else:
                rho_exp = np.exp(rho - rho.max())
                probs = rho_exp / rho_exp.sum()
            all_probs.append(probs)

        return np.array(all_probs)

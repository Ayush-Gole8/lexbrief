"""M0: TF-IDF (word 1-2 grams) + class-balanced logistic regression, C selected on val."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from lexbrief.labels import FINE_LABELS

logger = logging.getLogger(__name__)

MODEL_FILE = "tfidf_lr.joblib"


@dataclass
class TfidfBaseline:
    """Vectoriser + classifier pair with probability outputs over all 13 fine labels."""

    vectorizer: TfidfVectorizer
    clf: LogisticRegression
    c_scores: dict[float, float] = field(default_factory=dict)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probabilities ``[n, 13]`` in FINE_LABELS order (unseen classes get 0)."""
        p = self.clf.predict_proba(self.vectorizer.transform(texts))
        out = np.zeros((len(texts), len(FINE_LABELS)), dtype=np.float32)
        out[:, self.clf.classes_] = p
        return out

    def save(self, out_dir: str | Path) -> Path:
        path = Path(out_dir) / MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(run_dir: str | Path) -> TfidfBaseline:
        return joblib.load(Path(run_dir) / MODEL_FILE)


def train_tfidf(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    c_grid: list[float],
    ngram_max: int = 2,
    max_features: int = 50000,
    min_df: int = 2,
    max_iter: int = 2000,
    seed: int = 42,
) -> TfidfBaseline:
    """Fit TF-IDF on train, pick C by val macro-F1 over the 13 fine labels."""
    vec = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=True,
    )
    x_tr = vec.fit_transform(train_texts)
    x_va = vec.transform(val_texts)
    labels = list(range(len(FINE_LABELS)))
    best: tuple[float, float, LogisticRegression] | None = None
    scores: dict[float, float] = {}
    for c in c_grid:
        clf = LogisticRegression(
            C=c, class_weight="balanced", max_iter=max_iter, random_state=seed
        )
        clf.fit(x_tr, train_labels)
        f1 = f1_score(val_labels, clf.predict(x_va), labels=labels, average="macro", zero_division=0)
        scores[c] = float(f1)
        logger.info("M0 C=%g val macro-F1=%.4f", c, f1)
        if best is None or f1 > best[0]:
            best = (f1, c, clf)
    assert best is not None, "empty C grid"
    logger.info("M0 best C=%g (val macro-F1=%.4f)", best[1], best[0])
    return TfidfBaseline(vectorizer=vec, clf=best[2], c_scores=scores)

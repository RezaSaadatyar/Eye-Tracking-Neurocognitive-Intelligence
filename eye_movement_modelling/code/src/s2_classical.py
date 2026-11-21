# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Prespecified classical task classifiers; training uses train records only."""
from __future__ import annotations


def classical_candidates(seed: int = 17) -> dict:
    """Return fixed, compact baselines; no Test-driven hyperparameter tuning."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, RandomForestClassifier,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def preprocessing(scale: bool):
        steps = [("impute", SimpleImputer(
            strategy="median", keep_empty_features=True,
        ))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return steps

    return {
        "class_prior": Pipeline(
            preprocessing(False) + [
                ("model", DummyClassifier(strategy="prior")),
            ],
        ),
        "logistic_regression": Pipeline(
            preprocessing(True) + [
                ("model", LogisticRegression(
                    max_iter=800, class_weight="balanced", random_state=seed,
                )),
            ],
        ),
        "random_forest": Pipeline(
            preprocessing(False) + [
                ("model", RandomForestClassifier(
                    n_estimators=120, max_depth=8, min_samples_leaf=2,
                    class_weight="balanced", n_jobs=-1, random_state=seed,
                )),
            ],
        ),
        "gradient_boosting": Pipeline(
            preprocessing(False) + [
                ("model", HistGradientBoostingClassifier(
                    max_iter=65, max_leaf_nodes=7, min_samples_leaf=2,
                    learning_rate=0.05,
                    l2_regularization=1.0, random_state=seed,
                )),
            ],
        ),
    }

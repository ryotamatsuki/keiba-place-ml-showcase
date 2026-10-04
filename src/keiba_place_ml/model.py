"""Probability-model pipelines used by the public showcase."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RELATIVE_FEATURES,
    add_relative_ability_features,
    engineer_features,
    validate_training_frame,
)

RANDOM_SEED = 20261003


def make_logistic_pipeline(*, c_value: float = 1.0) -> Pipeline:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_xgb_relative_pipeline() -> Pipeline:
    """XGBoost candidate with within-race relative-ability features."""
    from xgboost import XGBClassifier

    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, [*NUMERIC_FEATURES, *RELATIVE_FEATURES]),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        n_jobs=2,
        random_state=RANDOM_SEED,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=20,
        reg_lambda=5.0,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def fit_logistic(train: pd.DataFrame, *, c_value: float = 1.0) -> tuple[Pipeline, float]:
    validate_training_frame(train)
    prior_mean = float(train["top3_label"].mean())
    engineered = engineer_features(train, prior_mean=prior_mean)
    model = make_logistic_pipeline(c_value=c_value)
    model.fit(engineered, train["top3_label"].astype(int))
    return model, prior_mean


def predict_logistic(model: Pipeline, frame: pd.DataFrame, *, prior_mean: float) -> np.ndarray:
    engineered = engineer_features(frame, prior_mean=prior_mean)
    p = np.asarray(model.predict_proba(engineered)[:, 1], dtype=float)
    return np.clip(p, 0.0, 1.0)


def fit_xgb_relative(train: pd.DataFrame) -> tuple[Pipeline, float]:
    validate_training_frame(train)
    prior_mean = float(train["top3_label"].mean())
    engineered = add_relative_ability_features(train, prior_mean=prior_mean)
    model = make_xgb_relative_pipeline()
    model.fit(engineered, train["top3_label"].astype(int))
    return model, prior_mean


def predict_xgb_relative(
    model: Pipeline, frame: pd.DataFrame, *, prior_mean: float
) -> np.ndarray:
    engineered = add_relative_ability_features(frame, prior_mean=prior_mean)
    p = np.asarray(model.predict_proba(engineered)[:, 1], dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Model produced non-finite probabilities")
    return np.clip(p, 0.0, 1.0)

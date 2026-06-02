from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from etaslip.modeling.dataset import DatasetSpec


def make_preprocessor(spec: DatasetSpec) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), list(spec.numeric_features)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                list(spec.categorical_features),
            ),
        ],
        remainder="drop",
    )


def train_logreg(df_train: pd.DataFrame, spec: DatasetSpec) -> Pipeline:
    X = df_train[list(spec.numeric_features) + list(spec.categorical_features)]
    y = df_train[spec.target_col].astype(int)

    pipe = Pipeline(
        steps=[
            ("prep", make_preprocessor(spec)),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=50000,
                    class_weight="balanced",
                    tol=2e-3,
                    C=0.25,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    # print(pipe.named_steps["clf"])
    return pipe


def train_xgb(
    df_train: pd.DataFrame, spec: DatasetSpec, *, random_state: int = 42
) -> Pipeline:
    from xgboost import XGBClassifier

    X = df_train[list(spec.numeric_features) + list(spec.categorical_features)]
    y = df_train[spec.target_col].astype(int)

    pipe = Pipeline(
        steps=[
            ("prep", make_preprocessor(spec)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=450,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.78,
                    colsample_bytree=0.78,
                    min_child_weight=18,
                    gamma=0.8,
                    reg_lambda=10.0,
                    reg_alpha=0.5,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    random_state=random_state,
                    n_jobs=-1,
                    scale_pos_weight=1.0,
                    tree_method="hist",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def predict_proba(pipe: Pipeline, df: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    X = df[list(spec.numeric_features) + list(spec.categorical_features)]
    return pipe.predict_proba(X)[:, 1]

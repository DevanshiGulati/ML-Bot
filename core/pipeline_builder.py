"""
Pipeline Builder
----------------
Builds a leak-proof sklearn Pipeline:
  ColumnTransformer (preprocessor) + model

Preprocessing is always applied inside the pipeline — so it
automatically runs correctly during both training and inference.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline      import Pipeline
from sklearn.compose       import ColumnTransformer
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def _make_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(model, numeric_cols: list, categorical_cols: list) -> Pipeline:
    """
    Returns a full sklearn Pipeline: preprocessor → model.

    Both training (fit) and inference (predict) use raw DataFrames —
    preprocessing happens automatically inside the pipeline.
    """
    return Pipeline([
        ("preprocessor", _make_preprocessor(numeric_cols, categorical_cols)),
        ("model",        model),
    ])


def infer_column_types(df: pd.DataFrame, target_col: str) -> tuple[list, list]:
    """Split feature columns into numeric and categorical lists."""
    feats = df.drop(columns=[target_col])
    return (
        feats.select_dtypes(include=[np.number]).columns.tolist(),
        feats.select_dtypes(exclude=[np.number]).columns.tolist(),
    )
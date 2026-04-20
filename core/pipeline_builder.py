"""
Pipeline Builder
----------------
Constructs a full sklearn Pipeline:
  ColumnTransformer (preprocessing) + model

The preprocessor handles:
  - Numeric  : SimpleImputer(median) → StandardScaler
  - Categorical: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
"""

import pandas as pd
import numpy as np
from sklearn.pipeline       import Pipeline
from sklearn.compose        import ColumnTransformer
from sklearn.impute         import SimpleImputer
from sklearn.preprocessing  import StandardScaler, OneHotEncoder


def _make_preprocessor(
    numeric_cols:     list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Build the ColumnTransformer preprocessing step."""

    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("numeric", numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",           # drop any unlisted columns safely
    )
    return preprocessor


def build_pipeline(
    model,
    numeric_cols:     list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """
    Returns a full sklearn Pipeline:
      preprocessor → model

    This pipeline can be:
      - Fitted with .fit(X, y)
      - Used for inference with .predict(X_new)
      Both X inputs must be raw DataFrames (NOT pre-processed).
    """
    preprocessor = _make_preprocessor(numeric_cols, categorical_cols)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        model),
    ])
    return pipeline


def infer_column_types(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[list[str], list[str]]:
    """
    Automatically split feature columns into numeric and categorical lists,
    excluding the target column.
    """
    features         = df.drop(columns=[target_col])
    numeric_cols     = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols

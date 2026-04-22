"""
Metadata Extractor
------------------
Profiles the uploaded DataFrame and returns a rich dict
that the Decision Agent uses to plan the pipeline.
"""

import numpy as np
import pandas as pd


def extract_metadata(df: pd.DataFrame, target_col: str) -> dict:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    feats  = df.drop(columns=[target_col])
    target = df[target_col]

    numeric_cols     = feats.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feats.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_counts = df.isnull().sum()
    cols_with_missing = {
        k: int(v) for k, v in missing_counts[missing_counts > 0].items()
    }

    target_unique = int(target.nunique())
    target_dist   = {
        str(k): round(float(v), 4)
        for k, v in target.value_counts(normalize=True).head(5).items()
    }

    # Class imbalance: minority < 10%
    imbalanced = False
    if target_unique <= 20:
        counts = target.value_counts(normalize=True)
        imbalanced = bool(counts.min() < 0.10)

    # High correlation pairs
    high_corr: list[dict] = []
    if len(numeric_cols) >= 2:
        corr = feats[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = (
            upper.stack()
                 .reset_index()
                 .rename(columns={"level_0": "col1", "level_1": "col2", 0: "corr"})
        )
        high_corr = pairs[pairs["corr"] > 0.90].to_dict(orient="records")[:10]

    return {
        "num_rows":                 int(len(df)),
        "num_features":             int(feats.shape[1]),
        "num_numeric_features":     len(numeric_cols),
        "num_categorical_features": len(categorical_cols),
        "numeric_columns":          numeric_cols,
        "categorical_columns":      categorical_cols,
        "target_column":            target_col,
        "target_dtype":             str(target.dtype),
        "target_unique_values":     target_unique,
        "target_distribution":      target_dist,
        "class_imbalance":          imbalanced,
        "missing_value_columns":    cols_with_missing,
        "has_missing_values":       bool(cols_with_missing),
        "high_correlation_pairs":   high_corr,
    }
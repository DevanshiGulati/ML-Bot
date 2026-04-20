"""
Metadata Extractor
------------------
Profiles an uploaded DataFrame and returns a rich metadata dict
that the Decision Agent uses to make its pipeline decisions.
"""

import pandas as pd
import numpy as np


def extract_metadata(df: pd.DataFrame, target_col: str) -> dict:
    """
    Extract comprehensive metadata from a DataFrame.

    Parameters
    ----------
    df         : raw uploaded DataFrame (target column still present)
    target_col : name of the column to predict

    Returns
    -------
    dict with all statistics the Decision Agent needs
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    feature_df = df.drop(columns=[target_col])
    target     = df[target_col]

    # ── Column type breakdown ────────────────────────────────────────────────
    numeric_cols     = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # ── Missing values ───────────────────────────────────────────────────────
    missing_counts  = df.isnull().sum()
    missing_pct     = (missing_counts / len(df) * 100).round(2)
    cols_with_missing = missing_counts[missing_counts > 0].to_dict()

    # ── Target analysis ──────────────────────────────────────────────────────
    target_unique   = int(target.nunique())
    target_dtype    = str(target.dtype)
    target_dist     = target.value_counts(normalize=True).head(5).to_dict()
    # Convert keys to str for JSON serialisation
    target_dist     = {str(k): round(float(v), 4) for k, v in target_dist.items()}

    # Class imbalance flag (classification)
    imbalanced = False
    if target_unique <= 20:
        counts = target.value_counts(normalize=True)
        if counts.min() < 0.10:        # minority class < 10 %
            imbalanced = True

    # ── Correlation (numeric only) ───────────────────────────────────────────
    high_corr_pairs: list[dict] = []
    if len(numeric_cols) >= 2:
        corr_matrix = feature_df[numeric_cols].corr().abs()
        upper       = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        pairs = (
            upper.stack()
                 .reset_index()
                 .rename(columns={"level_0": "col1", "level_1": "col2", 0: "corr"})
        )
        high = pairs[pairs["corr"] > 0.90]
        high_corr_pairs = high.to_dict(orient="records")

    # ── Feature stats summary ────────────────────────────────────────────────
    numeric_stats: dict = {}
    if numeric_cols:
        desc = feature_df[numeric_cols].describe().T
        numeric_stats = desc[["mean", "std", "min", "max"]].round(4).to_dict(orient="index")

    metadata = {
        # Dataset shape
        "num_rows":            int(len(df)),
        "num_features":        int(feature_df.shape[1]),
        "num_numeric_features":    len(numeric_cols),
        "num_categorical_features": len(categorical_cols),
        "numeric_columns":     numeric_cols,
        "categorical_columns": categorical_cols,

        # Target
        "target_column":        target_col,
        "target_dtype":         target_dtype,
        "target_unique_values": target_unique,
        "target_distribution":  target_dist,
        "class_imbalance":      imbalanced,

        # Data quality
        "missing_value_columns": cols_with_missing,
        "has_missing_values":    bool(cols_with_missing),
        "high_correlation_pairs": high_corr_pairs[:10],  # cap for prompt length

        # Feature stats
        "numeric_feature_stats": numeric_stats,
    }

    return metadata

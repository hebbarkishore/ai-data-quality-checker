from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _prep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan)
    num = num.fillna(num.median(numeric_only=True))
    return num


def isolation_forest_outliers(df: pd.DataFrame, contamination: float = 0.03) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"flags": [], "n_outliers": 0, "outlier_indices": []}
    X = _prep_numeric(df)
    model = IsolationForest(
        n_estimators=200,
        contamination=max(min(contamination, 0.49), 0.0),
        random_state=42,
    )
    model.fit(X)
    scores = model.decision_function(X)
    labels = model.predict(X)  # -1 outlier, 1 inlier
    flags = (labels == -1).tolist()
    idx = np.where(flags)[0].tolist()
    return {
        "flags": flags,
        "n_outliers": int(sum(flags)),
        "outlier_indices": idx,
        "scores": scores.tolist(),
    }


def iqr_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"flags": [], "n_outliers": 0, "outlier_indices": []}
    X = _prep_numeric(df)
    flags = [False] * len(X)
    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_flags = (X[col] < lower) | (X[col] > upper)
        flags = [f or bool(c) for f, c in zip(flags, col_flags.to_list())]
    idx = list(np.where(flags)[0])
    return {
        "flags": flags,
        "n_outliers": int(sum(flags)),
        "outlier_indices": idx,
    }

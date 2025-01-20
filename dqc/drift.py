from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def _psi(actual: np.ndarray, expected: np.ndarray) -> float:
    # Avoid zeros
    actual = np.where(actual == 0, 1e-8, actual)
    expected = np.where(expected == 0, 1e-8, expected)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_psi(current: pd.Series, reference: pd.Series, bins: int = 10) -> float:
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return float("nan")
    # Use reference quantiles as bin edges
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, qs))
    if len(edges) < 3:
        return float("nan")
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_dist = ref_counts / max(ref_counts.sum(), 1)
    cur_dist = cur_counts / max(cur_counts.sum(), 1)
    return _psi(cur_dist, ref_dist)


def _categorical_jsd(current: pd.Series, reference: pd.Series) -> float:
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    if ref.empty or cur.empty:
        return float("nan")
    cats = sorted(set(ref.unique()).union(set(cur.unique())))
    ref_counts = ref.value_counts().reindex(cats).fillna(0).to_numpy()
    cur_counts = cur.value_counts().reindex(cats).fillna(0).to_numpy()
    ref_dist = ref_counts / max(ref_counts.sum(), 1)
    cur_dist = cur_counts / max(cur_counts.sum(), 1)
    return float(jensenshannon(ref_dist, cur_dist))


def drift_report(current_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict[str, Any]:
    psi_rows: List[dict] = []
    jsd_rows: List[dict] = []

    for col in current_df.columns:
        if col not in reference_df.columns:
            continue
        cur = current_df[col]
        ref = reference_df[col]
        if pd.api.types.is_numeric_dtype(cur) and pd.api.types.is_numeric_dtype(ref):
            psi = _numeric_psi(cur, ref)
            psi_rows.append({"column": col, "psi": None if np.isnan(psi) else round(psi, 4)})
        else:
            jsd = _categorical_jsd(cur, ref)
            jsd_rows.append({"column": col, "jsd": None if np.isnan(jsd) else round(jsd, 4)})

    psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False) if psi_rows else pd.DataFrame()
    jsd_df = pd.DataFrame(jsd_rows).sort_values("jsd", ascending=False) if jsd_rows else pd.DataFrame()

    flags = {
        "psi_numeric_ge_0.25": psi_df[psi_df["psi"] >= 0.25]["column"].tolist() if not psi_df.empty else [],
        "jsd_categorical_ge_0.20": jsd_df[jsd_df["jsd"] >= 0.20]["column"].tolist() if not jsd_df.empty else [],
    }

    return {
        "psi_numeric": psi_df,
        "jsd_categorical": jsd_df,
        "flags": flags,
    }

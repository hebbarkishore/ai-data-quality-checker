from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def infer_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return a schema table with dtype and a sample non-null value per column."""
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_val = next((v for v in df[col].dropna().head(5).tolist()), None)
        rows.append({"column": col, "dtype": dtype, "sample_value": sample_val})
    return pd.DataFrame(rows)


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    res = []
    for col in df.columns:
        miss = df[col].isna().sum()
        pct = (miss / total * 100.0) if total else 0.0
        res.append({"column": col, "missing": miss, "missing_pct": round(pct, 2)})
    out = pd.DataFrame(res).sort_values("missing_pct", ascending=False)
    return out


def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    dup_mask = df.duplicated(keep=False)
    n_dups = int(dup_mask.sum())
    examples = []
    if n_dups > 0:
        examples = (
            df[dup_mask]
            .head(10)
            .to_dict(orient="records")
        )
    return {"count": n_dups, "examples": examples}


def cardinality_report(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        ratio = nunique / max(len(df), 1)
        rows.append({
            "column": col,
            "nunique": int(nunique),
            "unique_ratio": round(ratio, 4),
        })
    return pd.DataFrame(rows).sort_values("unique_ratio", ascending=False)


def constant_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1]


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T
    desc["missing"] = num.isna().sum()
    desc["missing_pct"] = (desc["missing"] / max(len(num), 1) * 100.0).round(2)
    return desc.reset_index().rename(columns={"index": "column"})

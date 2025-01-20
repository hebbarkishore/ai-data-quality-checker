from __future__ import annotations
from typing import Dict, Any
import pandas as pd


def build_report_bundle(
    *,
    current: pd.DataFrame,
    schema: pd.DataFrame,
    missing: pd.DataFrame,
    duplicates: dict,
    cardinality: pd.DataFrame,
    constant_cols: list,
    numeric_summary: pd.DataFrame,
    outlier_result: dict,
    drift_result: dict | None,
    pii_result: pd.DataFrame | None,
) -> Dict[str, Any]:
    return {
        "meta": {"rows": len(current), "cols": len(current.columns)},
        "schema": schema.to_dict(orient="records"),
        "missing": missing.to_dict(orient="records"),
        "duplicates": duplicates,
        "cardinality": cardinality.to_dict(orient="records"),
        "constant_columns": constant_cols,
        "numeric_summary": numeric_summary.to_dict(orient="records"),
        "outliers": outlier_result,
        "drift": None if drift_result is None else {
            "psi_numeric": None if drift_result.get("psi_numeric") is None or drift_result.get("psi_numeric").empty else drift_result["psi_numeric"].to_dict(orient="records"),
            "jsd_categorical": None if drift_result.get("jsd_categorical") is None or drift_result.get("jsd_categorical").empty else drift_result["jsd_categorical"].to_dict(orient="records"),
            "flags": drift_result.get("flags"),
        },
        "pii": None if pii_result is None else pii_result.to_dict(orient="records"),
    }

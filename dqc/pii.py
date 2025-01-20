\
import re
import pandas as pd
from typing import List

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?1\s*[-.]?\s*)?(?:\(?\d{3}\)?\s*[-.]?\s*)?\d{3}\s*[-.]?\s*\d{4})")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def pii_scan(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        series = df[col].dropna().astype(str)
        emails = series.str.contains(EMAIL_RE).sum()
        phones = series.str_contains(PHONE_RE).sum() if hasattr(series, "str_contains") else series.str.contains(PHONE_RE).sum()
        ssn = series.str.contains(SSN_RE).sum()
        cc = series.str.contains(CREDIT_CARD_RE).sum()
        if any([emails, phones, ssn, cc]):
            rows.append({
                "column": col,
                "email_hits": int(emails),
                "phone_hits": int(phones),
                "ssn_hits": int(ssn),
                "credit_card_hits": int(cc),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["column","email_hits","phone_hits","ssn_hits","credit_card_hits"]) 

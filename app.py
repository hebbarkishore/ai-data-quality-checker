import io
import json
from typing import Dict, Any

import pandas as pd
import streamlit as st

from dqc.quality_checks import (
    infer_schema,
    summarize_missing,
    detect_duplicates,
    cardinality_report,
    constant_columns,
    numeric_summary,
)
from dqc.anomaly import isolation_forest_outliers, iqr_outliers
from dqc.drift import drift_report
from dqc.pii import pii_scan
from dqc.report import build_report_bundle

st.set_page_config(page_title="File quality checker", layout="wide")

import pandas as pd
from dqc.quality_checks import summarize_missing, constant_columns


def test_missing_and_constant():
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": ["x", "y", None],
        "c": [5, 5, 5],
    })
    miss = summarize_missing(df)
    assert set(miss["column"]) == {"a", "b", "c"}
    const = constant_columns(df)
    assert const == ["c"]

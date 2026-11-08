
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features available before answer generation."""
    df = df.copy()

    df["has_context"] = (
        df["context"].fillna("").astype(str).str.strip().ne("").astype(int)
    )

    df["question_length"] = (
        df["question"].fillna("").astype(str).str.split().str.len()
    )

    df["context_length"] = (
        df["context"].fillna("").astype(str).str.split().str.len()
    )

    return df

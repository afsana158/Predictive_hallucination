import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic text/context features.

    Features:
    - has_context
    - question_length
    - answer_length
    - context_length
    """

    df = df.copy()

    df["has_context"] = (
        df["context"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    df["question_length"] = (
        df["question"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    df["answer_length"] = (
        df["answer"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    df["context_length"] = (
        df["context"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    return df
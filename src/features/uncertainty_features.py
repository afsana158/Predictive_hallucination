import numpy as np
import pandas as pd


def add_uncertainty_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add uncertainty-related features.

    Features:
        answer_length_normalized
        semantic_uncertainty
        retrieval_uncertainty
        feature_disagreement
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Normalized answer length
    # --------------------------------------------------

    df["answer_length_normalized"] = (
        df["answer_length"]
        /
        (df["question_length"] + 1)
    )


    # --------------------------------------------------
    # 2. Semantic uncertainty
    # --------------------------------------------------
    #
    # Lower question-answer similarity
    # means weaker semantic alignment.
    #
    # For unavailable values, keep NaN.
    # --------------------------------------------------

    df["semantic_uncertainty"] = (
        1.0 -
        df["question_answer_similarity"]
    )


    # --------------------------------------------------
    # 3. Retrieval uncertainty
    # --------------------------------------------------
    #
    # Lower retrieval support means
    # greater retrieval uncertainty.
    # --------------------------------------------------

    df["retrieval_uncertainty"] = (
        1.0 -
        df["retrieval_top1_similarity"]
    )


    # --------------------------------------------------
    # 4. Feature disagreement
    # --------------------------------------------------
    #
    # Compare semantic and retrieval support.
    #
    # Large disagreement means:
    # one signal is strong while the other is weak.
    # --------------------------------------------------

    semantic_score = (
        df["question_answer_similarity"]
    )

    retrieval_score = (
        df["retrieval_top1_similarity"]
    )

    df["feature_disagreement"] = (
        semantic_score -
        retrieval_score
    ).abs()

    return df
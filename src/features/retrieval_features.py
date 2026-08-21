import numpy as np
import pandas as pd
import faiss


def add_retrieval_features(
    df: pd.DataFrame,
    index,
    embedding_model,
    top_k: int = 3,
    batch_size: int = 64
) -> pd.DataFrame:
    """
    Add retrieval-based features using a FAISS index.

    Features:
        retrieval_top1_similarity
        retrieval_top3_mean_similarity
        retrieval_coverage
    """

    df = df.copy()

    questions = (
        df["question"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    query_embeddings = embedding_model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embeddings)

    scores, indices = index.search(
        query_embeddings,
        top_k
    )

    top1_similarity = []
    topk_mean_similarity = []
    retrieval_coverage = []

    for row_scores in scores:

        top1 = float(row_scores[0])
        topk_mean = float(np.mean(row_scores))

        top1_similarity.append(top1)
        topk_mean_similarity.append(topk_mean)

        # First prototype definition:
        # proportion of retrieved documents with
        # positive semantic similarity.
        threshold = 0.80 * row_scores[0]
        coverage = float(
            np.mean(row_scores >= threshold)
        )

        retrieval_coverage.append(coverage)

    df["retrieval_top1_similarity"] = (
        top1_similarity
    )

    df["retrieval_top3_mean_similarity"] = (
        topk_mean_similarity
    )

    df["retrieval_coverage"] = (
        retrieval_coverage
    )

    return df
import numpy as np
import pandas as pd
import faiss


def add_retrieval_features(
    df: pd.DataFrame,
    index,
    embedding_model,
    corpus_size: int,
    top_k: int = 3
) -> pd.DataFrame:

    df = df.copy()

    questions = (
        df["question"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    query_embeddings = embedding_model.encode(
        questions,
        batch_size=64,
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
        mean_topk = float(np.mean(row_scores))

        top1_similarity.append(top1)
        topk_mean_similarity.append(mean_topk)

        # Simple first-version coverage signal.
        # Higher retrieval similarity means stronger
        # semantic coverage of the query by the corpus.
        coverage = float(
            np.mean(row_scores > 0.5)
        )

        retrieval_coverage.append(coverage)

    df["retrieval_top1_similarity"] = top1_similarity
    df["retrieval_top3_mean_similarity"] = (
        topk_mean_similarity
    )
    df["retrieval_coverage"] = retrieval_coverage

    return df
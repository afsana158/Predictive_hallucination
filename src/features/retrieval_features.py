import numpy as np
import pandas as pd
import faiss


RELEVANCE_THRESHOLD = 0.50


def add_retrieval_features(
    df: pd.DataFrame,
    index,
    embedding_model,
    top_k: int = 3,
    batch_size: int = 64,
    exclude_document_ids=None
) -> pd.DataFrame:
    """
    Add FAISS retrieval features.

    Features:
        retrieval_top1_similarity
        retrieval_top3_mean_similarity
        retrieval_coverage

    exclude_document_ids:
        Optional mapping/list used to exclude a known source document
        from retrieval, primarily for HaluEval training examples.
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

    query_embeddings = np.asarray(
        query_embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(query_embeddings)

    # Retrieve one extra result when exclusions are requested.
    search_k = top_k + 1 if exclude_document_ids is not None else top_k

    scores, indices = index.search(
        query_embeddings,
        search_k
    )

    top1_similarity = []
    topk_mean_similarity = []
    retrieval_coverage = []

    for row_idx, (row_scores, row_indices) in enumerate(
        zip(scores, indices)
    ):

        selected_scores = []

        excluded_id = None

        if exclude_document_ids is not None:
            excluded_id = exclude_document_ids[row_idx]

        for score, doc_idx in zip(
            row_scores,
            row_indices
        ):

            if doc_idx < 0:
                continue

            # Skip the source document for leakage prevention.
            if (
                excluded_id is not None
                and str(doc_idx) == str(excluded_id)
            ):
                continue

            selected_scores.append(float(score))

            if len(selected_scores) == top_k:
                break

        # Fallback if fewer than top_k results remain.
        if len(selected_scores) == 0:
            selected_scores = [0.0] * top_k

        elif len(selected_scores) < top_k:
            selected_scores.extend(
                [selected_scores[-1]]
                * (top_k - len(selected_scores))
            )

        top1 = selected_scores[0]
        topk_mean = float(np.mean(selected_scores))

        coverage = float(
            np.mean(
                np.array(selected_scores)
                >= RELEVANCE_THRESHOLD
            )
        )

        top1_similarity.append(top1)
        topk_mean_similarity.append(topk_mean)
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
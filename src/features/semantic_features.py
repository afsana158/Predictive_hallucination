import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def add_semantic_features(
    df: pd.DataFrame,
    batch_size: int = 64
) -> pd.DataFrame:

    df = df.copy()

    questions = (
        df["question"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    answers = (
        df["answer"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    contexts = (
        df["context"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    question_embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True
    )

    answer_embeddings = model.encode(
        answers,
        batch_size=batch_size,
        show_progress_bar=True
    )

    context_embeddings = model.encode(
        contexts,
        batch_size=batch_size,
        show_progress_bar=True
    )

    qa_similarity = []
    qc_similarity = []
    ac_similarity = []

    for i, (q, a, c) in enumerate(
        zip(
            question_embeddings,
            answer_embeddings,
            context_embeddings
        )
    ):

        # Question ↔ Answer
        qa_similarity.append(
            cosine_similarity(q, a)
        )

        # Context-based similarities only make sense
        # when an actual context is available.
        if df.iloc[i]["has_context"] == 0:
            qc_similarity.append(0.0)
            ac_similarity.append(0.0)

        else:
            qc_similarity.append(
                cosine_similarity(q, c)
            )

            ac_similarity.append(
                cosine_similarity(a, c)
            )

    # Add all features AFTER the loop is complete
    df["question_answer_similarity"] = qa_similarity
    df["question_context_similarity"] = qc_similarity
    df["answer_context_similarity"] = ac_similarity

    return df
import pandas as pd


def convert_truthfulqa(df):
    """
    Convert raw TruthfulQA data into our canonical schema.

    label:
        0 = answer marked correct by TruthfulQA
        1 = answer marked incorrect by TruthfulQA
    """

    records = []

    for index, row in df.iterrows():

        question_id = f"TQA_{index:04d}"

        # Correct answers
        for answer_index, answer in enumerate(row["correct_answers"]):

            records.append({
                "question_id": question_id,
                "candidate_id": f"{question_id}_C_{answer_index:03d}",
                "source_dataset": "truthfulqa",
                "question": row["question"],
                "answer": answer,
                "context": "",
                "label": 0,
                "question_category": row["category"]
            })

        # Incorrect answers
        for answer_index, answer in enumerate(row["incorrect_answers"]):

            records.append({
                "question_id": question_id,
                "candidate_id": f"{question_id}_I_{answer_index:03d}",
                "source_dataset": "truthfulqa",
                "question": row["question"],
                "answer": answer,
                "context": "",
                "label": 1,
                "question_category": row["category"]
            })

    return pd.DataFrame(records)


def clean_canonical_data(df):
    """
    Clean the canonical dataset.

    Cleaning rules:
    1. Remove rows with empty questions.
    2. Remove rows with empty answers.
    3. Remove question-answer pairs with conflicting labels.
    4. Remove duplicate question-answer pairs with the same label.
    5. Validate that labels are binary.
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Remove empty questions
    # --------------------------------------------------

    df["question"] = df["question"].fillna("").str.strip()
    df = df[df["question"] != ""]


    # --------------------------------------------------
    # 2. Remove empty answers
    # --------------------------------------------------

    df["answer"] = df["answer"].fillna("").str.strip()
    df = df[df["answer"] != ""]


    # --------------------------------------------------
    # 3. Find conflicting question-answer pairs
    # --------------------------------------------------

    label_counts = (
        df
        .groupby(["question", "answer"])["label"]
        .nunique()
    )

    conflicting_pairs = label_counts[
        label_counts > 1
    ].index


    # --------------------------------------------------
    # 4. Remove ALL rows belonging to conflicting pairs
    # --------------------------------------------------

    if len(conflicting_pairs) > 0:

        conflict_index = pd.MultiIndex.from_frame(
            df[["question", "answer"]]
        )

        df = df[
            ~conflict_index.isin(conflicting_pairs)
        ]


    # --------------------------------------------------
    # 5. Remove duplicate Q&A pairs
    # --------------------------------------------------

    df = df.drop_duplicates(
        subset=["question", "answer"],
        keep="first"
    )


    # --------------------------------------------------
    # 6. Validate labels
    # --------------------------------------------------

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            "Dataset contains invalid labels."
        )


    # --------------------------------------------------
    # 7. Reset index
    # --------------------------------------------------

    df = df.reset_index(drop=True)

    return df
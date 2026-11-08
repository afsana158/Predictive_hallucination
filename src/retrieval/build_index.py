from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "processed"
        / "halueval_train.parquet"
    )

    output_dir = (
        project_root
        / "data"
        / "external"
        / "halueval_retrieval"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    corpus_path = (
        output_dir / "retrieval_corpus.parquet"
    )

    index_path = (
        output_dir / "halueval.faiss"
    )

    model_path = (
        output_dir / "embedding_model.txt"
    )

    print("Loading HaluEval...")
    df = pd.read_parquet(input_path)

    corpus = (
        df[["context"]]
        .dropna()
        .rename(columns={"context": "document"})
    )

    corpus["document"] = (
        corpus["document"]
        .astype(str)
        .str.strip()
    )

    corpus = corpus[
        corpus["document"] != ""
    ]

    corpus = (
        corpus
        .drop_duplicates(subset=["document"])
        .reset_index(drop=True)
    )

    corpus["document_id"] = (
        corpus.index.astype(str)
    )

    print(
        f"Documents in corpus: {len(corpus)}"
    )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        corpus["document"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"FAISS documents indexed: {index.ntotal}"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    corpus.to_parquet(
        corpus_path,
        index=False
    )

    model_path.write_text(
        MODEL_NAME,
        encoding="utf-8"
    )

    print("\nSaved:")
    print(index_path)
    print(corpus_path)
    print(model_path)


if __name__ == "__main__":
    main()
import pandas as pd
import spacy


nlp = spacy.load("en_core_web_sm")


def add_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    texts = (
        df["question"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    entity_counts = []
    person_counts = []
    organization_counts = []
    location_counts = []
    date_counts = []

    for doc in nlp.pipe(texts, batch_size=64):

        entity_counts.append(len(doc.ents))

        person_counts.append(
            sum(ent.label_ == "PERSON" for ent in doc.ents)
        )

        organization_counts.append(
            sum(ent.label_ == "ORG" for ent in doc.ents)
        )

        location_counts.append(
            sum(ent.label_ in {"GPE", "LOC", "FAC"} for ent in doc.ents)
        )

        date_counts.append(
            sum(ent.label_ in {"DATE", "TIME"} for ent in doc.ents)
        )

    df["entity_count"] = entity_counts
    df["person_count"] = person_counts
    df["organization_count"] = organization_counts
    df["location_count"] = location_counts
    df["date_count"] = date_counts

    return df
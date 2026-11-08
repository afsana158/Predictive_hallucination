
import re
import pandas as pd


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been",
    "by", "for", "from", "in", "into", "is", "it", "its",
    "of", "on", "or", "the", "their", "this", "that",
    "to", "was", "were", "what", "which", "who", "whom",
    "with", "where", "when", "how", "did", "does", "do",
    "has", "have", "had", "he", "she", "they", "them",
    "his", "her", "both", "also",
}


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+(?:['’-][a-z0-9]+)?",
        str(text).lower()
    )


def sentence_count(text):
    parts = re.split(r"[.!?]+", str(text))
    return sum(bool(part.strip()) for part in parts)


def extract_pregen_features(question, context):
    """Extract 27 features using only question and context."""

    q = str(question) if pd.notna(question) else ""
    c = str(context) if pd.notna(context) else ""

    q_lower = q.lower()
    c_lower = c.lower()

    q_tokens = tokenize(q)
    c_tokens = tokenize(c)

    q_content = {
        token for token in q_tokens
        if token not in STOPWORDS and not token.isdigit()
    }

    c_token_set = set(c_tokens)

    overlap_count = len(q_content & c_token_set)

    overlap_ratio = (
        overlap_count / len(q_content)
        if q_content else 0.0
    )

    return {
        "question_char_count": len(q),
        "question_word_count": len(q_tokens),
        "question_sentence_count": sentence_count(q),
        "question_question_mark_count": q.count("?"),

        "context_char_count": len(c),
        "context_word_count": len(c_tokens),
        "context_sentence_count": sentence_count(c),

        "question_content_token_count": len(q_content),
        "question_context_overlap_count": overlap_count,
        "question_context_overlap_ratio": overlap_ratio,
        "question_context_length_ratio": (
            len(q_tokens) / max(len(c_tokens), 1)
        ),

        "question_starts_who": int(bool(re.match(r"^\s*who\b", q_lower))),
        "question_starts_what": int(bool(re.match(r"^\s*what\b", q_lower))),
        "question_starts_when": int(bool(re.match(r"^\s*when\b", q_lower))),
        "question_starts_where": int(bool(re.match(r"^\s*where\b", q_lower))),
        "question_starts_which": int(bool(re.match(r"^\s*which\b", q_lower))),
        "question_starts_how": int(bool(re.match(r"^\s*how\b", q_lower))),

        "question_has_comparison": int(bool(re.search(
            r"\b(more|less|older|younger|first|later|earlier|"
            r"larger|smaller|same|versus|vs|compared)\b",
            q_lower
        ))),

        "question_has_multiple_options": int(bool(re.search(
            r"\b(both|either|or|versus|vs)\b", q_lower
        ))),

        "question_asks_count": int(bool(re.search(
            r"\b(how many|how much|number of|what percentage)\b",
            q_lower
        ))),

        "question_asks_date_or_year": int(bool(re.search(
            r"\b(what year|which year|what date|when|how old)\b",
            q_lower
        ))),

        "question_asks_location": int(bool(re.search(
            r"\b(where|which city|which country|which state|"
            r"what country|what city)\b",
            q_lower
        ))),

        "question_contains_negation": int(bool(re.search(
            r"\b(not|never|neither|without|no)\b", q_lower
        ))),

        "context_year_count": len(re.findall(
            r"\b(?:18\d{2}|19\d{2}|20\d{2})\b", c
        )),

        "context_numeric_token_count": len(re.findall(
            r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", c
        )),

        "context_has_definition_cue": int(bool(re.search(
            r"\b(is a|is an|was a|was an|refers to|known as|"
            r"also called)\b", c_lower
        ))),

        "context_has_relationship_cue": int(bool(re.search(
            r"\b(son|daughter|brother|sister|wife|husband|"
            r"father|mother|born|member|played|directed|"
            r"released|won|founded|located)\b", c_lower
        ))),
    }


def extract_pregen_features_df(df):
    """Add all 27 pre-generation features to a DataFrame."""

    df = df.copy()

    feature_records = [
        extract_pregen_features(question, context)
        for question, context in zip(df["question"], df["context"])
    ]

    features = pd.DataFrame(feature_records, index=df.index)

    return pd.concat([df, features], axis=1)

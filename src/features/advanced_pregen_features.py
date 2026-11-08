
import re
import pandas as pd

from src.features.pregeneration_features import tokenize


def extract_advanced_pregen_features(question, context):
    """
    Extract additional question/context-only signals.
    Does not use a generated answer.
    """

    q = str(question) if pd.notna(question) else ""
    c = str(context) if pd.notna(context) else ""

    q_lower = q.lower()
    c_lower = c.lower()

    q_tokens = tokenize(q)
    c_tokens = tokenize(c)

    features = {}

    # 1. Expected answer type
    features["expects_person"] = int(bool(re.search(
        r"\b(who|whom|whose|full name|which actor|which actress)\b",
        q_lower
    )))

    features["expects_location"] = int(bool(re.search(
        r"\b(where|which city|which country|which state|what country)\b",
        q_lower
    )))

    features["expects_date_or_year"] = int(bool(re.search(
        r"\b(when|what year|which year|what date|what day)\b",
        q_lower
    )))

    features["expects_number_or_duration"] = int(bool(re.search(
        r"\b(how many|how much|how long|how old|how many seasons)\b",
        q_lower
    )))

    features["expects_organization"] = int(bool(re.search(
        r"\b(which company|what company|which organization|what organization)\b",
        q_lower
    )))

    features["expects_title_or_work"] = int(bool(re.search(
        r"\b(which film|what film|which movie|what movie|which song|"
        r"which album|what book|which book|what character)\b",
        q_lower
    )))

    # 2. Question complexity and specificity
    features["requires_comparison"] = int(bool(re.search(
        r"\b(first|earlier|later|before|after|older|younger|"
        r"more|less|versus|vs|compared|which .* first)\b",
        q_lower
    )))

    features["requires_full_name"] = int(bool(re.search(
        r"\b(full name|complete name|what was the name)\b",
        q_lower
    )))

    features["requires_acronym"] = int(bool(re.search(
        r"\b(acronym|abbreviation|initials|two-letter code)\b",
        q_lower
    )))

    features["requires_duration"] = int(bool(re.search(
        r"\b(how long|how many years|how many seasons|for how many)\b",
        q_lower
    )))

    features["requires_entity_relationship"] = int(bool(re.search(
        r"\b(owns|owned by|founded by|written by|directed by|"
        r"played by|portrayed by|member of|son of|daughter of|"
        r"located in|based in)\b",
        q_lower
    )))

    # 3. Context answer-type cues
    features["context_person_cue"] = int(bool(re.search(
        r"\b(born|actor|actress|singer|director|writer|"
        r"musician|politician|businessman|businesswoman)\b",
        c_lower
    )))

    features["context_location_cue"] = int(bool(re.search(
        r"\b(city|country|state|located in|based in|"
        r"county|province|united states|united kingdom)\b",
        c_lower
    )))

    features["context_organization_cue"] = int(bool(re.search(
        r"\b(company|corporation|organization|university|"
        r"association|institution|incorporated)\b",
        c_lower
    )))

    features["context_work_title_cue"] = int(bool(re.search(
        r"\b(film|movie|novel|book|album|song|television series|"
        r"tv series|journal|magazine)\b",
        c_lower
    )))

    # 4. Context numeric and comparison evidence
    features["context_duration_cue"] = int(bool(re.search(
        r"\b(years|seasons|months|weeks|days|hours)\b",
        c_lower
    )))

    features["context_comparison_cue"] = int(bool(re.search(
        r"\b(first|second|earlier|later|before|after|"
        r"founded in|formed in|established in)\b",
        c_lower
    )))

    # 5. Basic question-context token overlap
    q_set = set(q_tokens)
    c_set = set(c_tokens)

    features["advanced_token_overlap"] = len(q_set & c_set)

    features["advanced_token_overlap_ratio"] = (
        len(q_set & c_set) / len(q_set)
        if q_set else 0.0
    )

    return features

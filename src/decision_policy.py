import math

def recommend_action(
    risk_probability,
    low_threshold=0.20,
    high_threshold=0.50
):
    """
    Convert estimated answer-failure risk into an action.

    Thresholds are provisional and must be validated
    before production use.
    """

    try:
        risk_probability = float(risk_probability)
    except (TypeError, ValueError):
        raise ValueError("Risk score must be numeric.")

    if not math.isfinite(risk_probability):
        raise ValueError("Risk score must be finite.")

    if not 0 <= risk_probability <= 1:
        raise ValueError("Risk score must be between 0 and 1.")

    if not (
        0 <= low_threshold < high_threshold <= 1
    ):
        raise ValueError("Invalid risk thresholds.")

    if risk_probability < low_threshold:
        level = "Low"
        action = "ANSWER"
        message = "Proceed with answer generation."

    elif risk_probability < high_threshold:
        level = "Medium"
        action = "RETRIEVE_MORE_EVIDENCE"
        message = "Retrieve additional evidence before answering."

    else:
        level = "High"
        action = "CLARIFY_OR_ABSTAIN"
        message = "Clarify the request or abstain from answering."

    return {
        "risk_score": round(risk_probability, 4),
        "risk_level": level,
        "recommended_action": action,
        "message": message
    }
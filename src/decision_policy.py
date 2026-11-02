
def recommend_action(risk_probability):
    """
    Convert estimated answer-failure risk into a recommendation.

    Thresholds are provisional and experimental.
    They are not calibrated or independently validated.
    """

    if not 0 <= risk_probability <= 1:
        raise ValueError("Risk probability must be between 0 and 1.")

    if risk_probability < 0.20:
        return {
            "risk_level": "Low",
            "recommended_action": "ANSWER",
            "message": (
                "Estimated risk is relatively low. "
                "The system recommends proceeding with an answer."
            )
        }

    elif risk_probability < 0.50:
        return {
            "risk_level": "Medium",
            "recommended_action": "RETRIEVE_MORE_EVIDENCE",
            "message": (
                "There is uncertainty. "
                "The system recommends retrieving additional evidence."
            )
        }

    else:
        return {
            "risk_level": "High",
            "recommended_action": "CLARIFY_OR_ABSTAIN",
            "message": (
                "Estimated risk is high. "
                "The system recommends clarification or abstaining."
            )
        }

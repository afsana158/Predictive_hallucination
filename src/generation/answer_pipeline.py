
from src.predictor import predict_answer_failure_risk
from src.generation.ollama_generator import generate_answer


def assess_and_generate(question, context):
    """
    Assess answer-failure risk before deciding whether to generate.

    Low risk:
        Generate an answer.

    Medium risk:
        Ask for additional evidence.

    High risk:
        Recommend clarification or abstention.

    The existing risk model is not modified.
    """

    # Step 1: Estimate risk before answer generation
    risk_result = predict_answer_failure_risk(
        question,
        context
    )

    action = risk_result["recommended_action"]

    # Step 2: Prepare the combined response
    result = {
        **risk_result,
        "answer": None,
        "generation_status": None
    }

    # Step 3: Follow the existing decision policy
    if action == "ANSWER":

        answer = generate_answer(
            question,
            context
        )

        result["answer"] = answer
        result["generation_status"] = "generated"

    elif action == "RETRIEVE_MORE_EVIDENCE":

        result["generation_status"] = "needs_more_evidence"

        result["recommendation_message"] = (
            "The estimated risk is medium. "
            "Please provide additional supporting context "
            "and run the assessment again."
        )

    elif action == "CLARIFY_OR_ABSTAIN":

        result["generation_status"] = "abstained"

        result["recommendation_message"] = (
            "The estimated risk is high. "
            "Please clarify the question or provide stronger evidence."
        )

    else:
        raise ValueError(
            f"Unknown recommended action: {action}"
        )

    return result

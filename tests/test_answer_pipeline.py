
from src.generation import answer_pipeline


def make_prediction(action):
    return {
        "risk_probability": 0.1,
        "risk_percentage": 10.0,
        "predicted_class": 0,
        "prediction": "Lower estimated answer-failure risk",
        "risk_level": "Low",
        "recommended_action": action,
        "recommendation_message": "Test recommendation",
        "model_status": "experimental_pilot_model",
        "model_name": "Logistic Regression",
    }


def test_low_risk_generates_answer(monkeypatch):

    monkeypatch.setattr(
        answer_pipeline,
        "predict_answer_failure_risk",
        lambda q, c: make_prediction("ANSWER")
    )

    monkeypatch.setattr(
        answer_pipeline,
        "generate_answer",
        lambda q, c: "Paris"
    )

    result = answer_pipeline.assess_and_generate(
        "What is the capital of France?",
        "France's capital is Paris."
    )

    assert result["answer"] == "Paris"
    assert result["generation_status"] == "generated"


def test_medium_risk_requests_more_evidence(monkeypatch):

    monkeypatch.setattr(
        answer_pipeline,
        "predict_answer_failure_risk",
        lambda q, c: make_prediction(
            "RETRIEVE_MORE_EVIDENCE"
        )
    )

    def unexpected_generation(q, c):
        raise AssertionError(
            "Answer should not be generated for medium risk."
        )

    monkeypatch.setattr(
        answer_pipeline,
        "generate_answer",
        unexpected_generation
    )

    result = answer_pipeline.assess_and_generate(
        "What is the capital of France?",
        "France is in Europe."
    )

    assert result["answer"] is None
    assert result["generation_status"] == "needs_more_evidence"


def test_high_risk_abstains(monkeypatch):

    monkeypatch.setattr(
        answer_pipeline,
        "predict_answer_failure_risk",
        lambda q, c: make_prediction(
            "CLARIFY_OR_ABSTAIN"
        )
    )

    def unexpected_generation(q, c):
        raise AssertionError(
            "Answer should not be generated for high risk."
        )

    monkeypatch.setattr(
        answer_pipeline,
        "generate_answer",
        unexpected_generation
    )

    result = answer_pipeline.assess_and_generate(
        "Which of these contradictory claims is correct?",
        "The evidence is insufficient."
    )

    assert result["answer"] is None
    assert result["generation_status"] == "abstained"

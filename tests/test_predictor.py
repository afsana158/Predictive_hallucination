import pytest

from src.predictor import (
    load_risk_model,
    predict_answer_failure_risk
)


def test_model_loads_successfully():
    artifact = load_risk_model()

    assert "model" in artifact
    assert "features" in artifact
    assert artifact["model_name"] == "Logistic Regression"


def test_prediction_returns_valid_result():
    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital is Paris."

    result = predict_answer_failure_risk(question, context)

    required_keys = {
        "risk_probability",
        "risk_percentage",
        "predicted_class",
        "prediction",
        "risk_level",
        "recommended_action",
        "recommendation_message",
        "model_status",
        "model_name"
    }

    assert required_keys.issubset(result.keys())

    assert 0 <= result["risk_probability"] <= 1
    assert result["predicted_class"] in [0, 1]
    assert result["risk_level"] in ["Low", "Medium", "High"]

    assert result["risk_percentage"] == round(
        result["risk_probability"] * 100, 2
    )


@pytest.mark.parametrize(
    "question, context",
    [
        ("", "Some context"),
        (None, "Some context"),
        ("   ", "Some context"),
        (123, "Some context"),
        ("What is AI?", ""),
        ("What is AI?", None),
        ("What is AI?", 123),
    ]
)
def test_invalid_inputs_are_rejected(question, context):
    with pytest.raises(ValueError):
        predict_answer_failure_risk(question, context)
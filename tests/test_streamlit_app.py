
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


APP_PATH = PROJECT_ROOT / "frontend" / "app.py"


def medium_risk_result():
    return {
        "risk_probability": 0.4673381501,
        "risk_percentage": 46.73,
        "predicted_class": 0,
        "prediction": "Lower estimated answer-failure risk",
        "risk_level": "Medium",
        "recommended_action": "RETRIEVE_MORE_EVIDENCE",
        "recommendation_message": (
            "Retrieve additional evidence before answering."
        ),
        "model_status": "experimental_pilot_model",
        "model_name": "Logistic Regression",
        "answer": None,
        "generation_status": "needs_more_evidence",
    }


def test_medium_override_generates_answer_and_preserves_risk(monkeypatch):

    from src.generation import answer_pipeline
    from src.generation import ollama_generator

    monkeypatch.setattr(
        answer_pipeline,
        "assess_and_generate",
        lambda question, context: medium_risk_result()
    )

    monkeypatch.setattr(
        ollama_generator,
        "generate_answer",
        lambda question, context: "Paris"
    )

    app = AppTest.from_file(
        str(APP_PATH),
        default_timeout=30
    ).run()

    # Enter the question and context.
    app.text_area[0].set_value(
        "What is the capital of France?"
    )

    app.text_area[1].set_value(
        "France is a country in Europe. Its capital city is Paris."
    )

    # Submit the risk assessment.
    submit_button = next(
        button for button in app.button
        if "Assess Risk" in button.label
    )

    submit_button.click().run()

    result = app.session_state["last_result"]

    assert result["risk_level"] == "Medium"
    assert result["risk_percentage"] == 46.73
    assert result["generation_status"] == "needs_more_evidence"

    # Click the Medium-risk override.
    override_button = next(
        button for button in app.button
        if "Generate Anyway" in button.label
    )

    override_button.click().run()

    result = app.session_state["last_result"]

    # Verify generated answer.
    assert result["answer"] == "Paris"
    assert result["generation_status"] == "generated"
    assert result["override_used"] is True

    # Verify original risk information was preserved.
    assert result["risk_percentage"] == 46.73
    assert result["risk_level"] == "Medium"
    assert result["recommended_action"] == "RETRIEVE_MORE_EVIDENCE"

    # Verify answer and warning are displayed.
    assert any(
        "Paris" in element.value
        for element in app.success
    )

    assert any(
        "override the Medium-risk recommendation" in element.value
        for element in app.warning
    )

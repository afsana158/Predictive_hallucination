
from pathlib import Path

import joblib
import pandas as pd

from src.features.pregeneration_features import extract_pregen_features
from src.decision_policy import recommend_action

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "pregeneration_risk_model.joblib"


def load_risk_model():
    """Load the saved experimental pre-generation risk model."""
    artifact = joblib.load(MODEL_PATH)
    return artifact


def predict_answer_failure_risk(question, context):
    """
    Estimate answer-failure risk using only the question and context.

    Returns the estimated probability and predicted class.
    """
        # Validate inputs before feature extraction
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context cannot be empty.")

    artifact = load_risk_model()
    model = artifact["model"]
    expected_features = artifact["features"]

    feature_values = extract_pregen_features(question, context)

    input_df = pd.DataFrame(
        [[feature_values[name] for name in expected_features]],
        columns=expected_features
    )

    probability = float(model.predict_proba(input_df)[0][1])
    predicted_class = int(model.predict(input_df)[0])

    decision = recommend_action(probability)

    return {
        "risk_probability": probability,
        "risk_percentage": round(probability * 100, 2),
        "predicted_class": predicted_class,
        "prediction": (
            "Potential answer failure"
            if predicted_class == 1
            else "Lower estimated answer-failure risk"
        ),
        "risk_level": decision["risk_level"],
        "recommended_action": decision["recommended_action"],
        "recommendation_message": decision["message"],
        "model_status": artifact.get("status", "unknown"),
        "model_name": artifact.get("model_name", "unknown"),
    }

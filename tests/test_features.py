from src.predictor import load_risk_model
from src.features.pregeneration_features import extract_pregen_features


def test_feature_extraction_returns_27_features():
    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital is Paris."

    features = extract_pregen_features(question, context)

    assert len(features) == 27
    assert all(isinstance(value, (int, float)) for value in features.values())


def test_feature_names_match_saved_model():
    artifact = load_risk_model()

    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital is Paris."

    features = extract_pregen_features(question, context)

    expected_features = set(artifact["features"])

    assert set(features.keys()) == expected_features
    assert len(artifact["features"]) == artifact["model"].n_features_in_
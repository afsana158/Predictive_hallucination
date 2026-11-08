import pytest

from src.decision_policy import recommend_action


@pytest.mark.parametrize(
    "probability, expected_level, expected_action",
    [
        (0.0, "Low", "ANSWER"),
        (0.1999, "Low", "ANSWER"),
        (0.20, "Medium", "RETRIEVE_MORE_EVIDENCE"),
        (0.4999, "Medium", "RETRIEVE_MORE_EVIDENCE"),
        (0.50, "High", "CLARIFY_OR_ABSTAIN"),
        (1.0, "High", "CLARIFY_OR_ABSTAIN"),
    ]
)
def test_risk_thresholds(probability, expected_level, expected_action):

    result = recommend_action(probability)

    assert result["risk_level"] == expected_level
    assert result["recommended_action"] == expected_action


@pytest.mark.parametrize(
    "probability",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        None,
        "invalid"
    ]
)
def test_invalid_probabilities_are_rejected(probability):

    with pytest.raises(ValueError):
        recommend_action(probability)


@pytest.mark.parametrize(
    "low, high",
    [
        (0.5, 0.5),
        (0.6, 0.5),
        (-0.1, 0.5),
        (0.2, 1.1)
    ]
)
def test_invalid_thresholds_are_rejected(low, high):

    with pytest.raises(ValueError):
        recommend_action(0.3, low_threshold=low, high_threshold=high)


def test_custom_thresholds():

    result = recommend_action(
        0.4,
        low_threshold=0.3,
        high_threshold=0.7
    )

    assert result["risk_level"] == "Medium"
    assert result["recommended_action"] == "RETRIEVE_MORE_EVIDENCE"
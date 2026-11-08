# Predictive Hallucination / LLM Risk Estimator

An experimental machine-learning system that estimates the risk of answer failure
before an LLM generates an answer, using only the input question and available context.

## 1. Project overview

The system extracts 27 pre-generation features from a question and its context
and passes them to a trained Logistic Regression model.

It returns an estimated answer-failure probability, risk category, and recommended action.

The current system estimates answer-failure risk. It should not be interpreted
as a fully validated hallucination detector.

## 2. Technology stack

- Python 3.11+
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Pytest

## 3. Project structure

```text
Predictive_hallucination/
├── models/
│   └── pregeneration_risk_model.joblib
├── src/
│   ├── features/
│   │   ├── basic_features.py
│   │   └── pregeneration_features.py
│   ├── decision_policy.py
│   └── predictor.py
├── tests/
│   ├── test_features.py
│   ├── test_predictor.py
│   └── test_decision_policy.py
├── requirements.txt
├── requirements-dev.txt
└── README.md

git clone https://github.com/afsana158/Predictive_hallucination.git
cd Predictive_hallucination
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
from src.predictor import predict_answer_failure_risk

question = "What is the capital of France?"
context = "France is a country in Europe. Its capital city is Paris."

result = predict_answer_failure_risk(question, context)

print(result)
)
The predictor returns:

- risk_probability
- risk_percentage
- predicted_class
- prediction
- risk_level
- recommended_action
- recommendation_message
- model_status
- model_name

Both question and context must be non-empty strings.

## 6. Risk decision policy

| Risk probability | Risk level | Recommended action |
|---|---|---|
| Below 0.20 | Low | ANSWER |
| 0.20 to below 0.50 | Medium | RETRIEVE_MORE_EVIDENCE |
| 0.50 to 1.00 | High | CLARIFY_OR_ABSTAIN |

These thresholds are provisional and require further validation.

## 7. Running automated tests

Run:

```bash
python -m pytest tests/ -v
```

## 8. Model information and limitations

The saved model is a Logistic Regression pipeline.

- Target: answer_failure_label
- Training rows: 298
- Failure cases: 30
- Features: 27
- Status: experimental_pilot_model

The dataset is relatively small. Predictions should be treated as experimental risk estimates, not guarantees of factual correctness.

The model has not been established as a calibrated or production-validated hallucination detector.

## 9. Current development status

- Baseline prediction backend: implemented
- Saved model integration: tested
- Feature compatibility: tested
- Decision policy: implemented
- Automated tests: 28 passed
- Streamlit frontend: pending
- Live demonstration: pending

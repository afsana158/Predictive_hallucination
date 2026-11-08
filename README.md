
# Predictive Hallucination / LLM Risk Estimator

An experimental machine-learning application combining pre-generation answer-failure risk estimation with context-grounded LLM answer generation.

The application uses a Logistic Regression model to estimate risk and Qwen3 through Ollama to generate answers when the decision policy permits.

**Research disclaimer:** This is an experimental answer-failure estimator, not a production-validated hallucination detector.

## 1. Project Overview

The application accepts a question and supporting context.

It extracts 27 pre-generation features and passes them to a trained Logistic Regression model.

Based on the estimated risk, the application can generate an answer, request additional evidence, or abstain.

Medium-risk users can optionally override the recommendation and generate an answer with an explicit warning.

## 2. Technology Stack

**Machine Learning**
- Python 3.11+
- Pandas
- NumPy
- Scikit-learn
- Joblib

**Answer Generation**
- Ollama
- Qwen3 4B Instruct

**Frontend**
- Streamlit
- Python
- Custom CSS

**Testing**
- Pytest
- Streamlit AppTest

## 3. Project Structure

    Predictive_hallucination/
    ├── data/
    ├── experiments/
    ├── frontend/
    │   └── app.py
    ├── models/
    │   └── pregeneration_risk_model.joblib
    ├── notebooks/
    ├── src/
    │   ├── predictor.py
    │   ├── decision_policy.py
    │   ├── features/
    │   │   ├── basic_features.py
    │   │   ├── pregeneration_features.py
    │   │   └── advanced_pregen_features.py
    │   └── generation/
    │       ├── __init__.py
    │       ├── ollama_generator.py
    │       └── answer_pipeline.py
    ├── tests/
    │   ├── test_predictor.py
    │   ├── test_features.py
    │   ├── test_decision_policy.py
    │   ├── test_answer_pipeline.py
    │   └── test_streamlit_app.py
    ├── requirements.txt
    ├── requirements-frontend.txt
    ├── requirements-dev.txt
    └── README.md

## 4. Installation

### Prerequisites

- Python 3.11 or newer
- Git
- Ollama

Clone the repository:

    git clone https://github.com/afsana158/Predictive_hallucination.git

    cd Predictive_hallucination

Create a virtual environment:

    python3.11 -m venv .venv

Activate it on macOS:

    source .venv/bin/activate

Install complete application dependencies:

    python -m pip install -r requirements-frontend.txt

Install development and testing dependencies:

    python -m pip install -r requirements-dev.txt

Verify:

    python -m pip check

## 5. Ollama and Qwen3 Setup

Download Ollama from:

https://ollama.com/download

Download the model:

    ollama pull qwen3:4b-instruct

Check installed models:

    ollama list

Start the Ollama application on macOS:

    open -a Ollama

Check the local Ollama API:

    curl http://localhost:11434/api/tags

The application uses:

- Ollama host: http://localhost:11434
- Model: qwen3:4b-instruct

Ollama must be running before answer generation is requested.

## 6. Running the Application

From the project root:

    python -m streamlit run frontend/app.py

Open the local Streamlit address, usually:

    http://localhost:8501

## 7. Application Workflow

1. Enter a question.
2. Provide supporting context.
3. The Logistic Regression model estimates answer-failure risk.
4. The decision policy selects the recommended action.
5. Qwen3 generates an answer when generation is permitted.

### Risk Decision Policy

| Probability | Risk | Recommended action |
|---|---|---|
| Below 0.20 | Low | ANSWER |
| 0.20 to below 0.50 | Medium | RETRIEVE_MORE_EVIDENCE |
| 0.50 to 1.00 | High | CLARIFY_OR_ABSTAIN |

These thresholds are provisional.

### Low Risk

The application generates an answer using Qwen3 and the supplied context.

### Medium Risk

The application recommends additional supporting evidence.

Users can provide additional evidence and reassess.

Alternatively, users can select:

**Generate Anyway — I Understand the Risk**

The override permits Qwen3 answer generation while preserving the original risk score, risk category and recommended action.

The generated answer displays an override warning.

### High Risk

The application abstains from generating an answer and recommends clarification or stronger supporting evidence.

## 8. Backend Components

**src/predictor.py**

Loads the saved model, extracts the expected features and predicts answer-failure risk.

Main function:

    predict_answer_failure_risk(question, context)

**src/features/pregeneration_features.py**

Extracts the 27 pre-generation features from the question and context.

**src/decision_policy.py**

Maps the predicted probability to a risk category and recommended action.

**src/generation/ollama_generator.py**

Connects to Ollama and generates context-grounded answers.

**src/generation/answer_pipeline.py**

Combines risk estimation with conditional answer generation.

**frontend/app.py**

Contains the Streamlit user interface, input validation, risk display, additional-evidence workflow and Medium-risk override.

## 9. Running Automated Tests

Run the complete test suite:

    python -m pytest tests/ -v

Check dependencies:

    python -m pip check

The current development checkpoint contains 32 automated tests covering:

- Feature extraction and model compatibility.
- Prediction output and invalid inputs.
- Decision-policy thresholds.
- Low, Medium and High risk paths.
- Answer-generation integration.
- Streamlit Medium-risk override.

The UI override test uses a mocked generator and does not require a live Ollama request.

## 10. Model Information

The saved model is a Logistic Regression pipeline.

| Property | Value |
|---|---|
| Target | answer_failure_label |
| Training observations | 298 |
| Failure cases | 30 |
| Input features | 27 |
| Model status | experimental_pilot_model |

The model estimates answer-failure risk, not hallucination probability specifically.

Its training and evaluation dataset is relatively small. The estimated probabilities are not established as calibrated or production-validated.

## 11. Limitations

- The model is experimental and requires further validation.
- Answer-failure labels include errors beyond factual hallucinations.
- The risk thresholds are provisional.
- Qwen3 may still generate unsupported or incorrect answers.
- Answer quality depends on the supplied context.
- Automatic evidence retrieval is not yet implemented.
- The application currently requires a locally running Ollama instance.

## 12. Development Status

- Logistic Regression baseline: implemented.
- 27-feature extraction: implemented.
- Risk decision policy: implemented.
- Qwen3 answer generation: integrated.
- Combined risk and generation pipeline: implemented.
- Streamlit frontend: implemented.
- Additional-evidence workflow: implemented.
- Medium-risk override: implemented.
- Automated tests: 32 passed in the current development checkpoint.
- Live deployment: pending.

## Repository

https://github.com/afsana158/Predictive_hallucination

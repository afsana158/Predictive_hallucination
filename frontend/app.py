
from pathlib import Path
import sys

import streamlit as st

# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.answer_pipeline import assess_and_generate


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LLM Risk Estimator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #0b1324,
            #111c32,
            #172844
        );
        color: #f1f5f9;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    .main-subtitle {
        color: #a8b7cc;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-heading {
        color: #cbd5e1;
        font-size: 19px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: #1b2b45;
        border: 1px solid #344963;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .disclaimer {
        background: #1b2b45;
        border-left: 4px solid #60a5fa;
        padding: 13px;
        border-radius: 7px;
        color: #cbd5e1;
        font-size: 13px;
        margin-top: 20px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 43px;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: #17263e;
        color: #f8fafc;
        border: 1px solid #405574;
        border-radius: 8px;
    }

    div[data-testid="stForm"] {
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

if "recent_checks" not in st.session_state:
    st.session_state.recent_checks = []


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def save_result(question, context, result):
    """Store the latest result and update recent checks."""

    st.session_state.last_result = result
    st.session_state.last_question = question
    st.session_state.last_context = context

    check = {
        "question": question,
        "risk": result["risk_percentage"],
        "level": result["risk_level"],
        "status": result["generation_status"]
    }

    st.session_state.recent_checks.insert(0, check)
    st.session_state.recent_checks = (
        st.session_state.recent_checks[:5]
    )


def run_assessment(question, context):
    """Run the combined risk and answer pipeline."""

    with st.spinner(
        "Estimating risk and processing your request..."
    ):
        result = assess_and_generate(question, context)

    save_result(question, context, result)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ LLM Risk Estimator")

st.markdown(
    """
    <div class="main-subtitle">
    AI Answer Generation and Pre-generation Risk Assessment
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    Evaluate the risk of answer failure before generation,
    then generate an answer when the decision policy permits.
    """
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ System Information")

    st.markdown("### Risk Estimator")
    st.success("Logistic Regression")

    st.write("Features: 27")
    st.write("Training observations: 298")
    st.write("Failure cases: 30")

    st.divider()

    st.markdown("### Answer Generator")

    st.success("Ollama")

    st.write("Model: `qwen3:4b-instruct`")

    st.divider()

    st.markdown("### Risk Thresholds")

    st.write("🟢 Low: Below 20%")
    st.write("🟡 Medium: 20% – below 50%")
    st.write("🔴 High: 50% and above")

    st.divider()

    st.caption(
        "The risk model is experimental. "
        "Its scores are not independently calibrated."
    )


# --------------------------------------------------
# MAIN TWO-COLUMN LAYOUT
# --------------------------------------------------

left_column, right_column = st.columns(
    [0.95, 1.35],
    gap="large"
)


# ==================================================
# LEFT: USER INPUT
# ==================================================

with left_column:

    st.markdown(
        '<div class="section-heading">Input details</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        with st.form("assessment_form"):

            question = st.text_area(
                "Question",
                placeholder=(
                    "Example: What is the capital of France?"
                ),
                height=110
            )

            context = st.text_area(
                "Supporting context",
                placeholder=(
                    "Enter the information or evidence "
                    "that the answer should be based on..."
                ),
                height=190
            )

            submitted = st.form_submit_button(
                "🔍 Assess Risk & Generate Answer",
                use_container_width=True,
                type="primary"
            )

            if submitted:

                if not question.strip():
                    st.warning("Please enter a question.")

                elif not context.strip():
                    st.warning(
                        "Please provide supporting context."
                    )

                else:
                    try:
                        run_assessment(
                            question,
                            context
                        )

                    except (ValueError, RuntimeError) as error:
                        st.error(str(error))

                    except Exception:
                        st.error(
                            "An unexpected error occurred. "
                            "Please check the backend and Ollama."
                        )


    st.markdown(
        """
        <div class="disclaimer">
        This is an experimental pilot model trained on
        298 observations, including 30 failure cases.
        Treat its risk estimate as a signal for review,
        not a guarantee of factual correctness.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# RIGHT: RESULTS
# ==================================================

with right_column:

    st.markdown(
        '<div class="section-heading">Risk assessment results</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.last_result

    if result is None:

        with st.container(border=True):

            st.info(
                "Enter a question and context, then click "
                "'Assess Risk & Generate Answer' "
                "to see the results."
            )

    else:

        with st.container(border=True):

            risk = result["risk_percentage"]
            level = result["risk_level"]

            st.metric(
                "Estimated answer-failure risk",
                f"{risk:.2f}%"
            )

            st.progress(
                max(0.0, min(1.0, risk / 100))
            )

            st.caption(
                "Estimated score from the experimental pilot model"
            )

            if level == "Low":
                st.success("🟢 LOW RISK")

            elif level == "Medium":
                st.warning("🟡 MEDIUM RISK")

            else:
                st.error("🔴 HIGH RISK")

            st.markdown(
                f"**Recommended action:** "
                f"`{result['recommended_action']}`"
            )

            st.write(
                result["recommendation_message"]
            )

            st.caption(
                f"Model: {result['model_name']} | "
                f"Status: {result['model_status']}"
            )

            st.divider()

            # ------------------------------------------
            # GENERATED ANSWER
            # ------------------------------------------

            if result["generation_status"] == "generated":

                st.markdown("### ✨ Generated Answer")

                st.success(result["answer"])

                st.caption(
                    "Generated by Qwen3 using the supplied context."
                )

            # ------------------------------------------
            # MEDIUM RISK
            # ------------------------------------------

            elif (
                result["generation_status"]
                == "needs_more_evidence"
            ):

                st.markdown("### 📚 Additional Evidence Needed")

                st.info(
                    "The model recommends strengthening "
                    "the supporting context before generating "
                    "an answer."
                )

                with st.form("additional_evidence_form"):

                    additional_context = st.text_area(
                        "Provide additional evidence",
                        placeholder=(
                            "Add more relevant facts, "
                            "sources or supporting information..."
                        ),
                        height=130
                    )

                    reassess = st.form_submit_button(
                        "Reassess with Additional Evidence",
                        use_container_width=True
                    )

                    if reassess:

                        if not additional_context.strip():

                            st.warning(
                                "Please enter additional evidence."
                            )

                        else:

                            combined_context = (
                                st.session_state.last_context
                                + "\n\nAdditional supporting evidence:\n"
                                + additional_context.strip()
                            )

                            try:
                                run_assessment(
                                    st.session_state.last_question,
                                    combined_context
                                )

                                st.rerun()

                            except Exception as error:

                                st.error(
                                    f"Reassessment failed: {error}"
                                )

            # ------------------------------------------
            # HIGH RISK
            # ------------------------------------------

            elif result["generation_status"] == "abstained":

                st.markdown("### ⚠️ Answer Withheld")

                st.error(
                    "The system has not generated an answer "
                    "because the estimated risk is high."
                )

                st.write(
                    "Clarify the question or provide stronger "
                    "supporting evidence before retrying."
                )

            # ------------------------------------------
            # RAW MODEL OUTPUT
            # ------------------------------------------

            with st.expander("Show raw prediction output"):

                st.json(result)


# ==================================================
# RECENT CHECKS
# ==================================================

st.divider()

st.markdown("## 🕘 Recent Checks")

if not st.session_state.recent_checks:

    st.caption("No assessments performed yet.")

else:

    for check in st.session_state.recent_checks:

        icon = (
            "🟢" if check["level"] == "Low"
            else "🟡" if check["level"] == "Medium"
            else "🔴"
        )

        col1, col2 = st.columns([4, 1])

        with col1:
            st.write(
                f"{icon} {check['question']}"
            )

        with col2:
            st.write(
                f"**{check['risk']:.2f}%**"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Predictive Hallucination / LLM Risk Estimator | "
    "Experimental research prototype"
)

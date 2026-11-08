
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "qwen3:4b-instruct"


def generate_answer(question, context, model=DEFAULT_MODEL):
    """
    Generate an answer using the supplied question and context.

    This function does not calculate risk.
    Risk estimation is handled separately by predictor.py.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = f"""
/no_think

Answer the question using only the provided context.
If the context does not contain enough information, say so.
Do not invent facts.
Return only the final answer. Do not include reasoning.

Question:
{question.strip()}

Context:
{context.strip()}

Final answer:
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    request = Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama API error ({error.code}): {details}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        ) from error

    answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    return answer

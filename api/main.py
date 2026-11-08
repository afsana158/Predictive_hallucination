"""Thin REST wrapper around src.predictor. Run from the repo root:
    uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.predictor import predict_answer_failure_risk

app = FastAPI(title="LLM Risk Estimator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictIn(BaseModel):
    question: str
    context: str


def _clean(v):
    """Make numpy / pandas values JSON friendly."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item"):
        return v.item()
    return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictIn):
    try:
        result = predict_answer_failure_risk(body.question, body.context)
    except ValueError as e:  # predictor rejected the input
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # anything else: show the real cause
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return _clean(result)

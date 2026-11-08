# Frontend (React + Vite)

1. Backend (repo root): `pip install fastapi uvicorn` then `uvicorn api.main:app --reload --port 8000`
2. Frontend: `cd frontend && npm install && npm run dev` and open http://localhost:5173

Vite proxies `/api/*` to port 8000. Adapt `api/main.py` (`_predict_fn`) and `src/api.js` (`normalize`) if your predictor uses different names.

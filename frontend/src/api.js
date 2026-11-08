// All backend calls live here. The UI never computes risk levels itself.
export function normalize(raw) {
  const p = Number(raw.risk_probability);
  return {
    probability: Number.isFinite(p) ? p : null,
    level: String(raw.risk_level ?? "Unknown"),
    action: raw.recommended_action ?? "",
    message: raw.recommendation_message ?? "",
    prediction: raw.prediction ?? "",
    modelStatus: raw.model_status ?? "",
    raw,
  };
}

export async function predict(question, context) {
  let res;
  try {
    res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context }),
    });
  } catch {
    throw new Error("Cannot reach the API. Start it with: uvicorn api.main:app --port 8000");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : `Server error ${res.status}`);
  }
  return normalize(data);
}

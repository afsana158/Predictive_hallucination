import { useState } from "react";
import { predict } from "./api.js";

const EXAMPLES = [
  {
    label: "Supported by context",
    question: "In what year was the Eiffel Tower completed?",
    context: "The Eiffel Tower, a wrought-iron lattice tower in Paris, was finished in 1889 for the World's Fair.",
  },
  {
    label: "Numeric detail",
    question: "How many people lived in Rome in 1500, and how did that compare to Paris?",
    context: "Rome was a mid-sized city in 1500, while Paris was among the largest cities in Europe at the time.",
  },
  {
    label: "Vague context",
    question: "What happens if you crack your knuckles a lot?",
    context: "Knuckle cracking is a common habit. People often wonder about its long-term effects.",
  },
];

const tone = (level) => {
  const l = level.toLowerCase();
  return l.startsWith("low") ? "low" : l.startsWith("med") ? "med" : l.startsWith("high") ? "high" : "none";
};

function RiskMeter({ probability, level }) {
  const pct = Math.min(Math.max(probability ?? 0, 0), 1) * 100;
  return (
    <div className={`meter ${tone(level)}`} role="img" aria-label={`Failure probability ${pct.toFixed(1)} percent`}>
      <div className="track"><div className="fill" style={{ width: `${pct}%` }} /></div>
      <div className="marker" style={{ left: `${pct}%` }} />
      <div className="scale"><span>0%</span><span>50%</span><span>100%</span></div>
    </div>
  );
}

function Result({ r }) {
  const [showRaw, setShowRaw] = useState(false);
  const pct = r.probability === null ? "n/a" : `${(r.probability * 100).toFixed(1)}%`;
  return (
    <section className={`result ${tone(r.level)}`} aria-live="polite">
      <p className="level">{r.level} risk</p>
      <p className="prob"><strong>{pct}</strong> estimated chance the model answers incorrectly</p>
      {r.probability !== null && <RiskMeter probability={r.probability} level={r.level} />}
      {r.action && <p className="action"><strong>{r.action}</strong>{r.message && <><br />{r.message}</>}</p>}
      {r.modelStatus && <p className="prob">Model status: {r.modelStatus}</p>}
      <button type="button" className="link" onClick={() => setShowRaw(!showRaw)}>
        {showRaw ? "Hide raw output" : "Show raw output"}
      </button>
      {showRaw && <pre>{JSON.stringify(r.raw, null, 2)}</pre>}
    </section>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [context, setContext] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  async function run(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const r = await predict(question.trim(), context.trim());
      setResult(r);
      setHistory((h) => [{ question: question.trim(), context, r }, ...h].slice(0, 8));
    } catch (err) {
      setResult(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <h1>Will this question trip up your LLM?</h1>
        <p>Estimate the chance of a wrong answer before the model generates anything.</p>
      </header>

      <div className="grid">
        <form onSubmit={run}>
          <div className="chips">
            {EXAMPLES.map((ex) => (
              <button type="button" key={ex.label} className="chip"
                onClick={() => { setQuestion(ex.question); setContext(ex.context); }}>
                {ex.label}
              </button>
            ))}
          </div>
          <label htmlFor="q">Question</label>
          <textarea id="q" rows={3} value={question} required onChange={(e) => setQuestion(e.target.value)}
            placeholder="Type the question you plan to send to the LLM" />
          <label htmlFor="c">Context</label>
          <textarea id="c" rows={6} value={context} required onChange={(e) => setContext(e.target.value)}
            placeholder="Paste the retrieved passage or document the model will see" />
          <button className="primary" disabled={loading || !question.trim() || !context.trim()}>
            {loading ? "Estimating..." : "Estimate risk"}
          </button>
        </form>

        <div>
          {error && <p className="error" role="alert">{error}</p>}
          {result ? <Result r={result} /> : !error && (
            <p className="empty">Enter a question and its context, then choose Estimate risk to see the result here.</p>
          )}
          {history.length > 0 && (
            <section className="history">
              <h2>Recent checks</h2>
              <ul>
                {history.map((h, i) => (
                  <li key={i}>
                    <button type="button" onClick={() => { setQuestion(h.question); setContext(h.context); setResult(h.r); setError(""); }}>
                      <span className={`dot ${tone(h.r.level)}`} />
                      <span className="hq">{h.question}</span>
                      <span className="hp">{h.r.probability === null ? "" : `${(h.r.probability * 100).toFixed(0)}%`}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      </div>

      <footer>
        This is an experimental pilot model trained on 298 questions. Treat the estimate as a signal for review, not a verdict.
      </footer>
    </main>
  );
}

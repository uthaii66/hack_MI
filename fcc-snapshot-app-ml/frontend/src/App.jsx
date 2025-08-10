import React, { useEffect, useState } from "react";
const API_BASE = "http://localhost:8000/api";

function RiskBanner({ label, score, factors }) {
  const cls =
    label === "LOW"
      ? "chip low"
      : label === "MEDIUM"
      ? "chip medium"
      : "chip high";
  return (
    <div className="card">
      <div
        className="row"
        style={{ alignItems: "center", justifyContent: "space-between" }}
      >
        <div>
          <div className="title" style={{ marginBottom: 4 }}>
            Member Snapshot
          </div>
          <div className="subtitle muted">
            Risk predicted with XGBoost; top factors via SHAP
          </div>
        </div>
        <div className={cls} title={`Prob: ${score.toFixed(2)}`}>
          Risk: {label} ({(score * 100).toFixed(0)}%)
        </div>
      </div>
      {!!factors?.length && (
        <div style={{ marginTop: 6 }}>
          <div className="section-title">Top Contributing Factors</div>
          <ul>
            {factors.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function SnapshotCard({ summary }) {
  return (
    <div className="card">
      <div
        className="row"
        style={{ alignItems: "center", justifyContent: "space-between" }}
      >
        <div className="title">Clinical Snapshot</div>
      </div>
      <div className="section-title">Key Findings</div>
      <ul>
        {summary.key_findings?.map((k, i) => (
          <li key={i}>{k}</li>
        ))}
      </ul>

      <div className="section-title">Abnormal Labs</div>
      {summary.abnormal_labs?.length ? (
        <ul>
          {summary.abnormal_labs.map((l, i) => (
            <li key={i}>
              <strong>{l.name}</strong>: {l.value} {l.unit}{" "}
              <span
                className="chip"
                style={{ background: "#eef2ff", color: "#1f4ae0" }}
              >
                Flag {l.flag}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <div className="muted">No abnormal labs.</div>
      )}

      <div className="section-title">Recent Events</div>
      {summary.recent_events?.map((e, i) => (
        <div className="event" key={i}>
          <div>
            <strong>{e.type}</strong> — {e.ts}
          </div>
          <div className="muted">{JSON.stringify(e.payload)}</div>
        </div>
      ))}

      <div className="section-title">Next Steps</div>
      <ul>
        {summary.recommended_next_steps?.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>
    </div>
  );
}

export default function App() {
  const [memberId, setMemberId] = useState("MEM12345");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  async function fetchSnapshot(id) {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE}/members/${id}/snapshot`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchSnapshot(memberId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [memberId]);

  return (
    <div className="container">
      <header>
        <h1 style={{ margin: 0, color: "#14213d" }}>
          Family Care Central — AI Snapshot (ML)
        </h1>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            className="input"
            value={memberId}
            onChange={(e) => setMemberId(e.target.value)}
          />
          <button
            className="btn"
            onClick={() => fetchSnapshot(memberId)}
            disabled={loading}
          >
            {loading ? "Loading…" : "Generate Snapshot"}
          </button>
        </div>
      </header>

      {error && (
        <div
          className="card"
          style={{
            border: "1px solid #ffd1d1",
            background: "#fff5f5",
            color: "#8b1111",
          }}
        >
          Error: {error}
        </div>
      )}

      {data ? (
        <>
          <RiskBanner
            label={data.risk.label}
            score={data.risk.score}
            factors={data.risk.factors}
          />
          <SnapshotCard summary={data.summary} />
          <div className="muted">
            Generated at: {new Date(data.generated_at).toLocaleString()}
          </div>
        </>
      ) : (
        <div className="card">
          Click “Generate Snapshot” to build a member summary.
        </div>
      )}
    </div>
  );
}

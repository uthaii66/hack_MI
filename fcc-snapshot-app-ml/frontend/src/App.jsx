import React, { useEffect, useMemo, useState } from "react";

const API_BASE = "http://localhost:8000/api";

// --- Helpers ---
const EVENT_ICONS = {
  LAB_RESULT: "🧪",
  VITALS: "❤️‍🩹",
  APPOINTMENT: "📅",
  MED_REFILL: "💊",
  CLAIM: "🧾",
  ADT_ADMIT: "🏥⬆️",
  ADT_DISCHARGE: "🏥⬇️",
};
const ALL_TYPES = [
  "ALL",
  "LAB_RESULT",
  "VITALS",
  "APPOINTMENT",
  "MED_REFILL",
  "CLAIM",
  "ADT_ADMIT",
  "ADT_DISCHARGE",
];

function EventIcon({ type }) {
  return (
    <span className="icon" title={type}>
      {EVENT_ICONS[type] || "📄"}
    </span>
  );
}

function RiskGauge({ prob, label }) {
  const pct = Math.max(0, Math.min(1, prob));
  const r = 54,
    cx = 60,
    cy = 60;
  const d = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
  const color =
    label === "HIGH" ? "#e11d48" : label === "MEDIUM" ? "#f59e0b" : "#10b981";

  return (
    <svg
      width="140"
      height="80"
      viewBox="0 0 120 80"
      role="img"
      aria-label={`Risk ${label} ${Math.round(pct * 100)}%`}
    >
      {/* base track */}
      <path
        d={d}
        pathLength="100"
        stroke="#e5e7eb"
        strokeWidth="10"
        fill="none"
        strokeLinecap="round"
      />
      {/* progress */}
      <path
        d={d}
        pathLength="100"
        stroke={color}
        strokeWidth="10"
        fill="none"
        strokeLinecap="round"
        strokeDasharray={`${pct * 100} ${100 - pct * 100}`}
        strokeDashoffset="0"
      />
      {/* percent label */}
      <text x="60" y="72" textAnchor="middle" fontSize="12" fill="#334155">
        {Math.round(pct * 100)}%
      </text>
    </svg>
  );
}

function FactorBars({ factors }) {
  // Expect strings like "feature: impact -0.23"; negative = protective (green), positive = risk (red)
  const parsed = (factors || []).map((f) => {
    const m = f.match(/^(.*?): impact\\s*(-?\\d+(?:\\.\\d+)?)/);
    return m
      ? { name: m[1], impact: parseFloat(m[2]) }
      : { name: f, impact: 0 };
  });
  return (
    <div className="bars">
      {parsed.map((p, i) => {
        const mag = Math.min(1, Math.abs(p.impact));
        const width = `${Math.round(mag * 100)}%`;
        const fg = p.impact >= 0 ? "#ef4444" : "#10b981";
        return (
          <div key={i}>
            <div className="bar-label">{p.name}</div>
            <div className="bar">
              <span style={{ width, background: fg }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [memberId, setMemberId] = useState("MEM12345");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [expanded, setExpanded] = useState({});
  const [filterType, setFilterType] = useState("ALL");
  const [abnormalOnly, setAbnormalOnly] = useState(false);

  async function fetchSnapshot(id) {
    try {
      setLoading(true);
      setError(null);
      const [snapRes, tlRes] = await Promise.all([
        fetch(`${API_BASE}/members/${id}/snapshot`),
        fetch(`${API_BASE}/members/${id}/timeline?days=120&n=40`),
      ]);
      if (!snapRes.ok) throw new Error(`Snapshot HTTP ${snapRes.status}`);
      if (!tlRes.ok) throw new Error(`Timeline HTTP ${tlRes.status}`);
      const snap = await snapRes.json();
      const tl = await tlRes.json();
      setData(snap);
      setTimeline(tl.events || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchSnapshot(memberId);
  }, []);

  const filteredTimeline = useMemo(() => {
    return (timeline || []).filter((e) => {
      if (filterType !== "ALL" && e.type !== filterType) return false;
      if (!abnormalOnly) return true;
      // abnormal filter: labs with H/L flags; vitals with low SpO2 or high BP
      if (e.type === "LAB_RESULT") {
        const f = e.payload?.flag;
        return f === "H" || f === "L";
      }
      if (e.type === "VITALS") {
        const s = e.payload?.spo2;
        const sbp = e.payload?.bp_systolic;
        const dbp = e.payload?.bp_diastolic;
        return (
          (typeof s === "number" && s < 94) ||
          (typeof sbp === "number" && sbp > 140) ||
          (typeof dbp === "number" && dbp > 90)
        );
      }
      return false;
    });
  }, [timeline, filterType, abnormalOnly]);

  const toggle = (idx) =>
    setExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }));

  const label = data?.risk?.label || "LOW";
  const prob = data?.risk?.score || 0;
  const chipClass =
    label === "HIGH"
      ? "chip high"
      : label === "MEDIUM"
      ? "chip medium"
      : "chip low";

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

      {data && (
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
                Risk predicted with XGBoost; SHAP visual & factors
              </div>
              <div className="narratives">
                {(data.risk?.narratives || []).map((n, i) => (
                  <span className="tag" key={i}>
                    {n}
                  </span>
                ))}
              </div>
            </div>
            <div className="gauge-wrap">
              <RiskGauge prob={prob} label={label} />
              <div
                className={chipClass}
                title={`Probability: ${(prob * 100).toFixed(0)}%`}
              >
                Risk: {label}
              </div>
            </div>
          </div>
          <div className="row" style={{ gap: 24, marginTop: 12 }}>
            <div>
              <div className="section-title">Top Factors</div>
              <FactorBars factors={data.risk?.factors} />
            </div>
            {data.risk?.shap_plot && (
              <div>
                <div className="section-title">SHAP Impact</div>
                <img
                  className="shap"
                  alt="SHAP"
                  src={`data:image/png;base64,${data.risk.shap_plot}`}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {data && (
        <div className="card">
          <div className="title">Clinical Snapshot</div>
          <div className="section-title">Key Findings</div>
          <ul>
            {data.summary?.key_findings?.map((k, i) => (
              <li key={i}>{k}</li>
            ))}
          </ul>
          <div className="section-title">Abnormal Labs</div>
          {data.summary?.abnormal_labs?.length ? (
            <ul>
              {data.summary.abnormal_labs.map((l, i) => (
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
        </div>
      )}

      <div className="card">
        <div
          className="row"
          style={{ alignItems: "center", justifyContent: "space-between" }}
        >
          <div className="title">Timeline</div>
          <div className="filter-row">
            <label className="muted">Type</label>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
            >
              {ALL_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <label
              className="muted"
              style={{ display: "flex", alignItems: "center", gap: 6 }}
            >
              <input
                type="checkbox"
                checked={abnormalOnly}
                onChange={(e) => setAbnormalOnly(e.target.checked)}
              />{" "}
              Abnormal only
            </label>
          </div>
        </div>
        <div className="section-title">Recent Events</div>
        {filteredTimeline.map((e, i) => (
          <div className="event" key={i} onClick={() => toggle(i)}>
            <div className="event-head">
              <EventIcon type={e.type} />
              <div>
                <strong>{e.type}</strong> —{" "}
                <span className="muted">{e.ts_iso || e.ts}</span>
              </div>
            </div>
            {expanded[i] && (
              <div className="detail">
                <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(e.payload, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ))}
        {!filteredTimeline.length && (
          <div className="muted">No events match the filter.</div>
        )}
      </div>

      {data && (
        <div className="muted">
          Generated at: {new Date(data.generated_at).toLocaleString()}
        </div>
      )}
    </div>
  );
}

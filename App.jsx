import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";

const API = process.env.NODE_ENV === "production" ? "" : "http://localhost:5000";

const COLORS = {
  positive: "#22c55e",
  neutral:  "#94a3b8",
  negative: "#ef4444",
  bg:       "#0b1020",
  card:     "#121833",
  cardLite: "#1a224a",
  border:   "#27305a",
  text:     "#e6e9ef",
  muted:    "#8a93b8",
  accent:   "#7c3aed",
  accent2:  "#06b6d4",
};

const TABS = [
  { id: "live",    label: "Live Scrape" },
  { id: "predict", label: "Predict Text" },
  { id: "upload",  label: "Upload CSV" },
  { id: "history", label: "History" },
  { id: "models",  label: "Models" },
];

export default function App() {
  const [tab, setTab] = useState("live");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/health`).then(r => r.json()).then(setHealth).catch(() => {});
    const t = setInterval(() => {
      fetch(`${API}/api/health`).then(r => r.json()).then(setHealth).catch(() => {});
    }, 15000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={S.app}>
      <Header health={health} />
      <nav style={S.tabs}>
        {TABS.map(t => (
          <button key={t.id}
            onClick={() => setTab(t.id)}
            style={{ ...S.tab, ...(tab === t.id ? S.tabActive : {}) }}>
            {t.label}
          </button>
        ))}
      </nav>
      <main style={S.main}>
        {tab === "live"    && <LiveScrape />}
        {tab === "predict" && <Predict />}
        {tab === "upload"  && <Upload />}
        {tab === "history" && <History />}
        {tab === "models"  && <Models health={health} />}
      </main>
      <footer style={S.footer}>
        <span>Real-time multi-source sentiment · RoBERTa + sklearn ensemble</span>
        <span style={{ color: COLORS.muted }}>
          {health?.roberta_ready ? "● RoBERTa loaded" : "○ RoBERTa loading…"}
          {"   ·   "}
          {health?.trained_on ? `Ensemble trained on ${health.trained_on} samples` : "Cold-start ensemble"}
        </span>
      </footer>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
function Header({ health }) {
  return (
    <header style={S.header}>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div style={S.logo}>S</div>
        <div>
          <div style={{ fontSize: 20, fontWeight: 700 }}>AI Sentiment Analyzer</div>
          <div style={{ fontSize: 12, color: COLORS.muted }}>
            Real-time · Multi-source · RoBERTa + sklearn ensemble
          </div>
        </div>
      </div>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <Pill label="device"  value={health?.device || "—"} />
        <Pill label="models"  value={(health?.sklearn_models?.length || 0) + (health?.roberta_ready ? 1 : 0)} />
        <Pill label="status"  value={health?.status || "..."} ok={health?.status === "ok"} />
      </div>
    </header>
  );
}

function Pill({ label, value, ok }) {
  return (
    <div style={S.pill}>
      <span style={{ color: COLORS.muted, fontSize: 11 }}>{label}</span>
      <span style={{
        fontSize: 13,
        fontWeight: 600,
        color: ok === undefined ? COLORS.text : ok ? COLORS.positive : COLORS.negative,
      }}>{value}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
function LiveScrape() {
  const [query, setQuery] = useState("");
  const [sources, setSources] = useState({ reddit: true, hackernews: true, trustpilot: false });
  const [limit, setLimit] = useState(30);
  const [reviews, setReviews] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [meta, setMeta] = useState(null);
  const esRef = useRef(null);

  const start = () => {
    if (!query.trim() || streaming) return;
    setReviews([]);
    setMeta(null);
    setStreaming(true);

    const srcStr = Object.entries(sources).filter(([, v]) => v).map(([k]) => k).join(",");
    if (!srcStr) { setStreaming(false); alert("Pick at least one source"); return; }

    const url = `${API}/api/scrape/stream?query=${encodeURIComponent(query)}&sources=${srcStr}&limit=${limit}`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "meta") setMeta(msg);
        else if (msg.type === "review") setReviews(prev => [...prev, msg.data]);
        else if (msg.type === "done") {
          setStreaming(false);
          setMeta(m => ({ ...(m || {}), done: true, total: msg.count, cached: msg.cached }));
          es.close();
        }
      } catch (e) { console.warn(e); }
    };
    es.onerror = () => { setStreaming(false); es.close(); };
  };

  const stop = () => { esRef.current?.close(); setStreaming(false); };

  useEffect(() => () => esRef.current?.close(), []);

  // Aggregations
  const counts = useMemo(() => {
    const c = { positive: 0, neutral: 0, negative: 0 };
    reviews.forEach(r => { if (c[r.sentiment] != null) c[r.sentiment]++; });
    return c;
  }, [reviews]);
  const total = reviews.length;
  const pieData = ["positive", "neutral", "negative"].map(k => ({ name: k, value: counts[k] }));
  const sourceData = useMemo(() => {
    const m = {};
    reviews.forEach(r => { const k = (r.source || "unknown").split(":")[0]; m[k] = (m[k] || 0) + 1; });
    return Object.entries(m).map(([name, count]) => ({ name, count }));
  }, [reviews]);

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Card>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: 10, alignItems: "end" }}>
          <Field label="Query (company name, product, topic — e.g. 'tesla', 'iphone 15', 'amazon.com')">
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === "Enter" && start()}
              placeholder="What do you want to analyze?"
              style={S.input}
              disabled={streaming}
            />
          </Field>
          <Field label="Limit">
            <input type="number" min={5} max={100} value={limit}
              onChange={e => setLimit(parseInt(e.target.value || "30"))}
              style={{ ...S.input, width: 80 }} disabled={streaming} />
          </Field>
          {!streaming
            ? <button onClick={start} style={S.primaryBtn} disabled={!query.trim()}>Scrape & Analyze</button>
            : <button onClick={stop} style={S.dangerBtn}>Stop</button>}
        </div>
        <div style={{ display: "flex", gap: 18, marginTop: 12 }}>
          {["reddit", "hackernews", "trustpilot"].map(s => (
            <label key={s} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 13 }}>
              <input
                type="checkbox"
                checked={sources[s]}
                onChange={e => setSources(prev => ({ ...prev, [s]: e.target.checked }))}
                disabled={streaming}
              />
              <span style={{ textTransform: "capitalize" }}>{s}</span>
              {s === "trustpilot" && <span style={{ color: COLORS.muted, fontSize: 11 }}>(use full slug e.g. amazon.com)</span>}
            </label>
          ))}
        </div>
        {meta && (
          <div style={{ marginTop: 10, fontSize: 12, color: COLORS.muted }}>
            {meta.cached ? "↻ Cached result · " : "● Live · "}
            {meta.sources ? `sources: ${meta.sources.join(", ")} · ` : ""}
            {streaming ? `streaming… ${total} so far` : meta.done ? `done (${meta.total ?? total} reviews)` : ""}
          </div>
        )}
      </Card>

      {total > 0 && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
            <StatBox label="Total" value={total} />
            <StatBox label="Positive" value={counts.positive} pct={total ? counts.positive / total : 0} color={COLORS.positive} />
            <StatBox label="Neutral"  value={counts.neutral}  pct={total ? counts.neutral / total : 0}  color={COLORS.neutral} />
            <StatBox label="Negative" value={counts.negative} pct={total ? counts.negative / total : 0} color={COLORS.negative} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Card title="Sentiment distribution">
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={80} label>
                    {pieData.map((d) => <Cell key={d.name} fill={COLORS[d.name]} />)}
                  </Pie>
                  <Legend />
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="By source">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={sourceData}>
                  <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
                  <XAxis dataKey="name" stroke={COLORS.muted} />
                  <YAxis stroke={COLORS.muted} />
                  <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}` }} />
                  <Bar dataKey="count" fill={COLORS.accent} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title={`Reviews (${total})${streaming ? " · streaming" : ""}`}>
            <div style={{ display: "grid", gap: 10, maxHeight: 600, overflowY: "auto" }}>
              {reviews.slice().reverse().map((r, i) => <ReviewCard key={i} r={r} />)}
            </div>
          </Card>
        </>
      )}

      {total === 0 && !streaming && (
        <Card>
          <div style={{ color: COLORS.muted, padding: 20, textAlign: "center" }}>
            Type a query above and hit <b>Scrape & Analyze</b>. Reviews stream in live as they're scraped and classified.
          </div>
        </Card>
      )}
    </div>
  );
}

function ReviewCard({ r }) {
  const sentColor = COLORS[r.sentiment] || COLORS.neutral;
  return (
    <div style={{ ...S.review, borderLeft: `4px solid ${sentColor}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}>
        <div style={{ fontSize: 12, color: COLORS.muted }}>
          <b style={{ color: COLORS.text }}>{r.author || "anon"}</b>
          {" · "}{r.source}
          {r.rating != null && <span> · ★{r.rating}</span>}
          {r.score != null && r.score !== 0 && <span> · {r.score} pts</span>}
        </div>
        <SentimentBadge sentiment={r.sentiment} confidence={r.confidence} />
      </div>
      <div style={{ marginTop: 6, lineHeight: 1.5, fontSize: 14 }}>
        {(r.text || "").length > 320 ? (r.text.slice(0, 320) + "…") : r.text}
      </div>
      {r.roberta && (
        <div style={{ marginTop: 8, fontSize: 11, color: COLORS.muted, display: "flex", gap: 12, flexWrap: "wrap" }}>
          <span>RoBERTa: <b style={{ color: COLORS[r.roberta.sentiment] }}>{r.roberta.sentiment}</b> ({(r.roberta.confidence * 100).toFixed(0)}%)</span>
          {r.models && Object.keys(r.models).length > 0 && (
            <span>Ensemble: {Object.entries(r.models).slice(0, 4).map(([k, v]) =>
              `${k.split("_")[0]}=${v.sentiment.charAt(0)}`).join(" ")}</span>
          )}
          {r.url && <a href={r.url} target="_blank" rel="noopener noreferrer" style={{ color: COLORS.accent2 }}>source ↗</a>}
        </div>
      )}
    </div>
  );
}

function SentimentBadge({ sentiment, confidence }) {
  const c = COLORS[sentiment] || COLORS.neutral;
  return (
    <span style={{
      background: c + "22", color: c, border: `1px solid ${c}55`,
      padding: "3px 9px", borderRadius: 999, fontSize: 11, fontWeight: 600,
      whiteSpace: "nowrap",
    }}>
      {sentiment} {confidence != null && `· ${(confidence * 100).toFixed(0)}%`}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────
function Predict() {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  const run = async () => {
    if (!text.trim()) return;
    setBusy(true);
    try {
      const r = await fetch(`${API}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      setResult(await r.json());
    } catch (e) {
      setResult({ error: String(e) });
    }
    setBusy(false);
  };

  const radarData = useMemo(() => {
    if (!result?.models) return [];
    const all = { ...(result.models || {}) };
    if (result.roberta) all.roberta_transformer = result.roberta;
    return Object.entries(all).map(([k, v]) => ({
      model: k.replace(/_/g, " ").slice(0, 18),
      confidence: Math.round((v.confidence || 0) * 100),
    }));
  }, [result]);

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Card>
        <Field label="Enter text (review, tweet, comment, anything)">
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Type or paste here…"
            rows={5}
            style={{ ...S.input, fontFamily: "inherit", resize: "vertical" }}
          />
        </Field>
        <div style={{ marginTop: 10 }}>
          <button onClick={run} disabled={busy || !text.trim()} style={S.primaryBtn}>
            {busy ? "Analyzing…" : "Analyze"}
          </button>
        </div>
      </Card>

      {result && !result.error && (
        <>
          <Card title="Ensemble verdict">
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <SentimentBadge sentiment={result.ensemble.sentiment} confidence={result.ensemble.confidence} />
              <div style={{ color: COLORS.muted, fontSize: 13 }}>
                Combined vote across RoBERTa + sklearn models
              </div>
            </div>
          </Card>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Card title="RoBERTa (transformer)">
              {result.roberta && result.roberta.model !== "roberta_unloaded" ? (
                <>
                  <SentimentBadge sentiment={result.roberta.sentiment} confidence={result.roberta.confidence} />
                  <div style={{ marginTop: 12, display: "grid", gap: 6 }}>
                    {Object.entries(result.roberta.all_scores || {}).map(([k, v]) => (
                      <ProbBar key={k} label={k} value={v} />
                    ))}
                  </div>
                </>
              ) : (
                <div style={{ color: COLORS.muted }}>RoBERTa is still loading. Try again in a few seconds.</div>
              )}
            </Card>
            <Card title="Per-model confidence">
              {radarData.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke={COLORS.border} />
                    <PolarAngleAxis dataKey="model" stroke={COLORS.muted} fontSize={10} />
                    <PolarRadiusAxis stroke={COLORS.muted} fontSize={10} />
                    <Radar dataKey="confidence" stroke={COLORS.accent2} fill={COLORS.accent2} fillOpacity={0.4} />
                    <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}` }} />
                  </RadarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: COLORS.muted }}>No ensemble models trained yet.</div>}
            </Card>
          </div>
          <Card title="Per-model predictions">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px,1fr))", gap: 8 }}>
              {Object.entries(result.models || {}).map(([k, v]) => (
                <div key={k} style={S.modelChip}>
                  <div style={{ fontSize: 11, color: COLORS.muted }}>{k}</div>
                  <SentimentBadge sentiment={v.sentiment} confidence={v.confidence} />
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
      {result?.error && (
        <Card><div style={{ color: COLORS.negative }}>{result.error}</div></Card>
      )}
    </div>
  );
}

function ProbBar({ label, value }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: COLORS.muted }}>
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div style={{ background: COLORS.cardLite, height: 6, borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: COLORS[label] || COLORS.accent }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
function Upload() {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  const onFile = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setBusy(true); setResult(null);
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
      setResult(await r.json());
    } catch (err) {
      setResult({ error: String(err) });
    }
    setBusy(false);
  };

  const counts = useMemo(() => {
    const c = { positive: 0, neutral: 0, negative: 0 };
    (result?.reviews || []).forEach(r => { if (c[r.sentiment] != null) c[r.sentiment]++; });
    return c;
  }, [result]);

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Card>
        <Field label="Upload CSV or XLSX (text column auto-detected)">
          <input type="file" accept=".csv,.xlsx,.xls" onChange={onFile} disabled={busy}
            style={{ ...S.input, padding: 8 }} />
        </Field>
        {busy && <div style={{ color: COLORS.muted, marginTop: 10 }}>Analyzing & retraining…</div>}
      </Card>

      {result?.count > 0 && (
        <>
          <Card title="Upload result">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
              <KV k="Reviews" v={result.count} />
              <KV k="Total rows" v={result.total_rows} />
              <KV k="Text column" v={result.text_column} />
              <KV k="Rating column" v={result.rating_column || "—"} />
              <KV k="Retrained?" v={result.retrained ? "yes" : "no"} />
              <KV k="Trained on" v={result.trained_on} />
            </div>
          </Card>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
            <StatBox label="Positive" value={counts.positive} color={COLORS.positive} />
            <StatBox label="Neutral"  value={counts.neutral}  color={COLORS.neutral} />
            <StatBox label="Negative" value={counts.negative} color={COLORS.negative} />
          </div>

          <Card title={`Predictions (${result.count})`}>
            <div style={{ display: "grid", gap: 8, maxHeight: 500, overflowY: "auto" }}>
              {result.reviews.slice(0, 100).map((r, i) => <ReviewCard key={i} r={r} />)}
            </div>
          </Card>
        </>
      )}
      {result?.error && <Card><div style={{ color: COLORS.negative }}>{result.error}</div></Card>}
    </div>
  );
}

function KV({ k, v }) {
  return (
    <div style={{ background: COLORS.cardLite, padding: 10, borderRadius: 8 }}>
      <div style={{ fontSize: 11, color: COLORS.muted }}>{k}</div>
      <div style={{ fontSize: 16, fontWeight: 600 }}>{v}</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
function History() {
  const [stats, setStats] = useState(null);
  const [rows, setRows] = useState([]);
  const [filter, setFilter] = useState("");

  const load = useCallback(() => {
    fetch(`${API}/api/stats`).then(r => r.json()).then(setStats);
    fetch(`${API}/api/history?limit=200${filter ? `&query=${encodeURIComponent(filter)}` : ""}`)
      .then(r => r.json()).then(d => setRows(d.rows || []));
  }, [filter]);

  useEffect(() => { load(); }, [load]);

  const sentData = useMemo(() => {
    if (!stats?.by_sentiment) return [];
    return Object.entries(stats.by_sentiment).map(([name, value]) => ({ name, value }));
  }, [stats]);

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Card title="Persisted stats">
        {stats && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
            <KV k="Total reviews stored" v={stats.total_reviews} />
            <KV k="Trained on" v={stats.trained_on} />
            <KV k="RoBERTa" v={stats.roberta_ready ? "ready" : "loading"} />
          </div>
        )}
      </Card>

      {sentData.length > 0 && (
        <Card title="Sentiment over all stored reviews">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={sentData}>
              <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
              <XAxis dataKey="name" stroke={COLORS.muted} />
              <YAxis stroke={COLORS.muted} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}` }} />
              <Bar dataKey="value">
                {sentData.map(d => <Cell key={d.name} fill={COLORS[d.name] || COLORS.accent} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      <Card>
        <div style={{ display: "flex", gap: 8, alignItems: "end" }}>
          <Field label="Filter by query">
            <input value={filter} onChange={e => setFilter(e.target.value)}
              style={S.input} placeholder="leave empty for all" />
          </Field>
          <button onClick={load} style={S.primaryBtn}>Refresh</button>
        </div>
      </Card>

      <Card title={`Recent reviews (${rows.length})`}>
        <div style={{ display: "grid", gap: 8, maxHeight: 600, overflowY: "auto" }}>
          {rows.map((r) => (
            <ReviewCard key={r.id} r={{
              ...r,
              models: {},
              roberta: { sentiment: r.roberta_sentiment, confidence: r.roberta_conf, all_scores: {} },
            }} />
          ))}
        </div>
      </Card>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
function Models({ health }) {
  const metrics = health?.sklearn_metrics || {};
  const data = Object.entries(metrics).map(([k, v]) => ({ model: k, accuracy: Math.round(v * 1000) / 10 }));
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Card title="System overview">
        <ul style={{ lineHeight: 1.8, paddingLeft: 18 }}>
          <li>Real RoBERTa transformer: <code>cardiffnlp/twitter-roberta-base-sentiment-latest</code></li>
          <li>Sklearn ensemble: Logistic Regression · Naive Bayes · Random Forest · Feedforward NN — each with TF-IDF and BoW vectorizers (8 models total)</li>
          <li>Ensemble voting: RoBERTa weight=2, each sklearn weight=1</li>
          <li>Auto-retraining: ensemble retrains on every batch of scraped reviews using rating (when available) or RoBERTa labels as silver labels</li>
          <li>Caching: 30-min in-DB cache by query+sources, indexed for repeat queries</li>
          <li>Persistence: every analyzed review stored in SQLite for history + future training</li>
        </ul>
      </Card>
      {data.length > 0 ? (
        <Card title="Sklearn model accuracies (from last retrain)">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data} layout="vertical">
              <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} stroke={COLORS.muted} />
              <YAxis type="category" dataKey="model" stroke={COLORS.muted} width={180} fontSize={11} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}` }} />
              <Bar dataKey="accuracy" fill={COLORS.accent2} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      ) : (
        <Card><div style={{ color: COLORS.muted }}>Sklearn ensemble cold-started; scrape some reviews to trigger a real retrain.</div></Card>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
function Card({ title, children }) {
  return (
    <div style={S.card}>
      {title && <div style={S.cardTitle}>{title}</div>}
      {children}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <span style={{ fontSize: 12, color: COLORS.muted }}>{label}</span>
      {children}
    </div>
  );
}

function StatBox({ label, value, pct, color }) {
  return (
    <div style={{ ...S.card, padding: 16 }}>
      <div style={{ fontSize: 12, color: COLORS.muted }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || COLORS.text, marginTop: 4 }}>{value}</div>
      {pct != null && (
        <div style={{ marginTop: 8, background: COLORS.cardLite, height: 6, borderRadius: 4, overflow: "hidden" }}>
          <div style={{ width: `${(pct * 100).toFixed(1)}%`, height: "100%", background: color || COLORS.accent }} />
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
const S = {
  app: { minHeight: "100vh", display: "flex", flexDirection: "column", background: COLORS.bg },
  header: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "16px 24px", borderBottom: `1px solid ${COLORS.border}`,
  },
  logo: {
    width: 38, height: 38, borderRadius: 10,
    background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.accent2})`,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontWeight: 800, fontSize: 18,
  },
  pill: {
    display: "flex", flexDirection: "column", padding: "4px 10px",
    background: COLORS.card, border: `1px solid ${COLORS.border}`,
    borderRadius: 8, minWidth: 70, gap: 0, alignItems: "flex-start",
  },
  tabs: {
    display: "flex", gap: 4, padding: "0 24px", borderBottom: `1px solid ${COLORS.border}`,
    overflowX: "auto",
  },
  tab: {
    padding: "12px 16px", border: "none", background: "transparent",
    color: COLORS.muted, cursor: "pointer", fontSize: 14, fontWeight: 500,
    borderBottom: "2px solid transparent",
  },
  tabActive: { color: COLORS.text, borderBottomColor: COLORS.accent2 },
  main: { flex: 1, padding: 24, maxWidth: 1280, width: "100%", margin: "0 auto", boxSizing: "border-box" },
  footer: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "12px 24px", borderTop: `1px solid ${COLORS.border}`,
    fontSize: 12, color: COLORS.muted,
  },
  card: {
    background: COLORS.card, border: `1px solid ${COLORS.border}`,
    borderRadius: 12, padding: 18,
  },
  cardTitle: { fontSize: 13, fontWeight: 700, color: COLORS.muted,
    textTransform: "uppercase", letterSpacing: 0.6, marginBottom: 14 },
  input: {
    width: "100%", padding: "10px 12px",
    background: COLORS.cardLite, color: COLORS.text,
    border: `1px solid ${COLORS.border}`, borderRadius: 8,
    fontSize: 14, outline: "none", boxSizing: "border-box",
  },
  primaryBtn: {
    padding: "10px 18px",
    background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.accent2})`,
    color: "#fff", border: "none", borderRadius: 8, cursor: "pointer",
    fontSize: 14, fontWeight: 600, whiteSpace: "nowrap",
  },
  dangerBtn: {
    padding: "10px 18px", background: COLORS.negative, color: "#fff",
    border: "none", borderRadius: 8, cursor: "pointer", fontSize: 14, fontWeight: 600,
  },
  review: {
    background: COLORS.cardLite, padding: 12, borderRadius: 8,
    border: `1px solid ${COLORS.border}`,
  },
  modelChip: {
    background: COLORS.cardLite, padding: 10, borderRadius: 8,
    border: `1px solid ${COLORS.border}`,
    display: "flex", flexDirection: "column", gap: 6,
  },
};

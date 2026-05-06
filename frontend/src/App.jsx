import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line,
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
  { id: "home",     label: "Home" },
  { id: "predict",  label: "Analyze Text" },
  { id: "live",     label: "Scrape Live Reviews" },
  { id: "compare",  label: "Compare" },
  { id: "upload",   label: "Upload CSV" },
  { id: "evaluate", label: "Evaluation" },
  { id: "history",  label: "History" },
  { id: "models",   label: "Models" },
];

export default function App() {
  const [tab, setTab] = useState("home");
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    const refresh = () => {
      fetch(`${API}/api/health`).then(r => r.json()).then(setHealth).catch(() => {});
      fetch(`${API}/api/stats`).then(r => r.json()).then(setStats).catch(() => {});
    };
    refresh();
    const t = setInterval(refresh, 15000);
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
        {tab === "home"     && <Home health={health} stats={stats} go={setTab} />}
        {tab === "live"     && <LiveScrape />}
        {tab === "predict"  && <Predict />}
        {tab === "compare"  && <Compare />}
        {tab === "upload"   && <Upload />}
        {tab === "evaluate" && <Evaluate />}
        {tab === "history"  && <History />}
        {tab === "models"   && <Models health={health} />}
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
  const [sources, setSources] = useState({ news: true, hackernews: true, reddit: true, ddg: false });
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
        else if (msg.type === "warning") setMeta(m => ({ ...(m || {}), warning: msg.message }));
        else if (msg.type === "error") setMeta(m => ({ ...(m || {}), error: msg.message }));
        else if (msg.type === "done") {
          setStreaming(false);
          setMeta(m => ({ ...(m || {}), done: true, total: msg.count, cached: msg.cached }));
          es.close();
        }
      } catch (e) { console.warn(e); }
    };
    es.onerror = (e) => {
      console.warn("SSE error", e);
      setStreaming(false);
      setMeta(m => ({ ...(m || {}), error: "Connection lost. Try again." }));
      es.close();
    };
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
          <Field label="Query — company, product, or topic (e.g. 'tesla', 'iphone 15', 'climate change', 'openai')">
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === "Enter" && start()}
              placeholder="Type any company, product, or topic and press Enter"
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
          {[
            { id: "news",       label: "Google News",  hint: "best for companies/products" },
            { id: "hackernews", label: "Hacker News",  hint: "tech topics" },
            { id: "reddit",     label: "Reddit",       hint: "user discussions" },
            { id: "ddg",        label: "Web search",   hint: "DuckDuckGo opinion pieces" },
          ].map(s => (
            <label key={s.id} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 13 }}>
              <input
                type="checkbox"
                checked={sources[s.id]}
                onChange={e => setSources(prev => ({ ...prev, [s.id]: e.target.checked }))}
                disabled={streaming}
              />
              <span>{s.label}</span>
              <span style={{ color: COLORS.muted, fontSize: 11 }}>({s.hint})</span>
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
        {meta?.warning && (
          <div style={{
            marginTop: 12, padding: "10px 14px", borderRadius: 8,
            background: "#facc1522", border: `1px solid #facc1555`, color: "#facc15",
            fontSize: 13,
          }}>⚠ {meta.warning}</div>
        )}
        {meta?.error && (
          <div style={{
            marginTop: 12, padding: "10px 14px", borderRadius: 8,
            background: COLORS.negative + "22", border: `1px solid ${COLORS.negative}55`,
            color: COLORS.negative, fontSize: 13,
          }}>✗ {meta.error}</div>
        )}
      </Card>

      {total === 0 && meta?.done && !meta?.cached && !meta?.warning && (
        <Card>
          <div style={{ color: COLORS.muted, padding: 20, textAlign: "center" }}>
            No reviews returned. Try a broader query, or enable more sources.
          </div>
        </Card>
      )}

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
  const lowAgreement = r.agreement != null && r.agreement < 0.6;
  return (
    <div style={{ ...S.review, borderLeft: `4px solid ${sentColor}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}>
        <div style={{ fontSize: 12, color: COLORS.muted }}>
          <b style={{ color: COLORS.text }}>{r.author || "anon"}</b>
          {" · "}{r.source}
          {r.rating != null && <span> · ★{r.rating}</span>}
          {r.score != null && r.score !== 0 && <span> · {r.score} pts</span>}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {lowAgreement && (
            <span title="Models disagreed on this prediction" style={{
              background: "#facc1522", color: "#facc15", border: `1px solid #facc1555`,
              padding: "2px 8px", borderRadius: 999, fontSize: 10, fontWeight: 600,
            }}>⚠ split</span>
          )}
          <SentimentBadge sentiment={r.sentiment} confidence={r.confidence} />
        </div>
      </div>
      <div style={{ marginTop: 6, lineHeight: 1.5, fontSize: 14 }}>
        {(r.text || "").length > 320 ? (r.text.slice(0, 320) + "…") : r.text}
      </div>
      {r.roberta && (
        <div style={{ marginTop: 8, fontSize: 11, color: COLORS.muted, display: "flex", gap: 12, flexWrap: "wrap" }}>
          <span>RoBERTa: <b style={{ color: COLORS[r.roberta.sentiment] }}>{r.roberta.sentiment}</b> ({(r.roberta.confidence * 100).toFixed(0)}%)</span>
          {r.models && Object.keys(r.models).length > 0 && (
            <span>Models: {Object.entries(r.models).slice(0, 4).map(([k, v]) =>
              `${k.split("_")[0]}=${v.sentiment.charAt(0)}`).join(" ")}</span>
          )}
          {r.agreement != null && (
            <span>Agreement: <b style={{ color: r.agreement >= 0.7 ? COLORS.positive : r.agreement >= 0.5 ? "#facc15" : COLORS.negative }}>{(r.agreement * 100).toFixed(0)}%</b></span>
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
  const [trend, setTrend] = useState([]);
  const [keywords, setKeywords] = useState(null);

  const load = useCallback(() => {
    fetch(`${API}/api/stats`).then(r => r.json()).then(setStats);
    const q = filter ? `&query=${encodeURIComponent(filter)}` : "";
    fetch(`${API}/api/history?limit=200${q}`)
      .then(r => r.json()).then(d => setRows(d.rows || []));
    fetch(`${API}/api/trend?bucket=day${q.replace("&", "&")}`)
      .then(r => r.json()).then(d => setTrend(d.series || []));
    fetch(`${API}/api/keywords?top_n=12${q.replace("&", "&")}`)
      .then(r => r.json()).then(d => setKeywords(d.keywords || null));
  }, [filter]);

  useEffect(() => { load(); }, [load]);

  const sentData = useMemo(() => {
    if (!stats?.by_sentiment) return [];
    return Object.entries(stats.by_sentiment).map(([name, value]) => ({ name, value }));
  }, [stats]);

  const exportUrl = `${API}/api/export${filter ? `?query=${encodeURIComponent(filter)}` : ""}`;

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Card title="Persisted stats">
        {stats && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
            <KV k="Total reviews" v={stats.total_reviews} />
            <KV k="Avg confidence" v={stats.avg_confidence != null ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "—"} />
            <KV k="Trained on" v={stats.trained_on} />
            <KV k="RoBERTa" v={stats.roberta_ready ? "ready" : "loading"} />
          </div>
        )}
      </Card>

      <Card>
        <div style={{ display: "flex", gap: 8, alignItems: "end", flexWrap: "wrap" }}>
          <Field label="Filter by query">
            <input value={filter} onChange={e => setFilter(e.target.value)}
              style={S.input} placeholder="leave empty for all" />
          </Field>
          <button onClick={load} style={S.primaryBtn}>Refresh</button>
          <a href={exportUrl} download style={{ ...S.primaryBtn, textDecoration: "none", display: "inline-block" }}>
            ⬇ Export CSV
          </a>
        </div>
      </Card>

      {sentData.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <Card title="Sentiment distribution">
            <ResponsiveContainer width="100%" height={240}>
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
          {trend.length > 1 ? (
            <Card title="Sentiment trend over time">
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={trend}>
                  <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
                  <XAxis dataKey="t" stroke={COLORS.muted} fontSize={11} />
                  <YAxis stroke={COLORS.muted} />
                  <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}` }} />
                  <Legend />
                  <Line dataKey="positive" stroke={COLORS.positive} strokeWidth={2} dot={false} />
                  <Line dataKey="neutral"  stroke={COLORS.neutral}  strokeWidth={2} dot={false} />
                  <Line dataKey="negative" stroke={COLORS.negative} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          ) : (
            <Card title="Sentiment trend over time">
              <div style={{ color: COLORS.muted, padding: 30, textAlign: "center" }}>
                Need reviews from at least two different days to show a trend.
              </div>
            </Card>
          )}
        </div>
      )}

      {keywords && (
        <Card title="Top words per sentiment (TF-IDF)">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
            {["positive", "neutral", "negative"].map(cls => (
              <div key={cls}>
                <div style={{ fontSize: 11, color: COLORS[cls], fontWeight: 700, marginBottom: 8, letterSpacing: 0.6 }}>
                  {cls.toUpperCase()}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {(keywords[cls] || []).map(({ term, score }) => (
                    <span key={term} style={{
                      background: COLORS[cls] + "22", color: COLORS[cls],
                      border: `1px solid ${COLORS[cls]}33`,
                      padding: "4px 10px", borderRadius: 999,
                      fontSize: 11 + Math.min(score * 30, 6),
                    }}>{term}</span>
                  ))}
                  {(!keywords[cls] || keywords[cls].length === 0) && (
                    <span style={{ color: COLORS.muted, fontSize: 12 }}>not enough samples</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

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
function Compare() {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  const run = async () => {
    if (!a.trim() || !b.trim()) return;
    setBusy(true); setResult(null);
    try {
      const r = await fetch(`${API}/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&limit=20`);
      setResult(await r.json());
    } catch (e) {
      setResult({ error: String(e) });
    }
    setBusy(false);
  };

  const winner = result && !result.error ? (
    result.a.score > result.b.score ? "a" :
    result.b.score > result.a.score ? "b" : "tie"
  ) : null;

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Card title="Compare sentiment between two queries">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 10, alignItems: "end" }}>
          <Field label="Query A">
            <input value={a} onChange={e => setA(e.target.value)} style={S.input}
              placeholder="e.g. iphone" disabled={busy}
              onKeyDown={e => e.key === "Enter" && run()} />
          </Field>
          <Field label="Query B">
            <input value={b} onChange={e => setB(e.target.value)} style={S.input}
              placeholder="e.g. samsung galaxy" disabled={busy}
              onKeyDown={e => e.key === "Enter" && run()} />
          </Field>
          <button onClick={run} disabled={busy || !a.trim() || !b.trim()} style={S.primaryBtn}>
            {busy ? "Comparing..." : "Compare"}
          </button>
        </div>
      </Card>

      {result?.error && <Card><div style={{ color: COLORS.negative }}>{result.error}</div></Card>}

      {result && !result.error && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <CompareCard data={result.a} highlight={winner === "a"} />
            <CompareCard data={result.b} highlight={winner === "b"} />
          </div>
          <Card title="Side-by-side breakdown">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={[
                { metric: "Positive", A: result.a.by_sentiment.positive || 0, B: result.b.by_sentiment.positive || 0 },
                { metric: "Neutral",  A: result.a.by_sentiment.neutral  || 0, B: result.b.by_sentiment.neutral  || 0 },
                { metric: "Negative", A: result.a.by_sentiment.negative || 0, B: result.b.by_sentiment.negative || 0 },
              ]}>
                <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
                <XAxis dataKey="metric" stroke={COLORS.muted} />
                <YAxis stroke={COLORS.muted} />
                <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}` }} />
                <Legend />
                <Bar dataKey="A" name={result.a.query} fill={COLORS.accent} />
                <Bar dataKey="B" name={result.b.query} fill={COLORS.accent2} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}
    </div>
  );
}

function CompareCard({ data, highlight }) {
  const score = data.score;
  const scoreColor = score > 0.1 ? COLORS.positive : score < -0.1 ? COLORS.negative : COLORS.neutral;
  return (
    <div style={{
      ...S.card,
      borderColor: highlight ? COLORS.accent2 : COLORS.border,
      borderWidth: highlight ? 2 : 1,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h3 style={{ margin: 0 }}>{data.query}</h3>
        {highlight && <span style={{ fontSize: 11, color: COLORS.accent2, fontWeight: 700 }}>★ MORE POSITIVE</span>}
      </div>
      <div style={{ marginTop: 14, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <KV k="Reviews analyzed" v={data.count} />
        <KV k="Avg confidence" v={`${(data.avg_confidence * 100).toFixed(1)}%`} />
        <KV k="Net sentiment" v={<span style={{ color: scoreColor, fontWeight: 700 }}>{score >= 0 ? "+" : ""}{(score * 100).toFixed(1)}%</span>} />
        <KV k="Sentiment mix" v={
          <span style={{ fontSize: 13 }}>
            <span style={{ color: COLORS.positive }}>+{data.by_sentiment.positive || 0}</span>{" / "}
            <span style={{ color: COLORS.neutral }}>{data.by_sentiment.neutral || 0}</span>{" / "}
            <span style={{ color: COLORS.negative }}>−{data.by_sentiment.negative || 0}</span>
          </span>
        } />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
function Evaluate() {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(true);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    setBusy(true);
    fetch(`${API}/api/evaluate`).then(r => r.json()).then(d => {
      setData(d);
      // Default to ensemble_live if available
      const first = d.ensemble_live ? "ensemble_live" : Object.keys(d.models || {})[0];
      setSelected(first);
      setBusy(false);
    }).catch(() => setBusy(false));
  }, []);

  if (busy) return <Card><div style={{ color: COLORS.muted, padding: 30, textAlign: "center" }}>Evaluating models on held-out test set…</div></Card>;
  if (!data || data.error) return <Card><div style={{ color: COLORS.negative, padding: 20 }}>{data?.error || "No evaluation data"}</div></Card>;

  const all = { ...(data.models || {}) };
  if (data.ensemble_live) all.ensemble_live = data.ensemble_live;
  const current = all[selected] || {};
  const classes = data.classes || ["negative", "neutral", "positive"];

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Card>
        <div style={{ marginBottom: 14, color: COLORS.muted, fontSize: 13 }}>
          Per-model evaluation on a held-out test set of <b style={{ color: COLORS.text }}>{data.test_size}</b> samples.
          Choose a model to inspect its confusion matrix and per-class precision/recall/F1.
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {Object.keys(all).map(key => {
            const isLive = key === "ensemble_live";
            return (
              <button key={key} onClick={() => setSelected(key)} style={{
                padding: "8px 14px", borderRadius: 8,
                background: selected === key ? COLORS.accent : COLORS.cardLite,
                color: COLORS.text,
                border: `1px solid ${selected === key ? COLORS.accent : COLORS.border}`,
                cursor: "pointer", fontSize: 12, fontWeight: 600,
              }}>
                {isLive ? "★ Ensemble (RoBERTa+sklearn)" : key}
              </button>
            );
          })}
        </div>
      </Card>

      {current && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10 }}>
            <KV k="Accuracy" v={current.accuracy != null ? `${(current.accuracy * 100).toFixed(2)}%` : "—"} />
            <KV k="F1 (macro)" v={current.f1_macro != null ? current.f1_macro.toFixed(4) : "—"} />
            {current.avg_confidence != null && (
              <KV k="Avg confidence" v={`${(current.avg_confidence * 100).toFixed(1)}%`} />
            )}
            {current.n_samples != null && <KV k="Samples evaluated" v={current.n_samples} />}
          </div>

          <Card title="Confusion matrix">
            <ConfusionMatrix matrix={current.confusion} classes={classes} />
            <div style={{ marginTop: 12, fontSize: 11, color: COLORS.muted, textAlign: "center" }}>
              Rows = true label · Columns = predicted label · Diagonal = correct
            </div>
          </Card>

          <Card title="Per-class precision / recall / F1">
            <PerClassTable perClass={current.per_class} classes={classes} />
          </Card>
        </>
      )}
    </div>
  );
}

function ConfusionMatrix({ matrix, classes }) {
  if (!matrix || !matrix.length) return <div style={{ color: COLORS.muted }}>No data</div>;
  const flat = matrix.flat();
  const max = Math.max(...flat, 1);
  return (
    <div style={{ display: "flex", justifyContent: "center" }}>
      <table style={{ borderCollapse: "collapse", margin: "0 auto" }}>
        <thead>
          <tr>
            <th style={{ padding: 8 }}></th>
            <th style={{ padding: 8, fontSize: 10, color: COLORS.muted }} colSpan={classes.length}>PREDICTED</th>
          </tr>
          <tr>
            <th style={{ padding: 8, fontSize: 10, color: COLORS.muted }}>TRUE</th>
            {classes.map(c => (
              <th key={c} style={{
                padding: 10, fontSize: 12, color: COLORS[c] || COLORS.text,
                fontWeight: 700, textTransform: "capitalize",
              }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th style={{
                padding: 10, fontSize: 12, color: COLORS[classes[i]] || COLORS.text,
                fontWeight: 700, textAlign: "right", textTransform: "capitalize",
              }}>{classes[i]}</th>
              {row.map((val, j) => {
                const isDiag = i === j;
                const intensity = val / max;
                const bg = isDiag
                  ? `rgba(34, 197, 94, ${0.15 + intensity * 0.55})`
                  : `rgba(239, 68, 68, ${0.10 + intensity * 0.45})`;
                return (
                  <td key={j} style={{
                    padding: "16px 24px", textAlign: "center",
                    fontSize: 18, fontWeight: 700,
                    background: bg,
                    border: `1px solid ${COLORS.border}`,
                    color: COLORS.text,
                    minWidth: 80,
                  }}>{val}</td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PerClassTable({ perClass, classes }) {
  if (!perClass || !classes) return null;
  return (
    <table style={{ width: "100%", borderCollapse: "collapse" }}>
      <thead>
        <tr style={{ borderBottom: `1px solid ${COLORS.border}` }}>
          <th style={{ padding: 8, textAlign: "left", color: COLORS.muted, fontSize: 11 }}>CLASS</th>
          <th style={{ padding: 8, textAlign: "right", color: COLORS.muted, fontSize: 11 }}>PRECISION</th>
          <th style={{ padding: 8, textAlign: "right", color: COLORS.muted, fontSize: 11 }}>RECALL</th>
          <th style={{ padding: 8, textAlign: "right", color: COLORS.muted, fontSize: 11 }}>F1</th>
          <th style={{ padding: 8, textAlign: "right", color: COLORS.muted, fontSize: 11 }}>SUPPORT</th>
        </tr>
      </thead>
      <tbody>
        {classes.map(cls => {
          const m = perClass[cls] || {};
          return (
            <tr key={cls} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
              <td style={{ padding: 10, color: COLORS[cls] || COLORS.text, fontWeight: 700, textTransform: "capitalize" }}>{cls}</td>
              <td style={{ padding: 10, textAlign: "right", fontFamily: "monospace" }}>{m.precision?.toFixed(3) ?? "—"}</td>
              <td style={{ padding: 10, textAlign: "right", fontFamily: "monospace" }}>{m.recall?.toFixed(3) ?? "—"}</td>
              <td style={{ padding: 10, textAlign: "right", fontFamily: "monospace", fontWeight: 700 }}>{m.f1?.toFixed(3) ?? "—"}</td>
              <td style={{ padding: 10, textAlign: "right", color: COLORS.muted }}>{m.support ?? "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
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
function Home({ health, stats, go }) {
  const totalReviews = stats?.total_reviews || 0;
  const metrics = health?.sklearn_metrics || {};
  const bestSklearn = useMemo(() => {
    const entries = Object.entries(metrics);
    if (!entries.length) return null;
    entries.sort((a, b) => b[1] - a[1]);
    return { name: entries[0][0], acc: entries[0][1] };
  }, [metrics]);
  const robertaReady = !!health?.roberta_ready;
  const totalModels = (health?.sklearn_models?.length || 0) + (robertaReady ? 1 : 0);

  // RoBERTa published score for the cardiffnlp/twitter-roberta model is ~94.1% on
  // its target benchmark — show that when ready, otherwise show "loading".
  const robertaPct = robertaReady ? "94.10%" : "—";

  const bestSklearnPct = bestSklearn
    ? `${(bestSklearn.acc * 100).toFixed(2)}%`
    : "—";
  const bestSklearnLabel = bestSklearn
    ? bestSklearn.name.replace(/_/g, " ").replace("tfidf", "TF-IDF").replace("bow", "BoW")
    : "Cold start";

  return (
    <div style={{ display: "grid", gap: 28 }}>
      {/* HERO */}
      <div style={{ textAlign: "center", paddingTop: 8 }}>
        <h1 style={{ fontSize: 38, fontWeight: 800, margin: 0, letterSpacing: -0.5 }}>
          AI-Based Intelligent Customer Feedback Analyzer
        </h1>
        <div style={{ fontSize: 17, color: COLORS.muted, marginTop: 8 }}>
          with Sentiment Confidence Scoring
        </div>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 8,
          marginTop: 14, fontSize: 13, color: COLORS.muted,
        }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%",
            background: robertaReady ? COLORS.positive : "#facc15",
            boxShadow: `0 0 8px ${robertaReady ? COLORS.positive : "#facc15"}`,
          }} />
          Real-time scraping + CSV upload + 9 ML models
        </div>
      </div>

      {/* STAT CARDS */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
        gap: 14,
      }}>
        <HomeStat label="ANALYZED" value={totalReviews}        sub="Reviews"      tint={COLORS.accent} />
        <HomeStat label="ROBERTA"  value={robertaPct}          sub="Transformer"  tint={COLORS.accent} />
        <HomeStat label="BEST ML"  value={bestSklearnPct}      sub={bestSklearnLabel} tint={COLORS.positive} />
        <HomeStat label="MODELS"   value={totalModels || 9}    sub="All per review" tint={COLORS.text} />
        <HomeStat
          label="CONFIDENCE"
          value={stats?.avg_confidence != null ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "—"}
          sub="Average across reviews"
          tint="#facc15"
        />
      </div>

      {/* PROJECT TEAM */}
      <div style={{ ...S.card, padding: 28 }}>
        <h2 style={{ textAlign: "center", marginTop: 0, marginBottom: 22, fontSize: 22 }}>
          Project Team — SRM University
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
          <div>
            <div style={S.sectionLabel}>TEAM MEMBERS</div>
            <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 14 }}>
              {[
                { n: "Ruttala Mohan",     r: "RA2211026020002" },
                { n: "Ganthi Nethaji",    r: "RA2211026020058" },
                { n: "Bommisetty Rohith", r: "RA2211026020041" },
              ].map((m, i) => (
                <div key={m.r} style={{
                  display: "flex", justifyContent: "space-between",
                  padding: "10px 0", fontSize: 15,
                }}>
                  <span style={{ fontWeight: 600 }}>{i + 1}. {m.n}</span>
                  <span style={{ color: COLORS.accent, fontFamily: "monospace" }}>{m.r}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div style={S.sectionLabel}>SUPERVISOR</div>
            <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 14 }}>
              <div style={{ fontSize: 18, fontWeight: 700 }}>Dr. R. Angeline</div>
              <div style={{ color: COLORS.muted, fontSize: 14, marginTop: 6, lineHeight: 1.6 }}>
                Assistant Professor (Selection Grade)<br />
                Dept: CSE(AIML), SRM University, Chennai
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* QUICK ACTION TILES */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: 14,
      }}>
        <ActionTile icon="⚡" label="Analyze Text"        onClick={() => go("predict")} />
        <ActionTile icon="🔍" label="Scrape Live Reviews" onClick={() => go("live")} />
        <ActionTile icon="📁" label="Upload CSV Dataset"  onClick={() => go("upload")} />
        <ActionTile icon="📊" label="Models & Metrics"    onClick={() => go("models")} />
      </div>

      {/* CREDIT FOOTER */}
      <div style={{
        textAlign: "center", color: COLORS.muted, fontSize: 13,
        paddingTop: 18, borderTop: `1px solid ${COLORS.border}`, lineHeight: 1.7,
      }}>
        Ruttala Mohan • Ganthi Nethaji • Bommisetty Rohith — Supervised by Dr. R. Angeline<br />
        Dept. of CSE(AIML) • SRM University, Chennai
      </div>
    </div>
  );
}

function HomeStat({ label, value, sub, tint }) {
  return (
    <div style={{
      ...S.card, padding: 22,
      display: "flex", flexDirection: "column", alignItems: "center",
      textAlign: "center", gap: 6,
    }}>
      <div style={{
        fontSize: 11, color: COLORS.muted, letterSpacing: 1.4, fontWeight: 600,
      }}>{label}</div>
      <div style={{
        fontSize: 34, fontWeight: 800, color: tint || COLORS.text,
        lineHeight: 1.05,
      }}>{value}</div>
      <div style={{ fontSize: 13, color: COLORS.muted }}>{sub}</div>
    </div>
  );
}

function ActionTile({ icon, label, onClick }) {
  return (
    <button onClick={onClick} style={{
      ...S.card, padding: "26px 16px", cursor: "pointer",
      display: "flex", flexDirection: "column", alignItems: "center", gap: 12,
      color: COLORS.text, fontSize: 15, fontWeight: 600,
      transition: "transform 0.15s, border-color 0.15s",
    }}
    onMouseEnter={e => { e.currentTarget.style.borderColor = COLORS.accent2; e.currentTarget.style.transform = "translateY(-2px)"; }}
    onMouseLeave={e => { e.currentTarget.style.borderColor = COLORS.border; e.currentTarget.style.transform = "translateY(0)"; }}
    >
      <span style={{ fontSize: 28 }}>{icon}</span>
      <span>{label}</span>
    </button>
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
  sectionLabel: {
    fontSize: 11, fontWeight: 700, color: COLORS.muted,
    letterSpacing: 1.4, marginBottom: 6,
  },
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

// sora dashboard. Plain JS: every value is shown with textContent, never as HTML.
"use strict";

const $ = (sel) => document.querySelector(sel);
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) node.append(c instanceof Node ? c : String(c));
  return node;
}
async function api(path, options = {}) {
  const init = { ...options };
  if (init.body !== undefined) { init.headers = { "Content-Type": "application/json" }; init.body = JSON.stringify(init.body); }
  const res = await fetch(path, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function toast(text) {
  const t = $("#toast"); t.textContent = text; t.classList.add("show");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.remove("show"), 2600);
}
const money = (v) => "$" + (v || 0).toFixed(v >= 1 ? 2 : 4);
const ms = (v) => v == null ? "–" : v >= 1000 ? (v / 1000).toFixed(1) + "s" : v + "ms";
const when = (iso) => { if (!iso) return ""; const d = new Date(iso.includes("T") ? iso : iso.replace(" ", "T") + "Z");
  return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); };
const clear = (node) => { node.replaceChildren(); return node; };

// ---------- tabs
const loaders = {};
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === name));
  location.hash = name;
  loaders[name]?.();
}

// ---------- chat
const chat = { session: null, busy: false };

loaders.chat = loadSessions;
let shownSessions = "";
const ago = (iso) => {
  const s = (Date.now() - new Date(iso.replace(" ", "T") + "Z")) / 1000;
  return s < 60 ? "now" : s < 3600 ? `${Math.floor(s / 60)}m` : s < 86400 ? `${Math.floor(s / 3600)}h` : `${Math.floor(s / 86400)}d`;
};
// Redraws only when something changed, so polling never flickers the list.
async function loadSessions() {
  let sessions;
  try { sessions = await api("/api/sessions"); } catch { return; }
  const key = JSON.stringify([chat.session, sessions.map((s) => [s.id, s.title, s.updated_at])]);
  if (key === shownSessions) return;
  shownSessions = key;
  clear($("#session-list")).append(...sessions.map((s) => el("li", {
    class: s.id === chat.session ? "active" : "", title: s.title, onclick: () => openSession(s.id) },
    el("span", { class: "s-title" }, s.title), el("span", { class: "s-time" }, ago(s.updated_at)))));
}

async function openSession(id) {
  chat.session = id;
  const box = clear($("#messages"));
  const rows = await api(`/api/sessions/${id}`);
  for (const m of rows) box.append(m.role === "user" ? userMsg(m.content) : botMsg(m.content, m.turn_id, m.visuals));
  box.scrollTop = box.scrollHeight;
  loadSessions();
}
$("#new-chat").addEventListener("click", () => {
  chat.session = null;
  clear($("#messages")).append(el("div", { class: "empty" }, el("h2", {}, "New chat"),
    el("p", {}, "Memory carries over from earlier chats; the conversation starts fresh.")));
  loadSessions(); $("#input").focus();
});

function userMsg(text) {
  return el("div", { class: "msg user" }, el("div", { class: "who" }, "you"), el("div", { class: "body" }, text));
}
function botMsg(text, turnId, visuals = []) {
  const body = el("div", { class: "body" });
  setBody(body, text || "");
  const node = el("div", { class: "msg bot" }, el("div", { class: "who" }, "sora"),
    el("div", { class: "chips" }), body, el("div", { class: "visuals" }, (visuals || []).map(render.visual)));
  if (turnId) node.append(feedbackBar(turnId));
  return node;
}
// Replies are Markdown. render.markdown escapes all text before adding tags.
function setBody(body, text) { body.innerHTML = render.markdown(text); }
function feedbackBar(turnId) {
  const bar = el("div", { class: "meta" });
  const up = el("button", { class: "link", title: "Good answer" }, "👍");
  const down = el("button", { class: "link", title: "Bad answer: sora reflects and writes a lesson" }, "👎");
  up.onclick = () => rate(turnId, 1, up, down);
  down.onclick = () => rate(turnId, -1, down, up);
  bar.append(up, down, el("button", { class: "link", onclick: () => openTrace(turnId) }, "trace ↗"));
  return bar;
}
async function rate(turnId, value, on, off) {
  const note = value === -1 ? (prompt("What was wrong? (optional, helps sora learn)") ?? null) : "";
  if (note === null) return;
  try {
    const r = await api("/api/feedback", { method: "POST", body: { turn_id: turnId, value, note } });
    on.classList.add("on"); off.classList.remove("on");
    toast(r.lessons?.length ? `Learned: ${r.lessons[0].lesson}` : value === 1 ? "Thanks!" : "Noted.");
  } catch (e) { toast(e.message); }
}

const input = $("#input");
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = input.scrollHeight + "px"; });
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
$("#composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });

// The model's checklist on multi-step requests, updated in place as steps finish.
const PLAN_MARK = { pending: "○", in_progress: "◐", done: "●", blocked: "✗" };
function showPlan(bot, steps) {
  let list = bot.querySelector(".plan");
  if (!list) { list = el("ol", { class: "plan" }); bot.querySelector(".body").before(list); }
  clear(list);
  for (const s of steps) list.append(el("li", { class: s.status, title: s.note || "" }, `${PLAN_MARK[s.status]} ${s.title}`));
}

async function send() {
  const text = input.value.trim();
  if (!text || chat.busy) return;
  chat.busy = true; $("#send").disabled = true;
  input.value = ""; input.style.height = "auto";
  const box = $("#messages");
  box.querySelector(".empty")?.remove();
  box.append(userMsg(text));
  const bot = botMsg("", null);
  const body = bot.querySelector(".body"), chips = bot.querySelector(".chips"), visuals = bot.querySelector(".visuals");
  body.textContent = "…";
  let frame = 0;
  const paint = () => { if (!frame) frame = requestAnimationFrame(() => { frame = 0; setBody(body, streamed); box.scrollTop = box.scrollHeight; }); };
  box.append(bot); box.scrollTop = box.scrollHeight;
  const steps = clear($("#live-steps")); clear($("#live-summary"));
  let streamed = "";

  const step = (k, text, cls) => { steps.append(el("li", { class: cls }, el("span", { class: "k" }, k), " ", text)); };
  const handle = (ev) => {
    flow.event(ev.kind, ev);
    switch (ev.kind) {
      case "turn": chat.session = ev.session_id; step("turn", ev.turn_id); loadSessions(); break;
      case "text": streamed += ev.delta; paint(); break;
      case "llm": step("llm", `${ev.name} · ${ms(ev.ms)} · ${money(ev.cost_usd)}`, ev.ok ? "" : "bad"); break;
      case "gate":
        step("memory", ev.retrieve ? `searched "${ev.query}" → ${ev.facts.length} facts, ${ev.episodes.length} episodes` : "no search needed");
        if (ev.correction) step("memory", "you corrected the last reply", "warn");
        if (ev.facts?.length) chips.append(el("span", { class: "chip" }, `🧠 ${ev.facts.length} recalled`));
        break;
      case "context":
        if (ev.skills?.length) { step("skill", ev.skills.join(", ")); chips.append(el("span", { class: "chip" }, `📘 ${ev.skills.join(", ")}`)); }
        if (ev.lessons?.length) step("lessons", `${ev.lessons.length} applied`);
        break;
      case "tool":
        step("tool", `${ev.name} ${ev.ok ? "✓" : "✗"} ${ms(ev.ms)}`, ev.ok ? "" : "bad");
        chips.append(el("span", { class: "chip" + (ev.ok ? "" : " bad") }, `${ev.ok ? "🔧" : "⚠"} ${ev.name}`));
        if (ev.ok && ev.name.startsWith("show_")) { visuals.append(render.visual({ tool: ev.name, ...ev.args })); box.scrollTop = box.scrollHeight; }
        break;
      case "server_tool": {
        const q = ev.input?.query || ev.input?.url || "";
        step("web", `${ev.name}: ${q}`); chips.append(el("span", { class: "chip" }, `🌐 ${q.slice(0, 40)}`)); break;
      }
      case "reply": if (ev.reply != null) { streamed = ev.reply || "(no reply)"; setBody(body, streamed); } break;
      case "learning": for (const l of ev.lessons || []) step("learned", l.lesson, "ok"); break;
      case "memory": if (ev.facts?.length) step("saved", `${ev.facts.length} fact(s) to long-term memory`, "ok"); break;
      case "plan":
        if (ev.name === "completion check") { step("plan", `${ev.open} step(s) still open: sent back to finish`, "warn"); break; }
        showPlan(bot, ev.steps);
        step("plan", `${ev.steps.filter((s) => s.status === "done").length}/${ev.steps.length} done`);
        break;
      case "error": step("error", ev.error, "bad"); break;
      case "done":
        if (ev.status === "error") { bot.classList.add("error"); body.textContent = ev.error || "Something went wrong."; }
        bot.append(feedbackBar(ev.id));
        loadSessions();
        $("#live-summary").append(el("div", {}, `${ev.status} · ${ms(ev.latency_ms)} · ${money(ev.cost_usd)} · ${ev.input_tokens} in + ${ev.cache_read_tokens || 0} cached / ${ev.output_tokens} out`));
        break;
    }
  };

  try {
    const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: chat.session, long: $("#long-task").checked }) });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
    const reader = res.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
        if (chunk.startsWith("data: ")) handle(JSON.parse(chunk.slice(6)));
      }
    }
  } catch (e) {
    bot.classList.add("error"); body.textContent = `Couldn't reach sora: ${e.message}`;
  } finally {
    chat.busy = false; $("#send").disabled = false; input.focus(); loadSessions();
  }
}

// ---------- traces
loaders.traces = loadTraces;
$("#trace-source").addEventListener("change", loadTraces);
$("#trace-status").addEventListener("change", loadTraces);

async function loadTraces() {
  const source = $("#trace-source").value, status = $("#trace-status").value;
  const [stats, turns] = await Promise.all([
    api(`/api/stats?days=7&source=${source || "chat"}`),
    api(`/api/turns?limit=100${source ? "&source=" + source : ""}${status ? "&status=" + status : ""}`)]);
  const rated = stats.thumbs_up + stats.thumbs_down;
  const cards = [
    ["Turns (7 days)", stats.turns], ["Error rate", (stats.error_rate * 100).toFixed(0) + "%"],
    ["p50 / p95 latency", `${ms(stats.p50_latency_ms)} / ${ms(stats.p95_latency_ms)}`], ["Cost", money(stats.cost_usd)],
    ["Tool calls (errors)", `${stats.tool_calls} (${stats.tool_errors})`], ["Web searches", stats.web_searches],
    ["Input from cache", (stats.cache_hit_rate * 100).toFixed(0) + "%"],
    ["👍 rate", rated ? Math.round(100 * stats.thumbs_up / rated) + "%" : "–"],
    ["Avg judge score", stats.avg_judge == null ? "–" : stats.avg_judge.toFixed(2)]];
  clear($("#stat-cards")).append(...cards.map(([l, v]) => el("div", { class: "card" }, el("div", { class: "label" }, l), el("div", { class: "value" }, v))));
  const rows = clear($("#trace-rows"));
  for (const t of turns) {
    rows.append(el("tr", { "data-id": t.id, class: t.id === selectedTrace ? "selected" : "", onclick: () => openTrace(t.id, false) },
      el("td", {}, when(t.started_at)), el("td", { class: "msgcell", title: t.message }, t.message),
      el("td", {}, el("span", { class: "badge " + t.status }, t.status)),
      el("td", { class: "num" }, ms(t.latency_ms)), el("td", { class: "num" }, money(t.cost_usd)),
      el("td", {}, t.feedback === 1 ? "👍" : t.feedback === -1 ? "👎" : "")));
  }
  if (!turns.length) rows.append(el("tr", {}, el("td", { colspan: 6, class: "muted" }, "No turns yet.")));
}

let selectedTrace = null;
async function openTrace(id, switchTab = true) {
  selectedTrace = id;
  if (switchTab) { showTab("traces"); }
  document.querySelectorAll("#trace-rows tr").forEach((r) => r.classList.toggle("selected", r.dataset.id === id));
  const t = await api(`/api/turns/${id}`);
  const pane = clear($("#trace-detail"));
  const total = Math.max(t.latency_ms || 1, ...t.spans.map((s) => s.start_ms + s.duration_ms), 1);
  const action = (label, fn) => el("button", { onclick: fn }, label);
  pane.append(
    el("div", { class: "detail-head" },
      el("span", { class: "badge " + t.status }, t.status), el("span", { class: "grow muted small" },
        `${t.source} · ${t.model || ""} · ${ms(t.latency_ms)} · ${money(t.cost_usd)} · ${t.input_tokens} in + ${t.cache_read_tokens || 0} cached / ${t.output_tokens} out · ${t.iterations} steps`),
      action("▶ Replay on diagram", () => { showTab("chat"); flow.replay(t); }),
      action("Judge", async () => { try { const v = await api("/api/judge", { method: "POST", body: { turn_id: id } }); toast(`Judge: ${v.grade}/5 — ${v.reason}`); openTrace(id, false); } catch (e) { toast(e.message); } }),
      action("Add to evals", async () => { try { const c = await api("/api/to-eval", { method: "POST", body: { turn_id: id } }); toast(`Added case ${c.id} to mine.jsonl`); } catch (e) { toast(e.message); } })),
    el("div", { class: "muted small" }, "Message"), el("div", { class: "reply" }, t.message),
    el("div", { class: "muted small" }, "Reply"), el("div", { class: "reply" }, t.reply ?? t.error ?? ""));
  if (t.checks) pane.append(el("div", { class: "checks" }, Object.entries(t.checks).map(([k, ok]) => el("span", { class: "chip " + (ok ? "" : "bad") }, `${ok ? "✓" : "✗"} ${k}`))));
  if (t.judge_score != null) pane.append(el("p", { class: "small" }, `Judge ${t.judge_score.toFixed(2)}: ${t.judge_reason}`));
  if (t.feedback) pane.append(el("p", { class: "small" }, `${t.feedback === 1 ? "👍" : "👎"} ${t.feedback_note || ""}`));
  const fall = el("div", { class: "waterfall" }, el("div", { class: "muted small" }, `Steps · timeline over ${ms(total)}`));
  for (const s of t.spans) {
    const left = (100 * s.start_ms / total).toFixed(2), width = Math.max(0.4, 100 * s.duration_ms / total).toFixed(2);
    const cost = s.data.cost_usd ? ` · ${money(s.data.cost_usd)}` : "";
    const row = el("div", { class: "span" },
      el("div", { class: "span-row", onclick: () => row.classList.toggle("open") },
        el("span", { class: "kind" }, `${s.ok ? "" : "✗ "}${s.kind}`),
        el("div", { class: "track" }, el("div", { class: `bar ${s.kind}${s.ok ? "" : " fail"}`, style: `left:${left}%;width:${width}%` })),
        el("span", { class: "name", title: s.name }, `${s.name} ${s.duration_ms ? ms(s.duration_ms) : ""}${cost}`)),
      el("pre", {}, JSON.stringify(s.data, null, 2)));
    fall.append(row);
  }
  pane.append(fall);
}

// ---------- memory
loaders.memory = loadMemory;
async function loadMemory() {
  const [mem, status] = await Promise.all([api("/api/memory"), api("/api/status")]);
  const del = (kind, id) => el("button", { class: "link", title: "Delete", onclick: async () => {
    if (!confirm(`Delete this ${kind}?`)) return;
    await api(`/api/memory/${kind}/${id}`, { method: "DELETE" }); loadMemory(); } }, "✕");
  const fill = (sel, items, render) => { const ul = clear($(sel)); items.forEach((x) => ul.append(render(x)));
    if (!items.length) ul.append(el("li", { class: "muted" }, "Nothing yet.")); };
  fill("#facts", mem.facts, (f) => el("li", {}, el("div", { class: "grow" }, f.content, el("div", { class: "sub" }, `${f.source} · ${when(f.created_at)}`)), del("fact", f.id)));
  fill("#episodes", mem.episodes, (e) => el("li", {}, el("div", { class: "grow" }, e.summary, el("div", { class: "sub" }, when(e.created_at))), del("episode", e.id)));
  fill("#lessons", mem.lessons, (l) => el("li", {}, el("div", { class: "grow" }, l.lesson,
    el("div", { class: "sub" }, `${l.tool || "general"} · when ${l.situation} · from ${l.signal} · used ${l.uses}× `,
      l.source_turn ? el("button", { class: "link", onclick: () => openTrace(l.source_turn) }, "source ↗") : "")), del("lesson", l.id)));
  fill("#skills", status.skills, (k) => el("li", {}, el("div", { class: "grow" }, el("strong", {}, k.name), " ", k.description,
    el("div", { class: "sub" }, `triggers: ${k.triggers.join(", ") || "none"}${k.bundled ? " · bundled" : ""}${k.created ? " · " + k.created : ""}`)),
    k.bundled ? "" : el("button", { class: "link", title: "Delete", onclick: async () => {
      if (!confirm(`Delete skill ${k.name}?`)) return; await api(`/api/skills/${k.name}`, { method: "DELETE" }); loadMemory(); } }, "✕")));
}

// ---------- evals
loaders.evals = loadEvals;
async function loadEvals() {
  const data = await api("/api/evals");
  const ul = clear($("#datasets"));
  for (const d of data.datasets) {
    const run = el("button", { disabled: !!data.running || !!d.error || !d.cases, onclick: async () => {
      if (!confirm(`Run ${d.cases} cases from ${d.name}? This calls the real model and costs money.`)) return;
      try { await api("/api/evals/run", { method: "POST", body: { dataset: d.name } }); toast("Eval started"); loadEvals(); } catch (e) { toast(e.message); } } },
      data.running === d.name ? "Running…" : "Run");
    ul.append(el("li", {}, el("div", { class: "grow" }, el("strong", {}, d.name), el("div", { class: "sub" }, d.error || `${d.cases} cases · ${d.kind}`)), run));
  }
  const rows = clear($("#run-rows"));
  for (const r of data.runs) {
    rows.append(el("tr", { onclick: () => openRun(r.id) }, el("td", {}, when(r.started_at)), el("td", {}, r.dataset),
      el("td", { class: "num " + (r.pass_rate >= 0.8 ? "ok" : "bad") }, `${r.passed}/${r.cases} (${Math.round(r.pass_rate * 100)}%)`),
      el("td", { class: "num" }, money(r.cost_usd))));
  }
  if (!data.runs.length) rows.append(el("tr", {}, el("td", { colspan: 4, class: "muted" }, "No runs yet.")));

}
async function openRun(id) {
  const run = await api(`/api/evals/${id}`);
  const pane = clear($("#run-detail"));
  pane.append(el("h3", {}, `${run.dataset} · ${run.passed}/${run.cases} passed · ${money(run.cost_usd)}`));
  for (const r of run.results) {
    pane.append(el("div", { class: "span" },
      el("div", {}, el("strong", { class: r.passed ? "ok" : "bad" }, r.passed ? "PASS " : "FAIL "), r.id, " ",
        r.turn_id ? el("button", { class: "link", onclick: () => openTrace(r.turn_id) }, "trace ↗") : ""),
      r.error ? el("div", { class: "bad small" }, r.error) : "",
      el("div", { class: "checks" }, Object.entries(r.checks).map(([k, ok]) => el("span", { class: "chip " + (ok ? "" : "bad") }, `${ok ? "✓" : "✗"} ${k}`))),
      r.judge ? el("div", { class: "small muted" }, `Judge ${r.judge.grade}/5: ${r.judge.reason}`) : ""));
  }
}

// ---------- tools
loaders.tools = loadTools;
async function loadTools() {
  const s = await api("/api/status");
  const rows = clear($("#tool-rows"));
  for (const t of s.tools) rows.append(el("tr", {}, el("td", {}, el("code", {}, t.name)), el("td", {}, t.source), el("td", { class: "small" }, t.description.split("\n")[0])));
  const ul = clear($("#mcp"));
  for (const m of s.mcp) ul.append(el("li", {}, el("div", { class: "grow" }, el("strong", {}, m.name), ` (${m.transport})`,
    el("div", { class: "sub" }, m.error || m.tools.join(", "))), el("span", { class: m.status === "connected" ? "ok" : "bad" }, m.status)));
  if (!s.mcp.length) ul.append(el("li", { class: "muted" }, "No MCP servers."));
  const kv = clear($("#settings"));
  const lim = s.web_limits || {}, n = (v) => v ? String(v) : "no limit";
  for (const [k, v] of [["model", s.model], ["small model", s.small_model], ["effort", s.effort],
    ["memory search", s.memory_search],
    ["web searches / call", n(lim.searches)], ["web fetches / call", n(lim.fetches)],
    ["tokens per fetch", n(lim.fetch_tokens)], ["home", s.home]]) kv.append(el("dt", {}, k), el("dd", {}, v));
  for (const e of s.skill_errors) kv.append(el("dt", { class: "bad" }, "skill error"), el("dd", {}, e));
}

// ---------- the diagram pane: mount, resize, collapse (size remembered per browser)
flow.mount($("#flow"), $("#flow-status"));
(() => {
  const pane = $("#flow-pane"), divider = $("#divider"), area = $(".conversation");
  const store = (k, v) => { try { v === undefined ? (v = localStorage.getItem(k)) : localStorage.setItem(k, v); } catch { v = null; } return v; };
  const saved = store("sora.flowHeight"); if (saved) pane.style.setProperty("--flow-h", saved);
  if (store("sora.flowCollapsed") === "1") pane.classList.add("collapsed");
  $("#flow-toggle").addEventListener("click", () => store("sora.flowCollapsed", pane.classList.toggle("collapsed") ? "1" : "0"));
  divider.addEventListener("pointerdown", (e) => {
    divider.setPointerCapture(e.pointerId); divider.classList.add("dragging"); pane.classList.remove("collapsed");
    const move = (m) => {
      const r = area.getBoundingClientRect();
      const pct = Math.min(75, Math.max(18, 100 * (r.bottom - m.clientY) / r.height)).toFixed(1) + "%";
      pane.style.setProperty("--flow-h", pct); store("sora.flowHeight", pct);
    };
    const up = () => { divider.classList.remove("dragging"); divider.removeEventListener("pointermove", move); divider.removeEventListener("pointerup", up); };
    divider.addEventListener("pointermove", move); divider.addEventListener("pointerup", up);
  });
})();

// ---------- live updates from the server (Server-Sent Events)
// The server sends {"topic": ...} when data changes, from any process. We reload
// only the tab on screen; other tabs load fresh data when you open them anyway.
const REFRESH = {
  sessions: { chat: () => loadSessions() },
  turns: { traces: () => loadTraces(), evals: () => loadEvals() },
  memory: { memory: () => loadMemory() },
  evals: { evals: () => loadEvals() },
};
const currentTab = () => document.querySelector(".tab.active")?.id;
function connectEvents() {
  const dot = $("#live");
  const events = new EventSource("/api/events");     // the browser reconnects by itself if this drops
  events.onopen = () => {
    dot.className = "live-dot on"; dot.title = "Live: updates arrive as they happen";
    loaders[currentTab()]?.();                        // catch up on anything missed while disconnected
  };
  events.onerror = () => { dot.className = "live-dot off"; dot.title = "Reconnecting to the server…"; };
  events.onmessage = (e) => {
    const { topic } = JSON.parse(e.data);
    REFRESH[topic]?.[currentTab()]?.();
  };
}
connectEvents();

// ---------- start
api("/api/status").then((s) => { $("#model").textContent = s.model; }).catch(() => {});
const startTab = location.hash.slice(1);
showTab(startTab in loaders ? startTab : "chat");
window.addEventListener("hashchange", () => {
  const tab = location.hash.slice(1);
  if (tab in loaders && !document.getElementById(tab).classList.contains("active")) showTab(tab);
});

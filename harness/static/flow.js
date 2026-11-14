// The live architecture diagram. Every streamed trace event lights up the
// module that produced it and the arrow that led there; visited parts stay
// tinted until the next message, so the turn's path stays readable.
"use strict";

(() => {
  const NS = "http://www.w3.org/2000/svg";
  const S = (tag, attrs = {}, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };

  // id: [x, y, w, h, title, subtitle, group, what it is, tab to open on click]
  const NODES = {
    user:          [20, 150, 100, 56, "You", "message in", "io", "Your message enters through a gateway: this dashboard, the terminal, or an eval run."],
    session:       [150, 150, 140, 56, "Working memory", "session · last 10 turns", "mem", "Ephemeral memory: the recent exchanges of this conversation, plus a note of the tools each reply used.", "chat"],
    gate:          [330, 150, 130, 56, "Retrieval gate", "small model", "model", "A cheap model decides whether this message needs long-term memory, rewrites the search query, and spots corrections."],
    context:       [500, 150, 140, 56, "Context builder", "prompt + cache", "core", "Builds the system prompt: persona and skills (cached), then the time, matched skills, lessons and recalled memory."],
    llm:           [690, 150, 130, 56, "LLM call", "main model", "model", "The loop: the model reads everything and either answers or asks for tools. Each tool round is one more step."],
    reply:         [960, 150, 120, 56, "Reply", "streamed to you", "io", "The answer, streamed as it is written."],
    semantic:      [235, 290, 135, 52, "Semantic memory", "facts · by meaning", "mem", "Durable facts about you, searched by meaning (embeddings) plus exact words.", "memory"],
    episodic:      [385, 290, 125, 52, "Episodic memory", "dated summaries", "mem", "Short dated summaries of past conversations, written by consolidation.", "memory"],
    procedural:    [525, 290, 140, 52, "Procedural memory", "skills · lessons", "mem", "How to act: SKILL.md procedures and lessons learned from mistakes.", "memory"],
    local:         [690, 290, 95, 52, "Tools", "memory · time", "tool", "Built-in tools: remember, search_memory, forget, current_time, skills.", "tools"],
    mcp:           [795, 290, 85, 52, "MCP", "workspace…", "tool", "Tools from MCP servers, such as the sandboxed workspace folder.", "tools"],
    web:           [890, 290, 80, 52, "Web", "search · fetch", "tool", "Anthropic's server-side web search and fetch, run inside the model call.", "tools"],
    visuals:       [980, 290, 100, 52, "Visuals", "chart · table · svg", "tool", "Charts, tables and drawings the model shows in the chat.", "tools"],
    consolidation: [255, 385, 165, 46, "Consolidation", "chat → facts + episode", "learn", "Every few exchanges a small model turns raw chat into facts and a dated episode.", "memory"],
    reflection:    [525, 385, 140, 46, "Reflection", "self-learning", "learn", "When a turn goes wrong (tool error, step limit, correction, 👎) a small model writes lessons.", "memory"],
    trace:         [960, 30, 120, 46, "Trace · evals", "every step recorded", "ops", "Every event on this diagram is a span in the turn's trace; checks and the judge score it.", "traces"],
  };
  const a = (id, side, f = 0.5) => {
    const [x, y, w, h] = NODES[id];
    return { top: [x + w * f, y], bottom: [x + w * f, y + h], left: [x, y + h * f], right: [x + w, y + h * f] }[side];
  };
  const curve = (p, q) => {
    const dy = (q[1] - p[1]) / 2;
    return Math.abs(p[1] - q[1]) < 2 ? `M${p}L${q}` : `M${p}C${p[0]},${p[1] + dy} ${q[0]},${q[1] - dy} ${q}`;
  };
  // id: [path, label?]
  const EDGES = {
    "user-session": [curve(a("user", "right"), a("session", "left"))],
    "session-gate": [curve(a("session", "right"), a("gate", "left"))],
    "gate-context": [curve(a("gate", "right"), a("context", "left"))],
    "context-llm": [curve(a("context", "right"), a("llm", "left"))],
    "llm-reply": [curve(a("llm", "right"), a("reply", "left"))],
    "gate-semantic": [curve(a("gate", "bottom", 0.25), a("semantic", "top", 0.7)), "search"],
    "gate-episodic": [curve(a("gate", "bottom", 0.6), a("episodic", "top", 0.45))],
    "semantic-context": [curve(a("semantic", "top", 0.9), a("context", "bottom", 0.12))],
    "episodic-context": [curve(a("episodic", "top", 0.85), a("context", "bottom", 0.4))],
    "procedural-context": [curve(a("procedural", "top", 0.4), a("context", "bottom", 0.75)), "recall"],
    "llm-local": [curve(a("llm", "bottom", 0.12), a("local", "top", 0.5))],
    "llm-mcp": [curve(a("llm", "bottom", 0.4), a("mcp", "top", 0.5)), "tool calls"],
    "llm-web": [curve(a("llm", "bottom", 0.68), a("web", "top", 0.4))],
    "llm-visuals": [curve(a("llm", "bottom", 0.92), a("visuals", "top", 0.3))],
    "llm-loop": [`M${a("llm", "top", 0.3)}C${a("llm", "top", 0.3)[0]},${110} ${a("llm", "top", 0.7)[0]},${110} ${a("llm", "top", 0.7)}`, "next step"],
    "reply-session": [`M${a("reply", "top", 0.4)}C${a("reply", "top", 0.4)[0]},70 ${a("session", "top")[0]},70 ${a("session", "top")}`, "save exchange + tool note"],
    "session-consolidation": [curve(a("session", "bottom", 0.4), a("consolidation", "left", 0.5))],
    "consolidation-semantic": [curve(a("consolidation", "top", 0.25), a("semantic", "bottom", 0.4))],
    "consolidation-episodic": [curve(a("consolidation", "top", 0.8), a("episodic", "bottom", 0.4))],
    "reply-reflection": [`M${a("reply", "bottom", 0.5)}C${a("reply", "bottom", 0.5)[0]},430 ${a("reflection", "right")[0] + 160},${a("reflection", "right")[1]} ${a("reflection", "right")}`, "if it went wrong"],
    "reflection-procedural": [curve(a("reflection", "top", 0.5), a("procedural", "bottom", 0.5)), "lesson"],
    "local-semantic": [`M${a("local", "bottom", 0.3)}C${a("local", "bottom", 0.3)[0]},372 ${a("semantic", "bottom", 0.85)[0]},372 ${a("semantic", "bottom", 0.85)}`],
  };

  let root, nodeEls = {}, edgeEls = {}, statusEl, timers = new Map(), started = 0, replaying = null;
  // What is running right now. A start event adds to it, the matching end event removes it,
  // so a module keeps blinking for the whole time it works (a model call can take 40 s).
  const running = new Set();

  function mount(container, status) {
    statusEl = status;
    root = S("svg", { viewBox: "12 24 1076 412", class: "flow", role: "img", "aria-label": "Live diagram of how sora handles a message" });
    container.replaceChildren(root);   // attached first: measuring paths needs a rendered SVG
    const defs = S("defs");
    for (const [id, cls] of [["arrow", "head"], ["arrow-on", "head on"]]) {
      const m = S("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
      m.append(S("path", { d: "M0,0L10,5L0,10z", class: cls }));
      defs.append(m);
    }
    root.append(defs);
    // swim-lane labels
    for (const [y, text] of [[136, "per message"], [278, "memory & tools"], [374, "after the reply"]])
      root.append(S("text", { x: 20, y, class: "lane" }, text));
    for (const [id, [d, label]] of Object.entries(EDGES)) {
      const g = S("g", { class: "edge", "data-id": id });
      g.append(S("path", { d, class: "line", "marker-end": "url(#arrow)" }));
      if (id.startsWith("llm-") && id !== "llm-reply" && id !== "llm-loop") g.firstChild.setAttribute("marker-start", "url(#arrow)");
      root.append(g);
      if (label) {
        const p = g.firstChild;
        const mid = p.getPointAtLength(p.getTotalLength() / 2);
        g.append(S("text", { x: mid.x + 4, y: mid.y - 4, class: "edge-label" }, label));
      }
      edgeEls[id] = g;
    }
    for (const [id, [x, y, w, h, title, sub, group, about, tab]] of Object.entries(NODES)) {
      const g = S("g", { class: `node ${group}`, "data-id": id, tabindex: 0 });
      g.append(S("rect", { x, y, width: w, height: h, rx: 10 }));
      g.append(S("text", { x: x + w / 2, y: y + h / 2 - 3, class: "title", "text-anchor": "middle" }, title));
      g.append(S("text", { x: x + w / 2, y: y + h / 2 + 13, class: "sub", "text-anchor": "middle" }, sub));
      const badge = S("g", { class: "badge" });
      badge.append(S("rect", { x: x + w - 6, y: y - 10, rx: 8, height: 18, width: 10 }));
      badge.append(S("text", { x: x + w - 1, y: y + 3 }, ""));
      g.append(badge);
      g.append(S("title", {}, about));
      if (tab) g.addEventListener("click", () => window.showTab?.(tab));
      root.append(g);
      nodeEls[id] = g;
    }
    say("Idle. Send a message and watch it move through sora.");
  }

  function say(text) { if (statusEl) statusEl.textContent = text; }

  function badge(id, text) {
    const b = nodeEls[id]?.querySelector(".badge");
    if (!b) return;
    const t = b.querySelector("text"), r = b.querySelector("rect");
    t.textContent = text;
    const [x, , w] = NODES[id];
    const width = Math.max(18, 6.5 * String(text).length + 12);
    r.setAttribute("width", width); r.setAttribute("x", x + w - width + 8);
    t.setAttribute("x", x + w - width + 14);
    b.classList.toggle("show", !!text);
  }

  function light(ids, kind = "node", bad = false) {
    for (const id of ids) {
      const node = (kind === "node" ? nodeEls : edgeEls)[id];
      if (!node) continue;
      node.classList.add("active", "visited");
      node.classList.toggle("bad", bad);
      if (kind === "edge") node.querySelector(".line").setAttribute("marker-end", "url(#arrow-on)");
      clearTimeout(timers.get(node));
      timers.set(node, setTimeout(() => {
        node.classList.remove("active");
        if (kind === "edge") node.querySelector(".line").setAttribute("marker-end", "url(#arrow)");
      }, 1400));
    }
    pulseTrace();
  }
  const on = (nodes, edges = [], bad = false) => { light(nodes, "node", bad); light(edges, "edge", bad); };

  function pulseTrace() {
    const t = nodeEls.trace;
    if (!t) return;
    t.classList.remove("pulse"); void t.getBBox(); t.classList.add("pulse", "visited");
  }

  function run(nodes, edges = []) {
    for (const [ids, els] of [[nodes, nodeEls], [edges, edgeEls]])
      for (const id of ids) {
        const n = els[id];
        if (!n) continue;
        n.classList.add("running", "visited");
        running.add(n);
        if (els === edgeEls) n.querySelector(".line").setAttribute("marker-end", "url(#arrow-on)");
      }
  }
  function stop(nodes = null, edges = []) {
    const targets = nodes === null ? [...running]
      : [...nodes.map((id) => nodeEls[id]), ...edges.map((id) => edgeEls[id])].filter(Boolean);
    for (const n of targets) {
      n.classList.remove("running");
      running.delete(n);
      if (n.classList.contains("edge") && !n.classList.contains("active")) n.querySelector(".line").setAttribute("marker-end", "url(#arrow)");
    }
  }

  function reset() {
    stop();
    for (const n of [...Object.values(nodeEls), ...Object.values(edgeEls)]) n.classList.remove("active", "visited", "bad", "pulse");
    for (const id of Object.keys(NODES)) badge(id, "");
    started = performance.now();
  }

  const MEMORY_TOOLS = new Set(["remember", "forget", "search_memory"]);
  const SKILL_TOOLS = new Set(["load_skill", "save_skill"]);
  const t = () => `${((performance.now() - started) / 1000).toFixed(1)}s`;

  const toolTarget = (name) => name.startsWith("show_") ? "visuals" : name.includes("__") ? "mcp" : "local";
  const modelStep = (name) => +name.replace(/\D+/g, "") || 1;

  // One trace event (live, or a span during replay) → what lights up.
  function event(kind, d = {}) {
    const name = d.name || "";
    switch (kind) {
      case "llm_start":
        if (name === "retrieval gate") { run(["gate"], ["session-gate"]); say("Retrieval gate running: does this need long-term memory?"); }
        else if (name.startsWith("model call")) {
          const step = modelStep(name);
          run(["llm"], [step === 1 ? "context-llm" : "llm-loop"]);
          badge("llm", `step ${step}`);
          say(`LLM call, step ${step}: the model is thinking${step > 1 ? " with the tool results" : ""}…`);
        } else if (name === "reflection") { run(["reflection"], ["reply-reflection"]); say("Reflection running: looking for a lesson…"); }
        else if (name === "consolidation") { run(["consolidation"], ["session-consolidation"]); say("Consolidation running: chat → facts + episode…"); }
        else if (name === "judge") { run(["trace"]); say("Judge running…"); }
        break;
      case "tool_start": {
        const target = toolTarget(name);
        run([target], [`llm-${target}`]);
        badge(target, name.replace(/^workspace__/, "").replace(/^show_/, ""));
        say(`Running tool ${name}…`);
        break;
      }
      case "turn":
        reset(); on(["user", "session"], ["user-session"]); say("Message received: loading this conversation's recent history.");
        break;
      case "llm":
        if (name === "retrieval gate") { stop(["gate"], ["session-gate"]); on(["gate"], ["session-gate"], d.ok === false); say(`Retrieval gate done (${t()}).`); }
        else if (name.startsWith("model call")) {
          const step = modelStep(name);
          stop(["llm", "reply"], ["context-llm", "llm-loop", "llm-reply"]);
          on(["llm"], [step === 1 ? "context-llm" : "llm-loop"], d.ok === false);
          badge("llm", `step ${step}`);
          say(`LLM call, step ${step}${d.cost_usd ? ` · $${d.cost_usd.toFixed(4)}` : ""}${d.cache_read_tokens ? ` · ${d.cache_read_tokens} tokens from cache` : ""}`);
        } else if (name === "reflection") { stop(["reflection"], ["reply-reflection"]); on(["reflection"], ["reply-reflection"]); say("Reflection done."); }
        else if (name === "consolidation") { stop(["consolidation"], ["session-consolidation"]); on(["consolidation"], ["session-consolidation"]); say("Consolidation done."); }
        else if (name === "judge") { stop(["trace"]); on(["trace"]); say("Judge done."); }
        break;
      case "gate":
        if (d.retrieve) {
          on(["semantic", "episodic"], ["gate-semantic", "gate-episodic"]);
          badge("semantic", `${(d.facts || []).length} found`);
          if ((d.episodes || []).length) badge("episodic", `${d.episodes.length} found`);
          say(`Memory search for "${d.query}": ${(d.facts || []).length} facts, ${(d.episodes || []).length} episodes.`);
        } else say("Retrieval gate: no long-term memory needed for this one.");
        if (d.correction) { badge("gate", "correction"); }
        break;
      case "context": {
        const edges = ["gate-context"], nodes = ["context"];
        if (d.facts) edges.push("semantic-context");
        if (d.episodes) edges.push("episodic-context");
        const proc = (d.skills || []).length + (d.lessons || []).length + (d.tool_notes || []).length;
        if (proc) { edges.push("procedural-context"); nodes.push("procedural"); badge("procedural", [(d.skills || []).join(", "), (d.lessons || []).length ? `${d.lessons.length} lesson(s)` : ""].filter(Boolean).join(" · ") || "tool notes"); }
        on(nodes, edges);
        say(`Context built: ${d.system_chars ? Math.round(d.system_chars / 4) + " tokens of system prompt" : "prompt ready"}${(d.skills || []).length ? `, skill: ${d.skills.join(", ")}` : ""}.`);
        break;
      }
      case "tool": {
        const target = toolTarget(name);
        stop([target], [`llm-${target}`]);
        const nodes = [target], edges = [`llm-${target}`];
        if (MEMORY_TOOLS.has(name)) { nodes.push("semantic"); edges.push("local-semantic"); }
        if (SKILL_TOOLS.has(name)) nodes.push("procedural");
        on(nodes, edges, d.ok === false);
        badge(target, name.replace(/^workspace__/, "").replace(/^show_/, ""));
        say(`Tool ${name} ${d.ok === false ? "failed" : "ran"}${d.ms != null ? ` in ${d.ms}ms` : ""}.`);
        break;
      }
      case "server_tool":
        if (name === "web_search" || name === "web_fetch" || name === "code_execution") {
          on(["web"], ["llm-web"]); badge("web", name.replace("web_", ""));
          const q = d.input?.query || d.input?.url;
          say(q ? `Web ${name.replace("web_", "")}: ${q}` : `Server tool ${name} running inside the model call.`);
        }
        break;
      case "text":                                   // the answer is streaming out of the model
        if (!running.has(nodeEls.reply)) { run(["reply"], ["llm-reply"]); say("Writing the reply…"); }
        break;
      case "reply":
        on(["reply", "session"], ["llm-reply", "reply-session"]);
        say(d.status && d.status !== "ok" ? `Reply finished (${d.status}).` : "Reply ready: saving the exchange with a note of the tools used.");
        break;
      case "learning":
        if ((d.lessons || []).length) { on(["procedural"], ["reflection-procedural"]); badge("procedural", `+${d.lessons.length} lesson`); say(`Learned: ${d.lessons[0].lesson}`); }
        break;
      case "memory":
        if (name === "consolidation") {
          on(["semantic", "episodic"], ["consolidation-semantic", "consolidation-episodic"]);
          say(`Consolidated: ${(d.facts || []).length} new fact(s)${d.episode ? " and an episode" : ""}.`);
        }
        break;
      case "error":
        stop(); on(["llm"], [], true); say(`Error: ${d.error || "turn failed"}`);
        break;
      case "done":
        stop(); on(["trace"]);
        say(`Done in ${d.latency_ms != null ? (d.latency_ms / 1000).toFixed(1) + "s" : t()} · $${(d.cost_usd || 0).toFixed(4)} · ${d.status}. Hover a module to see what it does; click to open it.`);
        break;
    }
  }

  // Replay a recorded turn: the spans, spaced out so the path is visible.
  function replay(turn) {
    clearInterval(replaying);
    const steps = [["turn", {}], ...turn.spans.filter((s) => s.kind !== "turn").flatMap((s) => {
        const data = { ...s.data, name: s.name, ms: s.duration_ms, ok: s.ok };
        return s.kind === "llm" || s.kind === "tool" ? [[`${s.kind}_start`, data], [s.kind, data]] : [[s.kind, data]];
      }),
      ["done", { latency_ms: turn.latency_ms, cost_usd: turn.cost_usd, status: turn.status }]];
    let i = 0;
    reset();
    replaying = setInterval(() => {
      if (i >= steps.length) { clearInterval(replaying); return; }
      const [kind, data] = steps[i++];
      event(kind, data);
      if (kind !== "done") say(`Replay ${i}/${steps.length}: ${statusEl.textContent}`);
    }, 650);
  }

  window.flow = { mount, event, reset, replay };
})();

// Rich replies: Markdown, charts, tables and drawings. No libraries.
// Safety rule: every piece of text is escaped before any tag is added, so a
// reply can format itself but can never inject HTML or script.
"use strict";

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function inline(text) {
  // text is already escaped
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
}

function markdown(source) {
  const lines = esc(source || "").split("\n");
  const out = [];
  let i = 0;
  const isTableRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {                                   // fenced code
      const code = [];
      for (i++; i < lines.length && !/^```/.test(lines[i]); i++) code.push(lines[i]);
      i++;
      out.push(`<pre><code>${code.join("\n")}</code></pre>`);
    } else if (isTableRow(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      const head = cells(line);
      const body = [];
      for (i += 2; i < lines.length && isTableRow(lines[i]); i++) body.push(cells(lines[i]));
      out.push(`<div class="md-table"><table><thead><tr>${head.map((h) => `<th>${inline(h)}</th>`).join("")}</tr></thead><tbody>` +
        body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>");
    } else if (/^#{1,4}\s/.test(line)) {
      const level = Math.min(line.match(/^#+/)[0].length + 2, 6);
      out.push(`<h${level}>${inline(line.replace(/^#+\s*/, ""))}</h${level}>`); i++;
    } else if (/^\s*([-*]|\d+[.)])\s+/.test(line)) {           // lists (one level of nesting)
      const ordered = /^\s*\d/.test(line);
      const items = [];
      while (i < lines.length && /^\s*([-*]|\d+[.)])\s+/.test(lines[i])) {
        const nested = /^\s{2,}/.test(lines[i]);
        items.push(`<li${nested ? ' class="nested"' : ""}>${inline(lines[i].replace(/^\s*([-*]|\d+[.)])\s+/, ""))}</li>`); i++;
      }
      out.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
    } else if (/^\s*(---|\*\*\*)\s*$/.test(line)) { out.push("<hr>"); i++; }
    else if (/^&gt;\s?/.test(line)) { out.push(`<blockquote>${inline(line.replace(/^&gt;\s?/, ""))}</blockquote>`); i++; }
    else if (!line.trim()) { i++; }
    else {
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^(```|#{1,4}\s|\s*([-*]|\d+[.)])\s+|&gt;)/.test(lines[i]) && !isTableRow(lines[i])) para.push(lines[i++]);
      if (!para.length) para.push(lines[i++]);
      out.push(`<p>${inline(para.join("<br>"))}</p>`);
    }
  }
  return out.join("");
}

// ---------- charts (SVG)
const NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, text) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
}
const PALETTE = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7", "--c8"];
const color = (i) => `var(${PALETTE[i % PALETTE.length]})`;
function niceTicks(max, min = 0) {
  const span = max - min || 1, raw = span / 4, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let t = Math.floor(min / step) * step; t <= max + step * 0.001; t += step) ticks.push(+t.toFixed(10));
  if (ticks[ticks.length - 1] < max) ticks.push(ticks[ticks.length - 1] + step);
  return ticks;
}
const fmt = (v) => Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(1) + "M" : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(1) + "k" : String(+v.toFixed(2));

function chart(spec) {
  const W = 600, H = 300, pad = { l: 52, r: 16, t: 14, b: spec.labels.length > 8 ? 70 : 40 };
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": spec.title });
  if (spec.type === "pie") return pie(spec, root, W, H);
  const all = spec.series.flatMap((s) => s.values);
  const ticks = niceTicks(Math.max(0, ...all), Math.min(0, ...all));
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const x0 = pad.l, x1 = W - pad.r, y0 = H - pad.b, y1 = pad.t;
  const y = (v) => y0 - ((v - lo) / (hi - lo || 1)) * (y0 - y1);
  const band = (x1 - x0) / spec.labels.length;
  for (const t of ticks) {
    root.append(svg("line", { x1: x0, x2: x1, y1: y(t), y2: y(t), class: "grid" }));
    root.append(svg("text", { x: x0 - 6, y: y(t) + 4, class: "tick", "text-anchor": "end" }, fmt(t)));
  }
  spec.labels.forEach((label, i) => {
    const cx = x0 + band * (i + 0.5);
    const rotate = spec.labels.length > 8;
    root.append(svg("text", { x: cx, y: y0 + 16, class: "tick", "text-anchor": rotate ? "end" : "middle",
      transform: rotate ? `rotate(-40 ${cx} ${y0 + 16})` : "" }, String(label).slice(0, 18)));
  });
  if (spec.type === "bar") {
    const n = spec.series.length, gap = band * 0.2, bw = (band - gap) / n;
    spec.series.forEach((s, si) => s.values.forEach((v, i) => {
      const top = Math.min(y(v), y(0)), h = Math.abs(y(v) - y(0));
      const r = svg("rect", { x: x0 + band * i + gap / 2 + bw * si, y: top, width: Math.max(1, bw - 2), height: Math.max(1, h), rx: 2, fill: color(si) });
      r.append(svg("title", {}, `${s.name ? s.name + " · " : ""}${spec.labels[i]}: ${v}`));
      root.append(r);
    }));
  } else {
    spec.series.forEach((s, si) => {
      const pts = s.values.map((v, i) => [x0 + band * (i + 0.5), y(v)]);
      root.append(svg("polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none", stroke: color(si), "stroke-width": 2.5, "stroke-linejoin": "round" }));
      pts.forEach(([px, py], i) => {
        const c = svg("circle", { cx: px, cy: py, r: 3.5, fill: color(si) });
        c.append(svg("title", {}, `${s.name ? s.name + " · " : ""}${spec.labels[i]}: ${s.values[i]}`));
        root.append(c);
      });
    });
  }
  root.append(svg("line", { x1: x0, x2: x1, y1: y(Math.max(0, lo)), y2: y(Math.max(0, lo)), class: "axis" }));
  if (spec.y_label) root.append(svg("text", { x: 12, y: (y0 + y1) / 2, class: "axis-label", transform: `rotate(-90 12 ${(y0 + y1) / 2})`, "text-anchor": "middle" }, spec.y_label));
  if (spec.x_label) root.append(svg("text", { x: (x0 + x1) / 2, y: H - 4, class: "axis-label", "text-anchor": "middle" }, spec.x_label));
  return root;
}

function pie(spec, root, W, H) {
  const values = spec.series[0].values, total = values.reduce((a, b) => a + b, 0) || 1;
  const cx = 150, cy = H / 2, R = 115, r = 62;
  let angle = -Math.PI / 2;
  values.forEach((v, i) => {
    const a2 = angle + (v / total) * Math.PI * 2, large = a2 - angle > Math.PI ? 1 : 0;
    const p = (rad, rr) => [cx + rr * Math.cos(rad), cy + rr * Math.sin(rad)];
    const [ax, ay] = p(angle, R), [bx, by] = p(a2, R), [cx2, cy2] = p(a2, r), [dx, dy] = p(angle, r);
    const d = values.length === 1 ? `M${cx - R},${cy}a${R},${R} 0 1,0 ${2 * R},0a${R},${R} 0 1,0 ${-2 * R},0Z`
      : `M${ax},${ay}A${R},${R} 0 ${large} 1 ${bx},${by}L${cx2},${cy2}A${r},${r} 0 ${large} 0 ${dx},${dy}Z`;
    const slice = svg("path", { d, fill: color(i), stroke: "var(--panel)", "stroke-width": 2 });
    slice.append(svg("title", {}, `${spec.labels[i]}: ${v} (${(100 * v / total).toFixed(1)}%)`));
    root.append(slice);
    angle = a2;
  });
  root.append(svg("text", { x: cx, y: cy + 5, class: "pie-total", "text-anchor": "middle" }, fmt(total)));
  spec.labels.slice(0, 12).forEach((label, i) => {
    const ly = 30 + i * 22;
    root.append(svg("rect", { x: 300, y: ly - 10, width: 12, height: 12, rx: 2, fill: color(i) }));
    root.append(svg("text", { x: 320, y: ly, class: "legend" }, `${String(label).slice(0, 28)} · ${(100 * values[i] / total).toFixed(1)}%`));
  });
  return root;
}

function legend(spec) {
  if (spec.type === "pie" || spec.series.length < 2) return null;
  const box = el("div", { class: "chart-legend" });
  spec.series.forEach((s, i) => box.append(el("span", {}, el("i", { style: `background:${color(i)}` }), s.name || `series ${i + 1}`)));
  return box;
}

function table(spec) {
  const wrap = el("div", { class: "md-table interactive" });
  const t = el("table");
  const rows = spec.rows.map((r) => [...r]);
  let sortCol = -1, asc = true;
  const head = el("tr");
  const body = el("tbody");
  const draw = () => body.replaceChildren(...rows.map((r) => el("tr", {}, r.map((c) => el("td", {}, c == null ? "" : String(c))))));
  spec.columns.forEach((name, ci) => head.append(el("th", { title: "Sort", onclick: () => {
    asc = sortCol === ci ? !asc : true; sortCol = ci;
    rows.sort((a, b) => { const x = a[ci], y = b[ci];
      const r = typeof x === "number" && typeof y === "number" ? x - y : String(x ?? "").localeCompare(String(y ?? ""), undefined, { numeric: true });
      return asc ? r : -r; });
    head.querySelectorAll("th").forEach((th, i) => th.dataset.sort = i === ci ? (asc ? "▲" : "▼") : "");
    draw(); } }, String(name))));
  t.append(el("thead", {}, head), body);
  draw();
  wrap.append(t);
  return wrap;
}

function drawing(spec) {
  // An <img> never runs scripts or loads anything from inside an SVG.
  return el("img", { class: "drawing", alt: spec.title, src: "data:image/svg+xml;charset=utf-8," + encodeURIComponent(spec.svg) });
}

function visual(v) {
  const fig = el("figure", { class: "visual" });
  fig.append(el("figcaption", {}, v.title || ""));
  try {
    if (v.tool === "show_chart") { fig.append(chart(v)); const l = legend(v); if (l) fig.append(l); }
    else if (v.tool === "show_table") fig.append(table(v));
    else if (v.tool === "show_diagram") fig.append(drawing(v));
  } catch (e) { fig.append(el("p", { class: "bad small" }, `Couldn't draw this: ${e.message}`)); }
  return fig;
}

window.render = { markdown, visual, esc };

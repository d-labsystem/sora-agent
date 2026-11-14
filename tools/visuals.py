"""Visual tools: the agent can show charts, tables and drawings in the chat.

These tools don't compute anything. Their *arguments* are the visual: the
model writes a small, structured spec, the tool checks it, and the dashboard
draws it (harness/static/app.js). The reply that used one stores the spec
with the message, so the visual comes back when you reopen the conversation.

Why specs, not images? The model can't output pixels, but it is very good at
writing structured data. A chart spec is a few hundred tokens and can be
redrawn at any size, restyled for dark mode, sorted or hovered. The same idea
is behind Vega-Lite and Claude's own artifacts.

Safety: the browser draws charts and tables from plain data, so there is
nothing to execute. SVG drawings can carry scripts, so they are checked here
and shown through an <img> tag, where browsers never run scripts or load
external resources from an SVG.
"""

from __future__ import annotations

import re

from tools.registry import Tool, ToolError

VISUAL_TOOLS = ("show_chart", "show_table", "show_diagram")
CHART_TYPES = ("bar", "line", "pie")
MAX_POINTS, MAX_SERIES = 100, 8
MAX_ROWS, MAX_COLUMNS = 200, 12
MAX_SVG = 60_000
UNSAFE_SVG = re.compile(r"<\s*(script|foreignobject|iframe)|\son\w+\s*=|javascript:", re.I)


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{where} must be a number, got {value!r}")
    return float(value)


def show_chart(type: str, title: str, labels: list, series: list,
               x_label: str = "", y_label: str = "") -> str:
    if type not in CHART_TYPES:
        raise ToolError(f"type must be one of {', '.join(CHART_TYPES)}")
    if not labels or len(labels) > MAX_POINTS:
        raise ToolError(f"give 1 to {MAX_POINTS} labels")
    if not series or len(series) > MAX_SERIES:
        raise ToolError(f"give 1 to {MAX_SERIES} series")
    if type == "pie" and len(series) != 1:
        raise ToolError("a pie chart takes exactly one series")
    for i, s in enumerate(series):
        if not isinstance(s, dict) or not isinstance(s.get("values"), list):
            raise ToolError(f"series {i} must be an object with a 'values' list")
        if len(s["values"]) != len(labels):
            raise ToolError(f"series {i} has {len(s['values'])} values but there are {len(labels)} labels")
        for j, v in enumerate(s["values"]):
            _number(v, f"series {i} value {j}")
        if type == "pie" and any(v < 0 for v in s["values"]):
            raise ToolError("pie values can't be negative")
    return f"Chart shown to the user: {type} chart '{title}' ({len(labels)} points, {len(series)} series)."


def show_table(title: str, columns: list, rows: list) -> str:
    if not columns or len(columns) > MAX_COLUMNS:
        raise ToolError(f"give 1 to {MAX_COLUMNS} columns")
    if len(rows) > MAX_ROWS:
        raise ToolError(f"at most {MAX_ROWS} rows; summarise or split the table")
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(columns):
            raise ToolError(f"row {i} must be a list of {len(columns)} cells")
        for cell in row:
            if not isinstance(cell, (str, int, float, bool)) and cell is not None:
                raise ToolError(f"row {i} has a cell that isn't text or a number")
    return f"Table shown to the user: '{title}' ({len(rows)} rows x {len(columns)} columns)."


def show_diagram(title: str, svg: str) -> str:
    svg = svg.strip()
    if not svg.lower().startswith(("<svg", "<?xml")) or "</svg>" not in svg.lower():
        raise ToolError("svg must be a complete <svg>...</svg> document")
    if "xmlns" not in svg:
        raise ToolError('add xmlns="http://www.w3.org/2000/svg" to the <svg> tag')
    if len(svg) > MAX_SVG:
        raise ToolError(f"svg is over {MAX_SVG} characters; simplify the drawing")
    if UNSAFE_SVG.search(svg):
        raise ToolError("svg can't contain scripts, event handlers or embedded HTML")
    return f"Diagram shown to the user: '{title}'."


_series = {"type": "array", "items": {"type": "object", "properties": {
    "name": {"type": "string"}, "values": {"type": "array", "items": {"type": "number"}}},
    "required": ["values"]}}


def visual_tools() -> list[Tool]:
    return [
        Tool("show_chart",
             "Show a chart in the chat. Use it when numbers compare or change: trends over time "
             "(line), comparing categories (bar), parts of a whole (pie, one series, few slices). "
             "Give the real numbers; never invent data. Mention the chart briefly in your reply "
             "instead of repeating every number.",
             {"type": "object", "properties": {
                 "type": {"type": "string", "enum": list(CHART_TYPES)},
                 "title": {"type": "string"},
                 "labels": {"type": "array", "items": {"type": "string"},
                            "description": "x-axis categories or pie slice names"},
                 "series": _series,
                 "x_label": {"type": "string"}, "y_label": {"type": "string"}},
              "required": ["type", "title", "labels", "series"]}, show_chart, source="visual"),
        Tool("show_table",
             "Show a sortable table in the chat. Use it for comparisons across several attributes "
             "(options x criteria, specs, schedules). Keep cells short.",
             {"type": "object", "properties": {
                 "title": {"type": "string"},
                 "columns": {"type": "array", "items": {"type": "string"}},
                 "rows": {"type": "array", "items": {"type": "array"}}},
              "required": ["title", "columns", "rows"]}, show_table, source="visual"),
        Tool("show_diagram",
             "Show an SVG drawing in the chat: architecture or flow diagrams, timelines, simple "
             "illustrations, icons. Write a complete <svg> with xmlns and a viewBox, readable text "
             "(font-size 12+), and colours that work on light and dark backgrounds. No scripts.",
             {"type": "object", "properties": {
                 "title": {"type": "string"}, "svg": {"type": "string"}},
              "required": ["title", "svg"]}, show_diagram, source="visual"),
    ]

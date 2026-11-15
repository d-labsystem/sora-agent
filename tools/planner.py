"""The planner: a checklist the model writes for itself on multi-step tasks.

Why. Given a request with several parts ("compare three laptops, check
prices, save a summary note"), a model working in one undirected loop can
lose a part or stop early, and nothing notices. Writing the steps down
first, and ticking them off, keeps all of them in view; the harness can
then check the list before the turn ends.

How it fits the loop.
  * The retrieval gate flags a request as complex (memory/gate.py). Long
    tasks count as complex too.
  * On those turns the per-turn prompt adds PLAN_NOTE: plan first.
  * The model calls update_plan with the whole list, and again whenever a
    step changes (in progress, done, blocked, or the plan is revised).
  * If it tries to finish with steps still open, the loop sends it back once
    (loop/agent.py, NUDGE). It's a nudge, not a wall: a step may be
    impossible, and then the model says so and marks it blocked.
  * Compaction keeps the checklist (render()) in the summary's continuation.

Why update_plan is registered on every turn. Tools are the first thing in
the cached prompt prefix; adding a tool only on complex turns would change
that prefix and cost a full cache write. The instruction to use it is what
varies, and it sits after the cache breakpoint.
"""

from __future__ import annotations

from tools.registry import Tool, ToolError

STATUSES = ("pending", "in_progress", "done", "blocked")
OPEN = ("pending", "in_progress")
MAX_STEPS = 12
MARK = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "blocked": "[!]"}

PLAN_NOTE = ("This request has several parts. Before anything else, call update_plan with 3 to 7 short, "
             "concrete steps that together finish the whole request. Work through them in order, calling "
             "update_plan again as each step starts and finishes (send the whole list each time). If a step "
             "turns out impossible, mark it blocked with a note saying why; if you learn something that "
             "changes the plan, revise it. Answer only when every step is done or blocked.")

NUDGE = ("Your plan still has open steps: {steps}. Finish them now, or, if one can't be done, mark it "
         "blocked in update_plan with the reason, then give your final answer.")


class Plan:
    """This turn's checklist. The harness calls reset() at the start of every
    turn and passes the turn's notify, so each change is traced."""

    def __init__(self):
        self.steps: list[dict] = []
        self.notify = None
        self.updates = 0

    def reset(self, notify=None) -> None:
        self.steps, self.notify, self.updates = [], notify, 0

    def open_steps(self) -> list[dict]:
        return [s for s in self.steps if s["status"] in OPEN]

    def render(self) -> str:
        return "\n".join(f"{MARK[s['status']]} {i}. {s['title']}" + (f"  ({s['note']})" if s.get("note") else "")
                         for i, s in enumerate(self.steps, 1))

    def update(self, steps: list) -> str:
        if not isinstance(steps, list) or not steps:
            raise ToolError("steps must be a non-empty list of {title, status}")
        if len(steps) > MAX_STEPS:
            raise ToolError(f"at most {MAX_STEPS} steps; group smaller ones together")
        clean = []
        for i, step in enumerate(steps, 1):
            if not isinstance(step, dict) or not str(step.get("title", "")).strip():
                raise ToolError(f"step {i} needs a title")
            status = step.get("status", "pending")
            if status not in STATUSES:
                raise ToolError(f"step {i}: status must be one of {', '.join(STATUSES)}")
            if status == "blocked" and not str(step.get("note", "")).strip():
                raise ToolError(f"step {i} is blocked: add a note saying why")
            clean.append({"title": " ".join(str(step["title"]).split())[:200], "status": status,
                          **({"note": " ".join(str(step["note"]).split())[:300]} if step.get("note") else {})})
        if sum(s["status"] == "in_progress" for s in clean) > 1:
            raise ToolError("only one step can be in_progress at a time")
        self.steps = clean
        self.updates += 1
        if self.notify:
            self.notify("plan", name="plan", steps=clean, open=len(self.open_steps()))
        left = self.open_steps()
        return "Plan saved:\n" + self.render() + (f"\n{len(left)} step(s) to go." if left else "\nAll steps closed.")

    def tool(self) -> Tool:
        step = {"type": "object", "properties": {
            "title": {"type": "string"},
            "status": {"type": "string", "enum": list(STATUSES)},
            "note": {"type": "string", "description": "why a step is blocked, or a short result"}},
            "required": ["title", "status"]}
        return Tool(
            "update_plan",
            "Write or update your checklist for a multi-step request: the whole list of steps each time, "
            "with each step's status (pending, in_progress, done, blocked). Use it when the instructions "
            "say a request has several parts; skip it for simple questions.",
            {"type": "object", "properties": {"steps": {"type": "array", "items": step}}, "required": ["steps"]},
            lambda steps: self.update(steps))

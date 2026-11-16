"""The loop: call the model, run the tools it asks for, repeat. This file is the agent.

The model call itself lives in loop/providers.py, so this file is the same
for Anthropic, OpenAI and Hugging Face models.

It ends when the model asks for no tools, or at max_iterations, where one
last call with tools off answers from what the turn already gathered.

How web search works here: web_search and web_fetch are listed in `tools`
like any other tool, but Anthropic runs them. Within one model call, Claude
can search, read the results and search again; the response then holds
`server_tool_use` blocks (what it searched for) and `web_search_tool_result`
blocks (what came back), next to its text. There is nothing for us to execute:
we report those blocks to the trace, and append the whole response to the
conversation so the results stay in context. If the server's own search loop
runs long, the response stops with "pause_turn", and sending the conversation
back as it is lets the server carry on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loop.compaction import compact, context_size, first_request
from loop.providers import Step, as_provider
from ops.pricing import cost
from tools.planner import NUDGE
from tools.registry import ToolRegistry

Notify = Callable[..., None]

LIMIT_NOTE = ("You have reached the step limit for this turn and cannot call more tools. "
              "Answer now from what you already gathered, and say what is missing.")
BUDGET_NOTE = ("You have reached the spending limit for this task and cannot call more tools. "
               "Answer now from what you already gathered, and say what is missing.")
RETRY_GROWTH = 1.25     # after a failed summary, wait until the context is 25% bigger
TRUNCATED = "\n\n(The reply hit the length limit and was cut off.)"
REFUSED = "(The model declined this request.)"
TRACE_OUTPUT_CHARS = 4000

System = str | list[dict[str, Any]]


@dataclass
class LoopResult:
    reply: str = ""
    iterations: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0
    cost_usd: float = 0.0
    model: str = ""
    limit_reached: bool = False
    stop_reason: str = ""
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    server_calls: list[dict[str, Any]] = field(default_factory=list)   # web_search, web_fetch
    budget_reached: bool = False
    compactions: int = 0
    plan_nudged: bool = False         # tried to finish with plan steps open, and was sent back

    @property
    def tool_errors(self) -> list[dict[str, Any]]:
        return [c for c in self.tool_calls if c["is_error"]]


def _record(result: LoopResult, step: Step, notify: Notify, iteration: int, ms: int) -> None:
    u = step.usage
    served_by = step.model or result.model
    spent = cost(served_by, u.input_tokens, u.output_tokens, u.web_searches, u.cache_read,
                 u.cache_write) + u.tool_fees_usd
    result.input_tokens += u.input_tokens
    result.output_tokens += u.output_tokens
    result.cache_read_tokens += u.cache_read
    result.cache_write_tokens += u.cache_write
    result.web_searches += u.web_searches
    result.cost_usd += spent
    result.model = served_by
    notify("llm", name=f"model call {iteration}", model=served_by, ms=ms, ok=True,
           stop_reason=step.stop_reason, input_tokens=u.input_tokens,
           output_tokens=u.output_tokens, cache_read_tokens=u.cache_read,
           cache_write_tokens=u.cache_write, web_searches=u.web_searches, cost_usd=spent,
           # what the model wrote in this step (the trace viewers show it as the output)
           **({"output": step.text[:TRACE_OUTPUT_CHARS]} if step.text else {}),
           **({"reasoning_tokens": u.reasoning_tokens} if u.reasoning_tokens else {}))
    for kind, data in step.server_events:
        notify(kind, **data)
        if kind == "server_tool" and data["name"] in ("web_search", "web_fetch"):
            result.server_calls.append({"tool": data["name"], "args": data["input"]})


def _compact(provider, model, system, messages, schemas, effort, task, result, notify, size) -> bool:
    """Replace `messages` with a summary. False if no summary came back."""
    started = time.perf_counter()
    notify("llm_start", name="compaction")
    c = compact(provider, model, system, messages, schemas, effort, task)
    u = c.usage
    spent = cost(c.model, u.input_tokens, u.output_tokens, 0, u.cache_read, u.cache_write)
    result.input_tokens += u.input_tokens
    result.output_tokens += u.output_tokens
    result.cache_read_tokens += u.cache_read
    result.cache_write_tokens += u.cache_write
    result.cost_usd += spent
    ok = c.messages is not None
    # An llm span, so the turn's totals include what the summary cost.
    notify("llm", name="compaction", model=c.model, ok=ok, stop_reason=c.stop_reason,
           ms=int((time.perf_counter() - started) * 1000), input_tokens=u.input_tokens,
           output_tokens=u.output_tokens, cache_read_tokens=u.cache_read,
           cache_write_tokens=u.cache_write, cost_usd=spent)
    notify("compaction", name="context compacted" if ok else "compaction failed", ok=ok,
           tokens_before=size, messages_before=len(messages), summary=c.summary,
           stop_reason=c.stop_reason)
    if ok:
        messages[:] = c.messages
        result.compactions += 1
    return ok


def _append_note(messages: list, note: str) -> None:
    """Add a note after everything already sent: as a text block after the
    last user message's tool results, or as a new user message."""
    last = messages[-1] if messages else None
    if last and last["role"] == "user" and isinstance(last["content"], list):
        last["content"].append({"type": "text", "text": note})
    else:
        messages.append({"role": "user", "content": note})


def run_loop(provider, model: str, system: System, messages: list, tools: ToolRegistry,
             notify: Notify, effort: str = "medium", max_iterations: int = 10,
             compact_at: int = 0,
             budget_usd: float = 0.0, plan=None, extra_spent=None) -> LoopResult:
    """Run one turn. `provider` is a loop.providers Provider (or a bare
    Anthropic client). `messages` is appended to, except that compaction
    replaces it with a summary when the context passes `compact_at` tokens
    (0 = never). `budget_usd` (0 = none) stops tool use once the turn has
    cost that much, then answers from what it has. `plan` (tools/planner.py)
    is this turn's checklist: finishing with steps still open sends the model
    back once to close them. `extra_spent()` adds spending made on the loop's
    behalf outside it (parallel helpers) to the budget check."""
    provider = as_provider(provider)
    result = LoopResult(model=model)
    schemas = tools.schemas()
    on_text = lambda delta: notify("text", delta=delta)   # text reaches the gateway as it streams
    task = first_request(messages)
    next_try = compact_at

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        started = time.perf_counter()
        notify("llm_start", name=f"model call {iteration}")      # live only: "this is running now"
        step = provider.step(model, system, messages, schemas, effort, on_text)
        _record(result, step, notify, iteration, int((time.perf_counter() - started) * 1000))
        result.stop_reason = step.stop_reason

        # Check why it stopped before trusting the content: a tool call cut
        # off by max_tokens has incomplete arguments, so it must not run.
        if step.stop_reason == "refusal":
            result.reply = REFUSED
            return result
        if step.stop_reason == "max_tokens":
            result.reply = step.text + TRUNCATED
            return result

        messages.append({"role": "assistant", "content": step.content})
        if step.stop_reason == "pause_turn":
            continue   # the server's search loop paused; resending resumes it

        calls = step.tool_calls
        if not calls:
            left = plan.open_steps() if plan is not None else []
            if left and not result.plan_nudged and iteration < max_iterations:
                # The completion check: once per turn, and never a wall. A step that
                # can't be done gets marked blocked with a reason, and that closes it.
                result.plan_nudged = True
                steps = "; ".join(f"{plan.steps.index(s) + 1}. {s['title']}" for s in left)
                notify("plan", name="completion check", open=len(left), steps=plan.steps)
                notify("text", delta="\n\n")
                messages.append({"role": "user", "content": NUDGE.format(steps=steps)})
                continue
            result.reply = step.text
            return result

        tool_results = []
        for call in calls:
            started = time.perf_counter()
            notify("tool_start", name=call.name)
            output, is_error = tools.execute(call.name, call.input)
            ms = int((time.perf_counter() - started) * 1000)
            result.tool_calls.append({"tool": call.name, "args": call.input,
                                      "output": output, "is_error": is_error})
            notify("tool", name=call.name, args=call.input, output=output[:TRACE_OUTPUT_CHARS],
                   ok=not is_error, ms=ms)
            tool_results.append({"type": "tool_result", "tool_use_id": call.id,
                                 "content": output, "is_error": is_error})
        # All results of one step go back in a single message.
        messages.append({"role": "user", "content": tool_results})

        # Between steps, with every tool call answered: the two long-task checks.
        if budget_usd and result.cost_usd + (extra_spent() if extra_spent else 0.0) >= budget_usd:
            result.budget_reached = True
            break
        size = context_size(step.usage)
        if compact_at and size >= next_try and iteration < max_iterations:
            # The checklist goes into the continuation, so the plan outlives the summary.
            now = (f"{task}\n\nYour plan so far (keep using update_plan):\n{plan.render()}"
                   if plan is not None and plan.steps else task)
            if _compact(provider, model, system, messages, schemas, effort, now, result, notify, size):
                next_try = compact_at
            else:
                next_try = int(size * RETRY_GROWTH)     # carry on as it is; try again later

    result.limit_reached = not result.budget_reached
    note = BUDGET_NOTE if result.budget_reached else LIMIT_NOTE
    final = result.iterations + 1
    started = time.perf_counter()
    notify("llm_start", name=f"model call {final}")
    # The note is appended to the conversation, never to the system prompt. With preserved
    # thinking, every thinking block is bound to the system prompt it was produced under;
    # editing the system on this last call made the API reject the whole turn (BrowseComp,
    # 2026-11-16: six long tasks crashed at their dollar cap with "Invalid signature in
    # thinking block ... the system prompt differs"). Appending changes nothing already sent.
    _append_note(messages, note)
    step = provider.step(model, system, messages, schemas, effort, on_text, final=True)
    _record(result, step, notify, final, int((time.perf_counter() - started) * 1000))
    result.reply = step.text
    return result

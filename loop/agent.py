"""The loop: call the model, run the tools it asks for, repeat. This file is the agent.

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

from ops.pricing import cost
from tools.registry import ToolRegistry

Notify = Callable[..., None]

MAX_TOKENS = 64000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
LIMIT_NOTE = ("You have reached the step limit for this turn and cannot call more tools. "
              "Answer now from what you already gathered, and say what is missing.")
TRUNCATED = "\n\n(The reply hit the length limit and was cut off.)"
REFUSED = "(The model declined this request.)"
TRACE_OUTPUT_CHARS = 4000


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

    @property
    def tool_errors(self) -> list[dict[str, Any]]:
        return [c for c in self.tool_calls if c["is_error"]]


def _call_model(client, model: str, effort: str, system: str, messages: list, tools: list,
                notify: Notify, **extra):
    """One streamed model call. Text reaches the gateway as it is generated."""
    with client.beta.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
        tools=tools,
        output_config={"effort": effort},
        betas=[FALLBACK_BETA],
        fallbacks="default",   # a declined request is retried on another model
        **extra,
    ) as stream:
        for delta in stream.text_stream:
            notify("text", delta=delta)
        return stream.get_final_message()


def _text(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


def _server_results(block) -> dict[str, Any]:
    """A short, traceable summary of what a server tool returned."""
    content = getattr(block, "content", None)
    if isinstance(content, list):
        return {"results": [{"title": getattr(r, "title", ""), "url": getattr(r, "url", "")}
                            for r in content][:10]}
    error = getattr(content, "error_code", None)
    if error:
        return {"error": error}
    url = getattr(content, "url", None)
    return {"url": url} if url else {}


def _record(result: LoopResult, response, notify: Notify, iteration: int, ms: int) -> None:
    usage = response.usage
    searches = getattr(getattr(usage, "server_tool_use", None), "web_search_requests", 0) or 0
    served_by = getattr(response, "model", "") or result.model
    spent = cost(served_by, usage.input_tokens, usage.output_tokens, searches)
    result.input_tokens += usage.input_tokens
    result.output_tokens += usage.output_tokens
    result.web_searches += searches
    result.cost_usd += spent
    result.model = served_by
    notify("llm", name=f"model call {iteration}", model=served_by, ms=ms, ok=True,
           stop_reason=response.stop_reason, input_tokens=usage.input_tokens,
           output_tokens=usage.output_tokens, web_searches=searches, cost_usd=spent)
    for block in response.content:
        if block.type == "server_tool_use":
            notify("server_tool", name=block.name, input=block.input)
        elif block.type.endswith("_tool_result") and block.type != "tool_result":
            notify("server_result", name=block.type, **_server_results(block))


def run_loop(client, model: str, system: str, messages: list, tools: ToolRegistry,
             notify: Notify, effort: str = "medium", max_iterations: int = 10,
             tool_notes: dict[str, list[str]] | None = None) -> LoopResult:
    """Run one turn. `messages` is only ever appended to."""
    result = LoopResult(model=model)
    schemas = tools.schemas(tool_notes)

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        started = time.perf_counter()
        response = _call_model(client, model, effort, system, messages, schemas, notify)
        _record(result, response, notify, iteration, int((time.perf_counter() - started) * 1000))
        result.stop_reason = response.stop_reason

        # Check why it stopped before trusting the content: a tool call cut
        # off by max_tokens has incomplete arguments, so it must not run.
        if response.stop_reason == "refusal":
            result.reply = REFUSED
            return result
        if response.stop_reason == "max_tokens":
            result.reply = _text(response) + TRUNCATED
            return result

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue   # the server's search loop paused; resending resumes it

        calls = [b for b in response.content if b.type == "tool_use"]
        if not calls:
            result.reply = _text(response)
            return result

        tool_results = []
        for call in calls:
            started = time.perf_counter()
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

    result.limit_reached = True
    started = time.perf_counter()
    response = _call_model(client, model, effort, f"{system}\n\n{LIMIT_NOTE}", messages, schemas,
                           notify, tool_choice={"type": "none"})
    _record(result, response, notify, max_iterations + 1, int((time.perf_counter() - started) * 1000))
    result.reply = _text(response)
    return result

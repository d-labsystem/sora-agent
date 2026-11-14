"""Compaction: when one turn's context gets long, replace it with a summary.

Why a turn grows. Every tool result stays in the conversation until the turn
ends, and every step sends everything before it again. A long research task
(dozens of searches and pages) would outgrow the context window, and answers
get worse well before that, so between steps the loop checks the size and
compacts when it passes a limit (Settings.compact_at).

The shape: replace everything. The summary takes the place of the whole
conversation, and the next step starts from it alone. Keeping the latest
turns word for word sounds better, but on models with preserved thinking
those turns' thinking was produced with the full history before it, and
replaying it after a summary is rejected. Replacing everything carries
nothing that can be invalidated; Anthropic's docs call this simple
compaction and report it works about as well as the elaborate shapes.

Two ways to get the summary:
    Claude      on-demand compaction (beta compact-2026-09-04): the API
                writes a signed `compaction` block that stands in for the
                messages it summarizes
    others      one ordinary call with SUMMARY_PROMPT; its text replaces them

Never mid tool round: the loop compacts only after a step's tool results are
in the conversation, never while a tool call waits for its result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loop.providers import COMPACT_BETA, AnthropicProvider, Usage

# Adapted from Anthropic's recommended client-side summary prompt. The last
# sentence matters: the summarizer still sees the tools, and without it it
# sometimes calls one instead of writing the summary.
SUMMARY_PROMPT = (
    "Summarize the transcript inside <summary></summary> tags. Include relevant information in the "
    "summary such that this task will be continued by a new context window without needing to redo "
    "work or be reprovided with relevant constraints or context. Be sure to preserve: (1) any "
    "difficulties or problems that came up, and how they were handled or resolved; (2) any "
    "possibilities, options, or approaches that were raised, tried, or set aside, and why; (3) "
    "anything that was asked for, decided, agreed, ruled out, or established as a preference, "
    "constraint, or boundary - stated exactly; (4) exactly where things stand now - what has been "
    "covered, settled, or completed so far, including results of searches, pages read and tool "
    "calls; (5) anything still open, unresolved, promised, or expected to happen next; (6) specific "
    "details that would be hard to reconstruct - names, numbers, dates, exact wording, links or "
    "references - kept exactly. Be complete on these even at the cost of length; keep everything "
    "else concise. Keep what the user said or asked for close to their own words; your own "
    "reasoning can be condensed to what it concluded. Do not call any tools while writing this "
    "summary; respond with text only.")

CONTINUE = ("The conversation so far was summarized above because it grew long. The user's request "
            "was: {task}\n\nContinue the task from where the summary leaves off. Don't redo work the "
            "summary says is done.")

SUMMARY_MAX_TOKENS = 16000


def context_size(usage: Usage) -> int:
    """How big the conversation is now: everything the last step read
    (uncached, cached and newly cached input) plus what it wrote, which goes
    back in on the next step."""
    return usage.input_tokens + usage.cache_read + usage.cache_write + usage.output_tokens


def first_request(messages: list) -> str:
    """The user's request for this turn: the last plain-text user message."""
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(message["content"], str):
            return message["content"]
    return "(see the summary)"


@dataclass
class Compaction:
    messages: list | None          # the conversation to continue with; None: no summary
    summary: str
    stop_reason: str
    model: str
    usage: Usage = field(default_factory=Usage)


def _usage_from_iterations(usage) -> Usage:
    """A compaction response reports zero top-level tokens; what it cost is
    in usage.iterations, under the 'compaction' entry."""
    total = Usage()
    for it in getattr(usage, "iterations", None) or []:
        if getattr(it, "type", "") != "compaction":
            continue
        total.input_tokens += it.input_tokens or 0
        total.output_tokens += it.output_tokens or 0
        total.cache_read += getattr(it, "cache_read_input_tokens", 0) or 0
        total.cache_write += getattr(it, "cache_creation_input_tokens", 0) or 0
    return total


def _server(provider: AnthropicProvider, model, system, messages, tools, effort, task) -> Compaction:
    """Anthropic on-demand compaction. Same system and tools as the turn, so the
    summarizer reads the same context; it reads the tools but never runs one."""
    response = provider.client.beta.messages.create(
        model=model, max_tokens=SUMMARY_MAX_TOKENS, system=system, messages=messages, tools=tools,
        output_config={"effort": effort}, betas=[COMPACT_BETA],
        compaction={"type": "summarize", "instructions": SUMMARY_PROMPT})
    usage = _usage_from_iterations(response.usage)
    block = next((b for b in response.content if b.type == "compaction"), None)
    served = getattr(response, "model", "") or model
    if response.stop_reason != "compaction" or block is None:
        return Compaction(None, "", response.stop_reason, served, usage)
    # The block goes first, inside a user message: an assistant message last
    # would be a prefill, which these models reject.
    return Compaction([{"role": "user", "content": [block, {"type": "text", "text": CONTINUE.format(task=task)}]}],
                      getattr(block, "content", "") or "", "compaction", served, usage)


def _client_side(provider, model, system, messages, effort, task) -> Compaction:
    """Any provider: ask for the summary as an ordinary reply, without tools."""
    step = provider.step(model, system, [*messages, {"role": "user", "content": SUMMARY_PROMPT}], [],
                         effort, lambda _delta: None, final=True)
    found = re.search(r"<summary>(.*?)(?:</summary>|$)", step.text, re.S)
    summary = (found.group(1) if found else step.text).strip()
    if step.stop_reason != "end_turn" or not summary:
        return Compaction(None, "", step.stop_reason, step.model, step.usage)
    text = f"<summary>\n{summary}\n</summary>\n\n{CONTINUE.format(task=task)}"
    return Compaction([{"role": "user", "content": text}], summary, "compaction", step.model, step.usage)


def compact(provider, model: str, system, messages: list, tools: list, effort: str,
            task: str | None = None) -> Compaction:
    """Summarize `messages`. Returns the conversation to continue with, or
    messages=None when no summary came back (the turn then carries on as it is)."""
    task = task or first_request(messages)
    if isinstance(provider, AnthropicProvider):
        return _server(provider, model, system, messages, tools, effort, task)
    return _client_side(provider, model, system, messages, effort, task)

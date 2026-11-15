"""Consolidation by LangMem: memories that are updated and deleted, not only added.

sora's own consolidator (memory/consolidation.py) only ever adds facts. Say
"I moved from Pune to Mumbai" and it stores "lives in Mumbai" next to "lives
in Pune"; both are then recalled and the model has to guess which is true.

LangMem's memory manager reads the conversation together with the stored
memories most related to it, and answers with a list of changes: new
memories, rewritten ones, and (with enable_deletes) ones to remove. It works
on LangGraph's store, which is why SORA_CONSOLIDATOR=langmem needs
SORA_MEMORY=langgraph.

How one run goes:
  1. search    the store is searched with the last 1, 2, 4, 8 and 16 messages
               (query_limit=20 gives five windows; LangMem's default of 5 gives
               one, the last message alone, which misses "Pune" when the user
               only says "I moved to Mumbai")
  2. extract   one model call: conversation + those memories -> changes
  3. apply     puts and deletes on the store
Then sora writes the dated episode with its own small call, as before.

LangMem calls the model through LangChain, not sora's providers, so a
callback reports each of its calls as an "llm" span with tokens and cost:
every dollar still shows up in the trace.
"""

from __future__ import annotations

import logging
import time

from loop.llm import ask_json, schema
from loop.providers import split_model
from memory.consolidation import EVERY, MAX_CHARS
from ops.pricing import PRICES, cost

log = logging.getLogger(__name__)

QUERY_LIMIT = 20

INSTRUCTIONS = """You maintain the long-term memory of a personal assistant.
Keep durable, standalone facts about the user and their world: people,
preferences, plans with dates, projects, decisions. Each memory must make
sense on its own: use names, not pronouns. One fact per memory.
When the conversation changes something already stored (a move, a new date,
a changed preference), update that memory, or delete it and record the new
fact, so no outdated version is left behind. Delete memories the user asks to
forget. Skip small talk, one-off requests, and things the assistant said that
the user did not confirm. If nothing durable was said, change nothing."""

EPISODE_FORMAT = schema({"episode": {"type": "string"}})
EPISODE_PROMPT = """In one sentence, what happened or was decided in this conversation excerpt?

{conversation}"""


def chat_model_name(spec: str) -> str:
    """sora's model spec as LangChain's init_chat_model name."""
    kind, model_id = split_model(spec)
    if kind == "anthropic":
        return f"anthropic:{model_id}"
    if kind == "openai":
        return f"openai:{model_id}"
    raise ValueError(f"LangMem consolidation supports Anthropic and OpenAI models, not {spec!r}")


def span_callback(notify, requested: str):
    """A LangChain callback that turns each model call into an llm span."""
    from langchain_core.callbacks import BaseCallbackHandler

    class Spans(BaseCallbackHandler):
        def __init__(self):
            self.started: dict = {}
            self.calls = 0
            self.cost_usd = 0.0

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            self.started[run_id] = time.perf_counter()
            if notify:
                notify("llm_start", name="langmem")

        def on_llm_end(self, response, *, run_id, **kwargs):
            message = response.generations[0][0].message if response.generations else None
            meta = getattr(message, "usage_metadata", None) or {}
            details = meta.get("input_token_details") or {}
            read, write = details.get("cache_read", 0) or 0, details.get("cache_creation", 0) or 0
            served = (getattr(message, "response_metadata", {}) or {}).get("model_name") or requested
            served = served if served in PRICES else requested
            # LangChain's input_tokens includes cached tokens; sora counts them apart
            fresh = max(0, (meta.get("input_tokens", 0) or 0) - read - write)
            spent = cost(served, fresh, meta.get("output_tokens", 0) or 0, cache_read=read, cache_write=write)
            self.calls += 1
            self.cost_usd += spent
            if notify:
                notify("llm", name="langmem", model=served, ok=True,
                       ms=int((time.perf_counter() - self.started.pop(run_id, time.perf_counter())) * 1000),
                       input_tokens=fresh, output_tokens=meta.get("output_tokens", 0) or 0,
                       cache_read=read, cache_write=write, cost_usd=spent)

        def on_llm_error(self, error, *, run_id, **kwargs):
            if notify:
                notify("llm", name="langmem", model=requested, ok=False, error=str(error),
                       ms=int((time.perf_counter() - self.started.pop(run_id, time.perf_counter())) * 1000),
                       input_tokens=0, output_tokens=0, cost_usd=0.0)

    return Spans()


def build_manager(backend, model_spec: str, chat_model=None):
    """LangMem's store manager over the backend's facts namespace. `chat_model`
    replaces the real model (tests pass a fake)."""
    from langmem import create_memory_store_manager
    return create_memory_store_manager(
        chat_model or chat_model_name(model_spec), instructions=INSTRUCTIONS,
        enable_inserts=True, enable_deletes=True, query_limit=QUERY_LIMIT,
        namespace=backend.facts_ns, store=backend.store)


def consolidate_langmem(store, model_spec: str, small, small_id: str, session_id: str,
                        notify=None, every: int = EVERY, force: bool = False,
                        manager=None) -> dict | None:
    """Like memory.consolidation.consolidate (same trigger, same loss-safety:
    messages are marked done only after memory is saved), with LangMem
    maintaining the facts. Never raises."""
    rows = store.unconsolidated(session_id)
    if not rows or (not force and len(rows) < every * 2):
        return None
    backend = store.semantic
    messages = [{"role": r["role"], "content": r["content"][:MAX_CHARS]} for r in rows]
    _, requested = split_model(model_spec)
    spans = span_callback(notify, requested)
    before = {f["id"]: f["content"] for f in backend.list_facts()}
    try:
        manager = manager or build_manager(backend, model_spec)
        manager.invoke({"messages": messages}, config={"callbacks": [spans]})
        conversation = "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages)
        episode = ask_json(small, small_id, EPISODE_PROMPT.format(conversation=conversation),
                           EPISODE_FORMAT, notify, name="episode")["episode"].strip()
    except Exception as exc:
        log.warning("LangMem consolidation failed, will retry later: %s", exc)
        if notify:
            notify("memory", name="consolidation failed", error=str(exc), pending=len(rows))
        return None

    after = {f["id"]: f["content"] for f in backend.list_facts()}
    result = {"messages": len(rows), "by": "langmem",
              "facts": [after[k] for k in after if k not in before],
              "updated": [after[k] for k in after if k in before and after[k] != before[k]],
              "removed": [before[k] for k in before if k not in after],
              "episode": episode, "calls": spans.calls, "langmem_cost_usd": round(spans.cost_usd, 6)}
    if episode:
        store.add_episode(episode, session_id)
    else:
        backend._changed()                   # LangMem wrote straight to the store
    store.mark_consolidated(session_id, rows[-1]["id"])
    if notify:
        notify("memory", name="consolidation", **result)
    return result

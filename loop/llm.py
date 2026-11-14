"""Small structured model calls that sit around the main loop.

The gate, consolidation, reflection and the eval judge all need the same
thing: ask the small model one question and get JSON back in a fixed shape.
Structured outputs (`output_config.format`) make the API enforce the schema,
so we never parse free text with regexes.

Every call reports itself as an "llm" span with tokens and cost, so the
dashboard can show what each helper call spent, not just the main loop.
"""

from __future__ import annotations

import time

from loop.providers import NoAnswer, Usage, as_provider
from ops.pricing import cost


def schema(properties: dict, required: list[str] | None = None) -> dict:
    """Build an output_config format from JSON-schema properties.
    Every property is required and no extras are allowed, which is what
    structured outputs expects for a strict shape."""
    return {"type": "json_schema", "schema": {
        "type": "object", "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False}}


def ask_json(provider, model: str, prompt: str, output_format: dict,
             notify=None, name: str = "call", max_tokens: int = 4096) -> dict:
    """One non-streaming call that must return JSON. Raises on refusal,
    truncation or bad JSON; callers decide how to fail safely. `provider` is
    a loop.providers Provider (or a bare Anthropic client)."""
    provider = as_provider(provider)
    started = time.perf_counter()
    ok = False
    usage, served = Usage(), model
    if notify:
        notify("llm_start", name=name)        # live only: the diagram shows it running
    try:
        result, usage, served = provider.ask_json(model, prompt, output_format, max_tokens)
        ok = True
        return result
    except NoAnswer as exc:
        usage = exc.usage                     # a cut-off answer is still billed
        raise
    finally:
        if notify:
            notify("llm", name=name, model=served, ok=ok,
                   ms=int((time.perf_counter() - started) * 1000),
                   input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                   cost_usd=cost(served, usage.input_tokens, usage.output_tokens,
                                 cache_read=usage.cache_read))

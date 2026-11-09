"""Small structured model calls that sit around the main loop.

The gate, consolidation, reflection and the eval judge all need the same
thing: ask the small model one question and get JSON back in a fixed shape.
Structured outputs (`output_config.format`) make the API enforce the schema,
so we never parse free text with regexes.

Every call reports itself as an "llm" span with tokens and cost, so the
dashboard can show what each helper call spent, not just the main loop.
"""

from __future__ import annotations

import json
import time

from ops.pricing import cost


def schema(properties: dict, required: list[str] | None = None) -> dict:
    """Build an output_config format from JSON-schema properties.
    Every property is required and no extras are allowed, which is what
    structured outputs expects for a strict shape."""
    return {"type": "json_schema", "schema": {
        "type": "object", "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False}}


def ask_json(client, model: str, prompt: str, output_format: dict,
             notify=None, name: str = "call", max_tokens: int = 4096) -> dict:
    """One non-streaming call that must return JSON. Raises on refusal,
    truncation or bad JSON; callers decide how to fail safely."""
    started = time.perf_counter()
    ok = False
    usage = None
    try:
        response = client.messages.create(
            model=model, max_tokens=max_tokens,
            output_config={"effort": "low", "format": output_format},
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ValueError(f"no usable answer ({response.stop_reason})")
        result = json.loads("".join(b.text for b in response.content if b.type == "text"))
        ok = True
        return result
    finally:
        if notify:
            tokens_in = getattr(usage, "input_tokens", 0) or 0
            tokens_out = getattr(usage, "output_tokens", 0) or 0
            notify("llm", name=name, model=model, ok=ok,
                   ms=int((time.perf_counter() - started) * 1000),
                   input_tokens=tokens_in, output_tokens=tokens_out,
                   cost_usd=cost(model, tokens_in, tokens_out))

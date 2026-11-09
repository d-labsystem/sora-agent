"""Estimated dollars per model call, from list prices. Your invoice is the real number."""

from __future__ import annotations

# (input, output) in dollars per million tokens
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
WEB_SEARCH_USD = 10.0 / 1000   # web search is billed per search, on top of tokens


def cost(model: str, input_tokens: int, output_tokens: int, web_searches: int = 0) -> float:
    """Unknown models count as $0 for tokens, so a new model never crashes a turn."""
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000 + web_searches * WEB_SEARCH_USD

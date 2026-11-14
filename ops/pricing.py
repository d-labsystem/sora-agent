"""Estimated dollars per model call, from list prices. Your invoice is the real number.

Prompt caching changes the bill for input tokens, in three buckets:
    input_tokens     not cached: full price
    cache_write      written to the cache this call: 1.25x the input price (5-minute TTL)
    cache_read       served from the cache: about 0.1x, and 0.05x on Opus 5.5 / Sonnet 5.5
"""

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
# Cache-read prices that aren't the usual 0.1x of input.
CACHE_READ = {"claude-fable-5-1": 0.25, "claude-opus-5-5": 0.20, "claude-sonnet-5-5": 0.20}
CACHE_WRITE_MULTIPLIER = 1.25
WEB_SEARCH_USD = 10.0 / 1000   # web search is billed per search, on top of tokens


def add_prices(prices: dict[str, tuple[float, float]]) -> None:
    """Prices for models this table doesn't know (e.g. OpenAI or Hugging Face
    models), from SORA_PRICES. Provider prefixes are not part of the name."""
    PRICES.update(prices)


def known_price(model: str) -> bool:
    return model in PRICES


def cost(model: str, input_tokens: int, output_tokens: int, web_searches: int = 0,
         cache_read: int = 0, cache_write: int = 0) -> float:
    """Unknown models count as $0 for tokens, so a new model never crashes a turn."""
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    price_read = CACHE_READ.get(model, price_in * 0.1)
    tokens = (input_tokens * price_in + output_tokens * price_out
              + cache_read * price_read + cache_write * price_in * CACHE_WRITE_MULTIPLIER)
    return tokens / 1_000_000 + web_searches * WEB_SEARCH_USD

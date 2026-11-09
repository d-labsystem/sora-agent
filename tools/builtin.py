"""Built-in tools: how the model manages its own memory and skills.

The harness already recalls memory before each turn (memory/gate.py) and
consolidates after (memory/consolidation.py). These tools add the agent's
side: it can choose to remember something now, search again with better
words, forget on request, and load or write skills.

Each tool is a small closure over the stores, built per agent. The current
session and turn live in a TurnContext that the harness updates every turn,
so anything a tool writes is tagged with where it came from.

Tool descriptions are prompts. They say when to use the tool, not only what
it does, because that is what the model reads when it decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from memory.skills import SkillLibrary
from memory.store import MemoryStore
from tools.registry import Tool, ToolError, ToolRegistry

MAX_FACT = 500


@dataclass
class TurnContext:
    session_id: str | None = None
    turn_id: str | None = None


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required}


def builtin_tools(store: MemoryStore, skills: SkillLibrary, context: TurnContext) -> list[Tool]:

    def remember(fact: str) -> str:
        fact = " ".join(fact.split())
        if not fact:
            raise ToolError("fact is empty")
        if len(fact) > MAX_FACT:
            raise ToolError(f"keep facts under {MAX_FACT} characters: one fact per call")
        if not store.add_fact(fact, source=f"remember:{context.turn_id or '-'}"):
            return f"Already known: {fact}"
        return f"Remembered: {fact}"

    def search_memory(query: str) -> str:
        facts = store.search_facts(query, 8)
        episodes = store.search_episodes(query, 5)
        if not facts and not episodes:
            return "Nothing in memory matches. Try other words, or ask the user."
        parts = []
        if facts:
            parts.append("Facts:\n" + "\n".join(f"- {f}" for f in facts))
        if episodes:
            parts.append("Past conversations:\n" + "\n".join(f"- {e}" for e in episodes))
        return "\n\n".join(parts)

    def forget(text: str) -> str:
        if len(text.strip()) < 3:
            raise ToolError("give at least 3 characters of the fact to forget")
        count = store.forget(text)
        return f"Forgot {count} fact(s) containing {text!r}." if count else \
            f"No fact contains {text!r}. Use search_memory to find the exact wording."

    def current_time() -> str:
        now = datetime.now().astimezone()
        return now.strftime("%A %d %B %Y, %H:%M %Z (UTC%z)")

    def load_skill(name: str) -> str:
        skill = skills.get(name.strip().lower())
        if skill is None:
            raise ToolError(f"No skill named {name!r}. Available:\n{skills.catalog() or '(none)'}")
        return f"Skill {skill.name}: {skill.description}\n\n{skill.body}"

    def save_skill(name: str, description: str, triggers: list, steps: str) -> str:
        try:
            path = skills.save(name, description, [str(t) for t in triggers], steps,
                               source_turn=context.turn_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        return f"Saved skill {name!r} to {path}. It will load when a message mentions: {', '.join(map(str, triggers))}."

    return [
        Tool("remember",
             "Save one durable fact about the user or their world (people, preferences, plans, "
             "decisions). Use when the user tells you something worth knowing next week, or asks "
             "you to remember. Write it standalone, with names instead of pronouns.",
             _schema({"fact": {"type": "string"}}), remember),
        Tool("search_memory",
             "Search saved facts and summaries of past conversations. Use when the user refers to "
             "something from before and the memory already in your prompt doesn't cover it. "
             "Search with names and topics, not whole sentences.",
             _schema({"query": {"type": "string"}}), search_memory),
        Tool("forget",
             "Delete saved facts that contain the given text. Use only when the user asks you to "
             "forget something, or a fact is now wrong (then remember the new one).",
             _schema({"text": {"type": "string"}}), forget),
        Tool("current_time",
             "The current local date, time and timezone. Use for anything relative to now: "
             "today, tomorrow, deadlines, ages, 'how long until'.",
             _schema({}), current_time),
        Tool("load_skill",
             "Load the full steps of a skill listed under 'Skills' in your instructions. Use when "
             "a request fits a skill whose steps aren't already in your prompt.",
             _schema({"name": {"type": "string"}}), load_skill),
        Tool("save_skill",
             "Save a reusable procedure as a skill, after you completed a multi-step task in a way "
             "that worked and is likely to recur, or when the user asks you to. Steps should be "
             "short, numbered and specific. Triggers are words a future request would contain.",
             _schema({"name": {"type": "string", "description": "lowercase-with-dashes"},
                      "description": {"type": "string"},
                      "triggers": {"type": "array", "items": {"type": "string"}},
                      "steps": {"type": "string"}}), save_skill),
    ]


def register_builtins(registry: ToolRegistry, store: MemoryStore, skills: SkillLibrary,
                      context: TurnContext) -> None:
    for tool in builtin_tools(store, skills, context):
        registry.register(tool)

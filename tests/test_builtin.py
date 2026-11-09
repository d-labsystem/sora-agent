"""Built-in tools, called the way the model calls them: through the registry."""

import pytest

from memory.skills import SkillLibrary
from memory.store import MemoryStore
from tools.builtin import TurnContext, register_builtins
from tools.registry import ToolRegistry


@pytest.fixture
def setup(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    skills = SkillLibrary(tmp_path / "skills")
    context = TurnContext("s1", "turn42")
    registry = ToolRegistry()
    register_builtins(registry, store, skills, context)
    return registry, store, skills


def run(registry, tool, /, **args):
    return registry.execute(tool, args)


def test_all_tools_registered_with_valid_schemas(setup):
    registry, _, _ = setup
    schemas = registry.schemas()
    assert {s["name"] for s in schemas} == {
        "remember", "search_memory", "forget", "current_time", "load_skill", "save_skill"}
    assert all(s["input_schema"]["type"] == "object" for s in schemas)


def test_remember_search_forget(setup):
    registry, store, _ = setup
    assert run(registry, "remember", fact="Priya   lives in Pune") == ("Remembered: Priya lives in Pune", False)
    assert run(registry, "remember", fact="Priya lives in Pune")[0].startswith("Already known")
    assert store.list_facts()[0]["source"] == "remember:turn42"
    output, error = run(registry, "search_memory", query="Pune")
    assert not error and "- Priya lives in Pune" in output
    assert run(registry, "forget", text="Pune")[0].startswith("Forgot 1")
    assert "Nothing in memory" in run(registry, "search_memory", query="Pune")[0]


def test_bad_inputs_become_readable_errors(setup):
    registry, _, _ = setup
    assert run(registry, "remember", fact="   ") == ("fact is empty", True)
    assert run(registry, "remember", fact="x" * 600)[1]
    assert run(registry, "forget", text="a")[1]
    assert run(registry, "remember") == ("Invalid arguments: missing fact", True)


def test_current_time_has_timezone(setup):
    output, error = run(setup[0], "current_time")
    assert not error and "UTC" in output


def test_skills_save_then_load_with_provenance(setup):
    registry, _, skills = setup
    output, error = run(registry, "save_skill", name="weekly-review",
                        description="Review the week.", triggers=["weekly review"], steps="1. List wins")
    assert not error and "weekly review" in output
    skill = skills.get("weekly-review")
    assert skill.extra["source_turn"] == "turn42" and len(skill.extra["created"]) == 10
    output, error = run(registry, "load_skill", name="Weekly-Review")
    assert not error and output.endswith("1. List wins")


def test_skill_errors_are_actionable(setup):
    registry, _, _ = setup
    output, error = run(registry, "load_skill", name="nope")
    assert error and "research-brief" in output          # lists what does exist
    output, error = run(registry, "save_skill", name="Bad Name", description="d",
                        triggers=["t"], steps="s")
    assert error and "lowercase" in output

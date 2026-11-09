"""Skills: parsing, matching, overriding, and saving."""

import pytest

from memory.skills import SkillLibrary, parse


@pytest.fixture
def library(tmp_path):
    return SkillLibrary(tmp_path / "skills")


def test_bundled_skill_loads(library):
    skill = library.get("research-brief")
    assert skill.bundled and "research" in skill.triggers and skill.body.startswith("1.")
    assert "- research-brief: Research a topic" in library.catalog()
    assert library.errors == []


def test_match_whole_words_best_first(library):
    library.save("trip-planning", "Plan a trip.", ["trip", "plan a trip", "compare"], "Steps")
    assert [s.name for s in library.match("Help me plan a trip to Pune")] == ["trip-planning"]
    assert library.match("my briefcase is lost") == []
    # "compare" fires both; trip-planning wins with two hits
    names = [s.name for s in library.match("compare trip options")]
    assert names == ["trip-planning", "research-brief"]


def test_save_round_trips_and_overrides(library, tmp_path):
    path = library.save("research-brief", "My own version.", ["dig into"], "Just my steps.")
    assert path == tmp_path / "skills" / "research-brief" / "SKILL.md"
    skill = library.get("research-brief")
    assert not skill.bundled and skill.body == "Just my steps." and skill.triggers == ["dig into"]
    assert library.delete("research-brief")
    assert library.get("research-brief").bundled          # bundled one is back
    assert not library.delete("research-brief")           # bundled ones can't be deleted


@pytest.mark.parametrize("name, desc, triggers, body, error", [
    ("../evil", "d", ["t"], "b", "lowercase"),
    ("Has Space", "d", ["t"], "b", "lowercase"),
    ("ok", "", ["t"], "b", "required"),
    ("ok", "d", [" "], "b", "trigger"),
    ("ok", "d", ["t"], "x" * 9000, "over"),
])
def test_save_rejects_bad_input(library, name, desc, triggers, body, error):
    with pytest.raises(ValueError, match=error):
        library.save(name, desc, triggers, body)


def test_newlines_cannot_inject_frontmatter(library):
    library.save("ok", "line one\ntriggers: everything", ["a,b", "c"], "body")
    skill = library.get("ok")
    assert skill.description == "line one triggers: everything"
    assert skill.triggers == ["a b", "c"]


def test_broken_file_is_reported_not_fatal(library, tmp_path):
    bad = tmp_path / "skills" / "broken" / "SKILL.md"
    bad.parent.mkdir(parents=True)
    bad.write_text("no frontmatter here")
    assert "research-brief" in library.all()
    assert "missing" in library.errors[0]


def test_parse_defaults_name_to_folder(tmp_path):
    skill = parse("---\ndescription: d\n---\nbody", tmp_path / "my-skill" / "SKILL.md")
    assert skill.name == "my-skill" and skill.triggers == []

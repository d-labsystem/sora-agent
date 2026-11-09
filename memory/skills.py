"""Skills: reusable procedures the agent can follow, and write.

Lessons (memory/learning.py) are short "don't do X" notes. A skill is the
positive version: a written-down procedure for a kind of task, like a recipe.
Each one is a folder with a SKILL.md file:

    ---
    name: research-brief
    description: Research a topic and answer with a short, sourced brief.
    triggers: research, look into, compare
    ---
    1. Restate the question ...

Two levels of loading keep the prompt small ("progressive disclosure"):

    catalog   every skill's name and description, always in the prompt (cheap)
    match     the full body of up to two skills whose triggers appear in the
              message, so the agent sees the steps exactly when they matter

Skills live in two places. Bundled ones ship with the code (sora/skills);
the user's live in ~/.sora/skills, where the save_skill tool writes. A user
skill with the same name overrides the bundled one.

Skills are plain Markdown on purpose: you can read, edit, version and share
them without the agent, and the format matches Claude's own skills.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

BUNDLED = Path(__file__).resolve().parent.parent / "skills"
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,49}$")
MAX_BODY = 8000
MAX_MATCHES = 2


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str]
    body: str
    path: Path
    bundled: bool = False
    extra: dict = field(default_factory=dict)


def parse(text: str, path: Path, bundled: bool = False) -> Skill:
    """Read a SKILL.md. The frontmatter is simple `key: value` lines, so we
    parse it by hand instead of adding a YAML dependency."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not match:
        raise ValueError(f"{path}: missing --- frontmatter ---")
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    name = meta.pop("name", path.parent.name)
    if not NAME.match(name):
        raise ValueError(f"{path}: bad skill name {name!r}")
    triggers = [t.strip().lower() for t in meta.pop("triggers", "").split(",") if t.strip()]
    return Skill(name, meta.pop("description", ""), triggers, match.group(2).strip(),
                 path, bundled, meta)


def render(name: str, description: str, triggers: list[str], body: str,
           **extra: str | None) -> str:
    """Write a SKILL.md. `extra` adds header lines such as source_turn and
    created, so every saved skill can be traced back to the turn that made it."""
    one_line = lambda s: " ".join(str(s).split())
    header = [f"name: {name}", f"description: {one_line(description)}",
              f"triggers: {', '.join(one_line(t).replace(',', ' ') for t in triggers)}"]
    header += [f"{key}: {one_line(value)}" for key, value in extra.items() if value]
    return "---\n" + "\n".join(header) + f"\n---\n{body.strip()}\n"


class SkillLibrary:
    def __init__(self, user_dir: Path, bundled_dir: Path = BUNDLED):
        self.user_dir = user_dir
        self.bundled_dir = bundled_dir
        self.errors: list[str] = []

    def all(self) -> dict[str, Skill]:
        """Read from disk every time: there are few files, and edits made by
        hand show up on the next turn without a restart."""
        skills: dict[str, Skill] = {}
        self.errors = []
        for folder, bundled in ((self.bundled_dir, True), (self.user_dir, False)):
            for path in sorted(folder.glob("*/SKILL.md")) if folder.is_dir() else []:
                try:
                    skill = parse(path.read_text(encoding="utf-8"), path, bundled)
                    skills[skill.name] = skill      # user skills override bundled ones
                except (OSError, ValueError, UnicodeDecodeError) as exc:
                    self.errors.append(str(exc))
        return skills

    def get(self, name: str) -> Skill | None:
        return self.all().get(name)

    def catalog(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self.all().values())

    def match(self, message: str, limit: int = MAX_MATCHES) -> list[Skill]:
        """Skills whose trigger phrases appear in the message, best first.
        Whole-word matching, so the trigger "brief" doesn't fire on "briefcase"."""
        text = message.lower()
        scored = []
        for skill in self.all().values():
            hits = sum(1 for t in skill.triggers
                       if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text))
            if hits:
                scored.append((hits, skill.name, skill))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _, _, skill in scored[:limit]]

    def save(self, name: str, description: str, triggers: list[str], body: str,
             source_turn: str | None = None) -> Path:
        """Write a user skill. Raises ValueError with a message the model can act on."""
        name = name.strip().lower()
        if not NAME.match(name):
            raise ValueError("name must be lowercase letters, digits and dashes, up to 50 characters")
        if not description.strip() or not body.strip():
            raise ValueError("description and body are both required")
        if not any(t.strip() for t in triggers):
            raise ValueError("give at least one trigger phrase")
        if len(body) > MAX_BODY:
            raise ValueError(f"body is over {MAX_BODY} characters; keep skills short and specific")
        path = self.user_dir / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        created = datetime.now().astimezone().strftime("%Y-%m-%d")
        path.write_text(render(name, description, [t for t in triggers if t.strip()], body,
                               source_turn=source_turn, created=created), encoding="utf-8")
        return path

    def delete(self, name: str) -> bool:
        """Delete a user skill. Bundled skills are part of the code: edit them there."""
        path = self.user_dir / name / "SKILL.md"
        if not NAME.match(name) or not path.is_file():
            return False
        path.unlink()
        path.parent.rmdir() if not any(path.parent.iterdir()) else None
        return True

"""Approvals: actions that reach other people wait for the user's reply.

Some tool calls can't be undone and act in the user's name: a calendar
invite emails every attendee. The model decides to call tools, so the model
alone must not be able to make those happen. The rule:

    a guarded call is never run when the model makes it. It is stored as a
    draft, and the model is told to show it to the user. The `approve` tool
    runs a draft only in a LATER turn of the SAME conversation, before it
    expires, and runs it exactly as drafted.

Why "a later turn". A turn starts only when the user sends a message, so a
later turn means the user has seen the draft and answered. Within one turn
the model can call tools as often as it likes but can't write the user's
message, so it can't approve its own draft, even if a calendar description
or a web page it just read tells it to.

Why "exactly as drafted". approve() takes only the draft id; the arguments
come from storage. Swapping the attendees after the user said yes is
impossible.

What it doesn't decide: whether the reply was a yes. The model reads the
reply and calls approve only on a clear yes (its instructions say so); the
guarantee is that a human was asked and answered first.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from tools.registry import Tool, ToolError, ToolRegistry

TTL = 600               # seconds a draft stays approvable


@dataclass
class Draft:
    id: str
    tool: str
    args: dict
    summary: str
    session_id: str | None
    turn_id: str | None
    created: float


def invite_rule(args: dict) -> str | None:
    """create_event needs approval only when it would email attendees."""
    people = [a for a in args.get("attendees") or [] if str(a).strip()]
    if not people:
        return None
    return (f"Create {args.get('title')!r} from {args.get('start')} to {args.get('end')} and email "
            f"invitations to {', '.join(people)}")


RULES: dict[str, Callable[[dict], str | None]] = {
    "calendar__create_event": invite_rule,
}


class Approvals:
    def __init__(self, context, rules: dict | None = None, ttl: float = TTL, clock=time.time):
        self.context, self.rules, self.ttl, self.clock = context, rules or RULES, ttl, clock
        self.drafts: dict[str, Draft] = {}
        self.originals: dict[str, Callable[..., str]] = {}

    def guard(self, registry: ToolRegistry) -> list[str]:
        """Wrap every registered tool that has a rule. Returns their names."""
        guarded = []
        for name, rule in self.rules.items():
            tool = registry.get(name)
            if tool is None or name in self.originals:
                continue
            self.originals[name] = tool.function
            tool.function = self._held(name, rule)
            guarded.append(name)
        return guarded

    def _held(self, name: str, rule):
        def call(**args) -> str:
            summary = rule(args)
            if summary is None:
                return self.originals[name](**args)
            draft = Draft(f"d{secrets.token_hex(3)}", name, dict(args), summary,
                          self.context.session_id, self.context.turn_id, self.clock())
            self.drafts[draft.id] = draft
            return (f"NOT DONE YET: this needs the user's approval first.\nDraft {draft.id}: {summary}.\n"
                    f"Show the user exactly this and ask whether to go ahead. Only if they clearly say yes "
                    f"in their reply, call approve(draft_id='{draft.id}'). It expires in {int(self.ttl // 60)} minutes.")
        return call

    def approve(self, draft_id: str) -> str:
        draft = self.drafts.get(draft_id.strip())
        if draft is None:
            raise ToolError(f"No pending draft {draft_id!r} (it may have run, expired, or never existed).")
        if draft.session_id != self.context.session_id:
            raise ToolError("That draft belongs to another conversation.")
        if draft.turn_id == self.context.turn_id:
            raise ToolError("The user hasn't replied yet. Show them the draft and wait for their answer; "
                            "a draft can only be approved in a later message.")
        if self.clock() - draft.created > self.ttl:
            del self.drafts[draft.id]
            raise ToolError("That draft expired. Make it again and ask the user to confirm.")
        del self.drafts[draft.id]                      # one approval runs it once
        return self.originals[draft.tool](**draft.args)

    def discard(self, draft_id: str) -> str:
        return ("Discarded draft " + draft_id) if self.drafts.pop(draft_id.strip(), None) else f"No draft {draft_id!r}."

    def tools(self) -> list[Tool]:
        schema = {"type": "object", "properties": {"draft_id": {"type": "string"}}, "required": ["draft_id"]}
        return [
            Tool("approve", "Carry out an action that was held for the user's approval, by its draft id. "
                 "Call it only when the user's latest message clearly says yes to that draft.",
                 schema, self.approve),
            Tool("discard_draft", "Drop a held action the user declined or changed their mind about.",
                 schema, self.discard),
        ]

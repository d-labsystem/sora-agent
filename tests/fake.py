"""A scripted model, so tests and the UI demo need no API key or network."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS


def text(value: str):
    return NS(type="text", text=value)


def tool_use(name: str, args: dict, call_id: str = "call_1"):
    return NS(type="tool_use", id=call_id, name=name, input=args)


def message(*blocks, stop_reason: str = "end_turn"):
    return NS(content=list(blocks), stop_reason=stop_reason,
              usage=NS(input_tokens=10, output_tokens=5))


class _Stream:
    def __init__(self, response):
        self.response = response
        self.text_stream = [b.text for b in response.content if b.type == "text"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.response


class FakeClient:
    """`script` is a list of loop responses, or a callable taking the request.
    `small` answers messages.create: a dict is returned as JSON, an exception
    is raised, and a list is consumed in order."""

    def __init__(self, script=(), small=None):
        self.script = script if callable(script) else list(script)
        self.small = small if small is not None else {"retrieve": False, "query": "", "reason": "n/a"}
        self.requests = []
        self.small_requests = []
        self.beta = NS(messages=NS(stream=self._stream))
        self.messages = NS(create=self._create)

    def _stream(self, **request):
        self.requests.append(copy.copy({**request, "messages": list(request["messages"])}))
        response = self.script(request) if callable(self.script) else self.script.pop(0)
        return _Stream(response)

    def _create(self, **request):
        self.small_requests.append(request)
        answer = self.small.pop(0) if isinstance(self.small, list) else self.small
        if callable(answer):
            answer = answer(request)
        if isinstance(answer, Exception):
            raise answer
        return message(text(json.dumps(answer)))

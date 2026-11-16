"""Long tasks: compaction between steps and the dollar cap, offline (no key, no cost)."""

from types import SimpleNamespace as NS

from fake import FakeClient, text, tool_use
from harness.app import Settings, Sora
from loop.agent import BUDGET_NOTE, run_loop
from loop.compaction import CONTINUE, SUMMARY_PROMPT, context_size
from loop.providers import COMPACT_BETA, OpenAICompatProvider, Usage
from test_providers import FakeOpenAI, call_piece, chunk, echo_registry, usage


def big(*blocks, stop_reason="end_turn", tokens=9000):
    """A response whose context is `tokens` big (input) plus 100 output."""
    return NS(content=list(blocks), stop_reason=stop_reason, model="claude-opus-5-5",
              usage=NS(input_tokens=tokens, output_tokens=100, cache_read_input_tokens=0,
                       cache_creation_input_tokens=0, server_tool_use=None))


BLOCK = NS(type="compaction", content="User asked to echo twice; first echo done: 'echo a'.",
           signature="sig-123", encrypted_content=None)


def compaction_response(ok=True):
    """What on-demand compaction returns: no top-level tokens, the cost in iterations."""
    return NS(content=[BLOCK] if ok else [], stop_reason="compaction" if ok else "max_tokens",
              model="claude-opus-5-5",
              usage=NS(input_tokens=0, output_tokens=0, iterations=[
                  NS(type="compaction", input_tokens=9100, output_tokens=300,
                     cache_read_input_tokens=0, cache_creation_input_tokens=0)]))


class CompactingClient(FakeClient):
    """FakeClient plus beta.messages.create for compaction requests."""

    def __init__(self, script, summaries):
        super().__init__(script)
        self.summaries = list(summaries)
        self.compaction_requests = []
        self.beta.messages.create = self._compact

    def _compact(self, **request):
        self.compaction_requests.append({**request, "messages": list(request["messages"])})
        return self.summaries.pop(0)


def spans():
    seen = []
    return seen, lambda kind, **data: seen.append((kind, data))


def test_context_size_counts_everything_the_next_step_resends():
    assert context_size(Usage(input_tokens=100, output_tokens=50, cache_read=1000, cache_write=10)) == 1160


def test_claude_compacts_after_tool_results_and_continues_from_the_block():
    client = CompactingClient(
        [big(tool_use("echo", {"x": "a"}, "c1"), stop_reason="tool_use"),
         big(text("Both done."), tokens=800)],
        [compaction_response()])
    seen, notify = spans()
    messages = [{"role": "user", "content": "echo a, then b"}]
    result = run_loop(client, "claude-opus-5-5", "sys", messages, echo_registry(), notify,
                      compact_at=5000)
    assert result.reply == "Both done." and result.compactions == 1

    sent = client.compaction_requests[0]
    assert sent["compaction"] == {"type": "summarize", "instructions": SUMMARY_PROMPT}
    assert sent["betas"] == [COMPACT_BETA] and sent["system"] == "sys"
    # never mid tool round: the tool result is already in the conversation
    assert sent["messages"][-1]["content"][0]["type"] == "tool_result"

    # the next step starts from the block, inside a user message, with the beta
    after = client.requests[1]
    first = after["messages"][0]
    assert len(after["messages"]) == 1 and first["role"] == "user"
    assert first["content"][0] is BLOCK
    assert first["content"][1]["text"] == CONTINUE.format(task="echo a, then b")
    assert COMPACT_BETA in after["betas"]

    # the summary's cost comes from usage.iterations, and it is an llm span
    llm = [d for kind, d in seen if kind == "llm" and d["name"] == "compaction"][0]
    assert (llm["input_tokens"], llm["output_tokens"]) == (9100, 300) and llm["cost_usd"] > 0
    event = [d for kind, d in seen if kind == "compaction"][0]
    assert event["ok"] and event["tokens_before"] == 9100 and "echo a" in event["summary"]
    assert result.input_tokens == 9000 + 9100 + 800


def test_a_failed_summary_carries_on_and_waits_for_more_growth():
    client = CompactingClient(
        [big(tool_use("echo", {"x": "a"}, "c1"), stop_reason="tool_use"),
         big(tool_use("echo", {"x": "b"}, "c2"), stop_reason="tool_use", tokens=9500),
         big(text("done"), tokens=9600)],
        [compaction_response(ok=False)])
    seen, notify = spans()
    messages = [{"role": "user", "content": "go"}]
    result = run_loop(client, "claude-opus-5-5", "sys", messages, echo_registry(), notify,
                      compact_at=5000)
    assert result.reply == "done" and result.compactions == 0
    assert len(client.compaction_requests) == 1          # 9,600 < 9,100 x 1.25: no second try
    assert len(messages) == 6                            # nothing was replaced (3 steps, 2 results, 1 ask)
    assert [d["ok"] for kind, d in seen if kind == "compaction"] == [False]


def test_budget_cap_stops_tools_and_answers_with_what_it_has():
    client = CompactingClient(
        [big(tool_use("echo", {"x": "a"}, "c1"), stop_reason="tool_use"),   # 9,000 in: $0.038
         big(tool_use("echo", {"x": "b"}, "c2"), stop_reason="tool_use"),
         big(text("Partial: only a."), tokens=500)], [])
    seen, notify = spans()
    result = run_loop(client, "claude-opus-5-5", [{"type": "text", "text": "sys"}],
                      [{"role": "user", "content": "go"}], echo_registry(), notify,
                      max_iterations=60, budget_usd=0.05)
    assert result.budget_reached and not result.limit_reached
    assert result.reply == "Partial: only a." and result.iterations == 2
    final = client.requests[2]
    assert final["tool_choice"] == {"type": "none"} and final["system"] == client.requests[0]["system"]
    assert final["messages"][-1]["content"][-1] == {"type": "text", "text": BUDGET_NOTE}


def test_other_providers_compact_with_their_own_summary():
    client = FakeOpenAI(streams=[
        [chunk(calls=[call_piece(0, id="call_1", name="echo", args='{"x": "a"}')]),
         chunk(finish="tool_calls"), chunk(usage=usage(prompt=9000, completion=50))],
        [chunk(content="<summary>Echoed a; b is next.</summary>"), chunk(finish="stop")],   # the summary
        [chunk(content="Both done."), chunk(finish="stop")],
    ])
    messages = [{"role": "user", "content": "echo a, then b"}]
    seen, notify = spans()
    result = run_loop(OpenAICompatProvider(client), "gpt-5.4-mini", "sys", messages, echo_registry(),
                      notify, compact_at=5000)
    assert result.reply == "Both done." and result.compactions == 1
    summary_request = client.requests[1]
    assert summary_request["messages"][-1] == {"role": "user", "content": SUMMARY_PROMPT}
    assert "tools" not in summary_request                 # the summary call gets no tools
    continued = client.requests[2]["messages"]
    assert continued[0]["role"] == "system" and len(continued) == 2
    assert "Echoed a; b is next." in continued[1]["content"] and "echo a, then b" in continued[1]["content"]


def test_long_turns_get_more_steps_a_cap_and_a_status(tmp_path):
    script = [big(tool_use("echo", {"x": str(i)}, f"c{i}"), stop_reason="tool_use", tokens=100)
              for i in range(12)] + [big(text("stopped"), tokens=100)]
    # each step: 100 in x $4/M + 100 out x $20/M = $0.0024, so a $0.006 cap stops after step 3
    agent = Sora(Settings(home=tmp_path, long_budget_usd=0.006, compact_at=0), FakeClient(script),
                 connect_mcp=False)
    agent.registry = echo_registry()
    turn = agent.respond("keep echoing", long=True)
    assert turn["status"] == "budget" and turn["checks"]["within_budget"] is False
    assert turn["iterations"] == 3                        # then one final tools-off call

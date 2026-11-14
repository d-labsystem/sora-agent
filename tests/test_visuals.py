"""Visual tools: validation, and visuals saved with the reply."""

import pytest

from fake import FakeClient, message, text, tool_use
from harness.app import Settings, Sora
from test_app import SmallModel
from tools.registry import ToolError
from tools.visuals import show_chart, show_diagram, show_table

SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>'


def test_chart_validation():
    ok = show_chart("bar", "Cost", ["a", "b"], [{"name": "x", "values": [1, 2.5]}])
    assert ok.startswith("Chart shown to the user")
    for args, error in [(("radar", "t", ["a"], [{"values": [1]}]), "type"),
                        (("bar", "t", ["a", "b"], [{"values": [1]}]), "has 1 values"),
                        (("bar", "t", ["a"], [{"values": ["1"]}]), "must be a number"),
                        (("bar", "t", ["a"], [{"values": [True]}]), "must be a number"),
                        (("pie", "t", ["a"], [{"values": [1]}, {"values": [2]}]), "exactly one"),
                        (("pie", "t", ["a"], [{"values": [-1]}]), "negative")]:
        with pytest.raises(ToolError, match=error):
            show_chart(*args)


def test_table_validation():
    assert "2 rows x 2 columns" in show_table("t", ["a", "b"], [[1, "x"], [2, None]])
    with pytest.raises(ToolError, match="row 0"):
        show_table("t", ["a", "b"], [[1]])
    with pytest.raises(ToolError, match="isn't text"):
        show_table("t", ["a"], [[{"nested": 1}]])


@pytest.mark.parametrize("svg, error", [
    ("<div>hi</div>", "complete"),
    ('<svg viewBox="0 0 1 1"></svg>', "xmlns"),
    ('<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>', "scripts"),
    ('<svg xmlns="http://www.w3.org/2000/svg"><rect onclick="x()"/></svg>', "scripts"),
    ('<svg xmlns="http://www.w3.org/2000/svg"><a href="javascript:x()"/></svg>', "scripts"),
    ('<svg xmlns="http://www.w3.org/2000/svg"><foreignObject/></svg>', "scripts"),
])
def test_unsafe_or_broken_svg_rejected(svg, error):
    with pytest.raises(ToolError, match=error):
        show_diagram("t", svg)


def test_visuals_saved_with_the_reply(tmp_path):
    chart = {"type": "line", "title": "Cost", "labels": ["Mon", "Tue"], "series": [{"name": "$", "values": [1, 2]}]}
    client = FakeClient([
        message(tool_use("show_chart", chart, "c1"), tool_use("show_diagram", {"title": "d", "svg": SVG}, "c2"),
                tool_use("show_table", {"title": "bad", "columns": ["a"], "rows": [[1, 2]]}, "c3"),
                stop_reason="tool_use"),
        message(text("Here's the chart.")),
        message(text("ok")),
    ], small=SmallModel())
    agent = Sora(Settings(home=tmp_path), client, connect_mcp=False)
    turn = agent.respond("chart my costs")
    saved = agent.store.messages(turn["session_id"])[1]
    assert [v["tool"] for v in saved["visuals"]] == ["show_chart", "show_diagram"]   # the failed table isn't kept
    assert saved["visuals"][0]["labels"] == ["Mon", "Tue"]
    assert {"show_chart", "show_table", "show_diagram"} <= {t["name"] for t in client.requests[0]["tools"]}
    agent.respond("thanks", session_id=turn["session_id"])
    note = client.requests[-1]["messages"][1]["content"]
    assert "show_diagram(title='d')" in note and "<svg" not in note                    # big args stay out of history

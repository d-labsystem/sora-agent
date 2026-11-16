"""Attachments, the bare-model baseline, and GAIA's scorer: offline."""

import zipfile

import pytest

from fake import FakeClient, message, text
from harness.app import BARE_SYSTEM, Settings, Sora
from harness.attachments import docx_text, pptx_text, prepare, safe_unzip, xlsx_text
from ops.bench import gaia
from ops.bench.runner import Item, sample

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
M = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, body in files.items():
            z.writestr(name, body)
    return path


def xlsx(path):
    return make_zip(path, {
        "xl/workbook.xml": f'<workbook {M} {R}><sheets><sheet name="Sales" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                      '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/sharedStrings.xml": f'<sst {M}><si><t>Item</t></si><si><t>Price</t></si><si><t>Tea</t></si></sst>',
        "xl/worksheets/sheet1.xml": f'<worksheet {M}><sheetData>'
                                    '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                                    '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="C2"><v>4.5</v></c></row>'
                                    '</sheetData></worksheet>'})


def test_office_files_become_text(tmp_path):
    assert xlsx_text(xlsx(tmp_path / "s.xlsx")) == "## Sheet: Sales\nItem,Price\nTea,,4.5"     # gap kept: column C
    docx = make_zip(tmp_path / "d.docx", {"word/document.xml":
        f'<w:document {W}><w:body><w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>world</w:t></w:r></w:p>'
        '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
        '</w:body></w:document>'})
    assert docx_text(docx) == "Hello world\na | b"
    pptx = make_zip(tmp_path / "p.pptx", {
        "ppt/slides/slide2.xml": f'<p:sld {A} xmlns:p="x"><a:p><a:r><a:t>Second</a:t></a:r></a:p></p:sld>',
        "ppt/slides/slide10.xml": f'<p:sld {A} xmlns:p="x"><a:p><a:r><a:t>Tenth</a:t></a:r></a:p></p:sld>',
        "ppt/slides/slide1.xml": f'<p:sld {A} xmlns:p="x"><a:p><a:r><a:t>First</a:t></a:r></a:p></p:sld>'})
    assert pptx_text(pptx) == "## Slide 1\nFirst\n\n## Slide 2\nSecond\n\n## Slide 3\nTenth"   # numeric order


def test_prepare_shows_images_and_pdfs_and_files_the_rest(tmp_path):
    src, ws = tmp_path / "in", tmp_path / "ws"
    src.mkdir()
    (src / "chart.png").write_bytes(b"\x89PNG fake")
    (src / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    (src / "data.csv").write_text("a,b\n1,2\n")
    (src / "talk.mp3").write_bytes(b"ID3")
    xlsx(src / "book.xlsx")
    p = prepare([src / n for n in ("chart.png", "paper.pdf", "data.csv", "talk.mp3", "book.xlsx", "gone.txt")], ws)
    assert [b["type"] for b in p.blocks] == ["image", "document"]
    assert p.blocks[0]["source"]["media_type"] == "image/png"
    assert (ws / "data.csv").read_text() == "a,b\n1,2\n" and (ws / "book.xlsx.txt").read_text().startswith("## Sheet")
    notes = p.text()
    assert "chart.png: an image, shown to you above" in notes and "'book.xlsx.txt'" in notes
    assert "talk.mp3: audio or video" in notes and "gone.txt: missing" in notes


def test_zips_unpack_safely(tmp_path):
    ws = tmp_path / "ws"
    xlsx(tmp_path / "inner.xlsx")
    make_zip(tmp_path / "bundle.zip", {"notes/readme.txt": "hi", "inner.xlsx": (tmp_path / "inner.xlsx").read_bytes()})
    p = prepare([tmp_path / "bundle.zip"], ws)
    assert (ws / "bundle" / "notes" / "readme.txt").read_text() == "hi"
    assert (ws / "bundle" / "inner.xlsx.txt").is_file() and "bundle/inner.xlsx.txt" in p.text()
    evil = make_zip(tmp_path / "evil.zip", {"../../escaped.txt": "gotcha"})
    with pytest.raises(ValueError, match="outside the workspace"):
        safe_unzip(evil, ws / "evil")
    assert not (tmp_path / "escaped.txt").exists()
    assert "not unpacked" in prepare([evil], ws).text()


GATE = {"retrieve": False, "query": "", "correction": False, "complex": False, "reason": "q"}


def test_respond_sends_attachments_and_stores_only_their_names(tmp_path):
    (tmp_path / "pic.png").write_bytes(b"\x89PNG fake")
    client = FakeClient([message(text("A cat."))], small=GATE)
    agent = Sora(Settings(home=tmp_path / "home", memory_mirror=False), client, connect_mcp=False)
    turn = agent.respond("What is this?", attachments=[tmp_path / "pic.png"])
    content = client.requests[0]["messages"][-1]["content"]
    assert content[0]["type"] == "image" and content[-1]["text"].startswith("What is this?\n\nAttached files:")
    stored = agent.store.messages(turn["session_id"])[0]["content"]
    assert "pic.png: an image" in stored and "base64" not in stored and "iVBOR" not in stored
    assert any(s["kind"] == "attachments" for s in turn["spans"])


def test_the_bare_baseline_has_none_of_soras_harness(tmp_path):
    client = FakeClient([message(text("42"))], small=GATE)
    agent = Sora(Settings(home=tmp_path, memory_mirror=False, bare=True), client, connect_mcp=False)
    agent.respond("What's 6 x 7?")
    request = client.requests[0]
    assert request["system"][0]["text"] == BARE_SYSTEM
    names = {t.get("name") for t in request["tools"]}
    assert not names & {"remember", "search_memory", "update_plan", "load_skill", "show_chart", "approve"}
    assert {"web_search", "web_fetch"} <= names
    assert client.small_requests == []                       # no gate, no consolidation, no reflection


# ---- GAIA's scorer: the behaviour of the benchmark's published scorer.py

@pytest.mark.parametrize("answer, gold, ok", [
    ("1234", "1234", True), ("$1,234", "1234", True), ("12%", "12", True), ("1234.0", "1234", True),
    ("1235", "1234", False), ("about 1234", "1234", False),
    ("Saint Petersburg", "Saint Petersburg", True), ("saint petersburg.", "Saint Petersburg", True),
    ("St. Petersburg", "Saint Petersburg", False),
    ("b, a", "a, b", False), ("A, 2", "a,2", True), ("a, b, c", "a, b", False), ("apple; 3", "apple, 3", True),
    (None, "x", False),
])
def test_gaia_scorer(answer, gold, ok):
    assert gaia.score(answer, gold) is ok


def test_final_answer_extraction_and_contamination_flag():
    assert gaia.final_answer("thinking...\nFINAL ANSWER: **Paris**") == "Paris"
    assert gaia.final_answer("Final answer: 3.5") == "3.5" and gaia.final_answer("Final Answer: Rome.") == "Rome"
    assert gaia.final_answer("final answer: a\nmore\nFINAL ANSWER: b") == "b" and gaia.final_answer("no") is None
    searched = {"spans": [{"kind": "server_tool", "name": "web_search", "data": {"input": {"query": "GAIA benchmark answer"}}}]}
    fine = {"spans": [{"kind": "server_tool", "name": "web_search", "data": {"input": {"query": "gaiatsu hotel"}}}]}
    assert gaia.contaminated(searched) and not gaia.contaminated(fine)


def test_stratified_sampling_keeps_each_levels_share():
    items = ([Item(f"a{i}", "", "", {"level": "level 1"}) for i in range(53)]
             + [Item(f"b{i}", "", "", {"level": "level 2"}) for i in range(86)]
             + [Item(f"c{i}", "", "", {"level": "level 3"}) for i in range(26)])
    picked = sample(items, 60, seed=0, strata="level")
    counts = {lvl: sum(i.meta["level"] == lvl for i in picked) for lvl in ("level 1", "level 2", "level 3")}
    assert counts == {"level 1": 19, "level 2": 31, "level 3": 10} and len(set(i.id for i in picked)) == 60
    assert [i.id for i in picked] == [i.id for i in sample(list(reversed(items)), 60, seed=0, strata="level")]


def test_gaia_loads_locally_and_attaches_files(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist([
        {"task_id": "t1", "Question": "How many?", "Level": 1, "Final answer": "7", "file_name": "",
         "file_path": "", "Annotator Metadata": None},
        {"task_id": "t2", "Question": "Total?", "Level": 2, "Final answer": "4.5", "file_name": "s.xlsx",
         "file_path": "2023/validation/s.xlsx", "Annotator Metadata": None}]), tmp_path / "metadata.parquet")
    xlsx(tmp_path / "files" / "s.xlsx") if (tmp_path / "files").mkdir() is None else None
    g = gaia.Gaia()
    items = g.load(tmp_path)
    assert [(i.id, i.meta["level"], i.meta["input"]) for i in items] == [("t1", "level 1", "none"),
                                                                         ("t2", "level 2", "spreadsheet")]
    assert "attachments" not in g.case(items[0]) and g.case(items[0])["message"].endswith("Question: How many?")
    assert g.case(items[1])["attachments"] == [str(tmp_path / "files" / "s.xlsx")]
    graded = g.grade(items[1], {"reply": "Sum is 4.5\nFINAL ANSWER: 4.50", "spans": []})
    assert graded["checks"] == {"correct": True} and graded["judge"]["extracted"] == "4.50"



def test_closed_book_has_no_tools_at_all(tmp_path):
    from ops.bench import CLOSED_BOOK
    client = FakeClient([message(text("Paris"))], small=GATE)
    agent = Sora(Settings(home=tmp_path, memory_mirror=False, **CLOSED_BOOK), client, connect_mcp=False)
    agent.respond("Capital of France?")
    assert client.requests[0].get("tools") in (None, [])

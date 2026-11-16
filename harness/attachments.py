"""Attachments: files sent along with a message.

Two ways a file reaches the model, depending on what it is:

    seen        images and PDFs go to the model itself, as content blocks:
                Claude sees images and reads PDFs (text and layout) natively
    read        everything else goes into the workspace, where the model reads
                it with workspace__read_file. Office files (xlsx, docx, pptx)
                are converted to text here first

Office files are zip archives of XML, so the standard library (zipfile and
ElementTree) can read them: no extra packages. A spreadsheet becomes one CSV
block per sheet, a document its paragraphs and tables, a deck its slides.

Zip archives are unpacked into the workspace, but every member path is
checked to stay inside it ("../../.ssh/id_rsa" in an archive is refused,
the zip-slip attack), and the total size is capped.

Not supported yet: audio and video (they need transcription). Such a file
is still copied into the workspace and the model is told it can't read it,
so it can say so instead of guessing.
"""

from __future__ import annotations

import base64
import csv
import io
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp"}
MAX_IMAGE = 5 * 1024 * 1024          # Anthropic's per-image limit
MAX_PDF = 32 * 1024 * 1024
MAX_UNZIPPED = 200 * 1024 * 1024
UNSUPPORTED = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".avi", ".mkv"}

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "rel": "http://schemas.openxmlformats.org/package/2006/relationships"}


@dataclass
class Prepared:
    blocks: list[dict] = field(default_factory=list)      # image/document blocks for the message
    notes: list[str] = field(default_factory=list)        # what the model is told about each file

    def text(self) -> str:
        return "Attached files:\n" + "\n".join(f"- {n}" for n in self.notes) if self.notes else ""


# ---- Office files as text

def _cell_col(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def xlsx_text(path: Path) -> str:
    """Each sheet as CSV, cell values only (formulas show their cached result)."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")) for si in root.findall("m:si", NS)]
        book = ElementTree.fromstring(z.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels.findall("rel:Relationship", NS)}
        out = []
        for sheet in book.find("m:sheets", NS):
            target = targets[sheet.get(f"{{{NS['r']}}}id")].lstrip("/")
            target = target if target.startswith("xl/") else f"xl/{target}"
            rows = []
            for row in ElementTree.fromstring(z.read(target)).iter(f"{{{NS['m']}}}row"):
                values = {}
                for c in row.findall("m:c", NS):
                    v, kind = c.find("m:v", NS), c.get("t")
                    if kind == "inlineStr":
                        text = "".join(t.text or "" for t in c.iter(f"{{{NS['m']}}}t"))
                    elif v is None:
                        continue
                    elif kind == "s":
                        text = shared[int(v.text)]
                    else:
                        text = v.text or ""
                    values[_cell_col(c.get("r"))] = text
                if values:
                    rows.append([values.get(i, "") for i in range(max(values) + 1)])
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            out.append(f"## Sheet: {sheet.get('name')}\n{buffer.getvalue().strip()}")
        return "\n\n".join(out)


def docx_text(path: Path) -> str:
    """Paragraphs in order; table rows as ' | '-joined cells."""
    with zipfile.ZipFile(path) as z:
        body = ElementTree.fromstring(z.read("word/document.xml")).find("w:body", NS)
    lines = []
    for block in body:
        if block.tag == f"{{{NS['w']}}}p":
            lines.append("".join(t.text or "" for t in block.iter(f"{{{NS['w']}}}t")))
        elif block.tag == f"{{{NS['w']}}}tbl":
            for row in block.iter(f"{{{NS['w']}}}tr"):
                lines.append(" | ".join("".join(t.text or "" for t in cell.iter(f"{{{NS['w']}}}t"))
                                        for cell in row.iter(f"{{{NS['w']}}}tc")))
    return "\n".join(lines).strip()


def pptx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                        key=lambda n: int(re.search(r"\d+", n.rsplit("/", 1)[1]).group(0)))
        out = []
        for i, name in enumerate(slides, 1):
            root = ElementTree.fromstring(z.read(name))
            paragraphs = ["".join(t.text or "" for t in p.iter(f"{{{NS['a']}}}t")) for p in root.iter(f"{{{NS['a']}}}p")]
            out.append(f"## Slide {i}\n" + "\n".join(p for p in paragraphs if p.strip()))
        return "\n\n".join(out)


CONVERT = {".xlsx": xlsx_text, ".docx": docx_text, ".pptx": pptx_text}


# ---- preparing files

def safe_unzip(path: Path, into: Path) -> list[Path]:
    """Extract, refusing members that would land outside `into` or that
    together exceed MAX_UNZIPPED."""
    into = into.resolve()
    out, total = [], 0
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            target = (into / member.filename).resolve()
            if target != into and into not in target.parents:
                raise ValueError(f"{path.name}: {member.filename!r} would be written outside the workspace")
            total += member.file_size
            if total > MAX_UNZIPPED:
                raise ValueError(f"{path.name} unpacks to more than {MAX_UNZIPPED // 2**20} MB")
            if member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


def _place(path: Path, shown: str, prepared: Prepared, seen: bool = True) -> None:
    """`path` is the file's copy in the workspace; `shown` its name relative to the workspace."""
    ext = path.suffix.lower()
    if seen and ext in IMAGE_TYPES and path.stat().st_size <= MAX_IMAGE:
        prepared.blocks.append({"type": "image", "source": {"type": "base64", "media_type": IMAGE_TYPES[ext],
                                                           "data": base64.b64encode(path.read_bytes()).decode()}})
        prepared.notes.append(f"{shown}: an image, shown to you above")
        return
    if seen and ext == ".pdf" and path.stat().st_size <= MAX_PDF:
        prepared.blocks.append({"type": "document", "title": path.name,
                                "source": {"type": "base64", "media_type": "application/pdf",
                                           "data": base64.b64encode(path.read_bytes()).decode()}})
        prepared.notes.append(f"{shown}: a PDF, shown to you above")
        return
    if ext in CONVERT:
        try:
            text_name = f"{shown}.txt"              # written next to the file
            path.with_name(path.name + ".txt").write_text(CONVERT[ext](path), encoding="utf-8")
            prepared.notes.append(f"{shown}: converted to text in the workspace as {text_name!r} "
                                  f"(read it with workspace__read_file)")
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
            prepared.notes.append(f"{shown}: couldn't be converted to text ({exc}); the original is in the workspace")
        return
    if ext in UNSUPPORTED:
        prepared.notes.append(f"{shown}: audio or video, which this assistant can't play or transcribe yet")
        return
    prepared.notes.append(f"{shown}: in the workspace (read it with workspace__read_file)")


def prepare(paths: list[Path], workspace: Path) -> Prepared:
    """Copy each file into the workspace and decide how the model gets it."""
    workspace.mkdir(parents=True, exist_ok=True)
    prepared = Prepared()
    for original in paths:
        original = Path(original)
        if not original.is_file():
            prepared.notes.append(f"{original.name}: missing")
            continue
        copy = workspace / original.name
        if copy.resolve() != original.resolve():
            shutil.copyfile(original, copy)
        if original.suffix.lower() == ".zip":
            folder = workspace / original.stem
            try:
                members = safe_unzip(copy, folder)
            except (ValueError, zipfile.BadZipFile) as exc:
                prepared.notes.append(f"{original.name}: not unpacked ({exc})")
                continue
            prepared.notes.append(f"{original.name}: a zip archive, unpacked into the workspace folder "
                                  f"{original.stem!r} ({len(members)} files)")
            for member in members:
                # files inside an archive go to the workspace (not inline), to keep the prompt small
                _place(member, f"{original.stem}/{member.relative_to(folder.resolve())}", prepared, seen=False)
            continue
        _place(copy, original.name, prepared)
    return prepared

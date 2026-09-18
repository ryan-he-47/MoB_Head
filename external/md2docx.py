# One-off: md -> docx for the advisor pack. Handles headings, paragraphs,
# bullets/numbered lists, pipe tables, blockquotes, **bold** / *italic*.
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


def add_runs(par, text):
    # split on **bold** and *italic*
    for tok in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*)", text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); r.italic = True
        else:
            par.add_run(tok)


def flush_table(doc, rows):
    rows = [r for r in rows if re.match(r"^\s*\|", r)]
    if not rows:
        return
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    # drop separator row (---)
    cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
    if not cells:
        return
    ncols = max(len(c) for c in cells)
    t = doc.add_table(rows=len(cells), cols=ncols)
    t.style = "Table Grid"
    for i, row in enumerate(cells):
        for j in range(ncols):
            txt = row[j] if j < len(row) else ""
            cell = t.cell(i, j)
            cell.paragraphs[0].text = ""
            add_runs(cell.paragraphs[0], txt)
            if i == 0:
                for r in cell.paragraphs[0].runs:
                    r.bold = True
    doc.add_paragraph()


def convert(src: Path, dst: Path):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    lines = src.read_text(encoding="utf-8").splitlines()
    i, table_buf = 0, []
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*\|", line):
            table_buf.append(line); i += 1; continue
        if table_buf:
            flush_table(doc, table_buf); table_buf = []
        s = line.strip()
        if not s or s == "---":
            i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            h = doc.add_heading(level=min(level, 4))
            add_runs(h, m.group(2))
            i += 1; continue
        if s.startswith("> "):
            p = doc.add_paragraph(style="Intense Quote")
            add_runs(p, s[2:]); i += 1; continue
        m = re.match(r"^[-*]\s+(.*)$", s)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, m.group(1)); i += 1; continue
        m = re.match(r"^\d+\.\s+(.*)$", s)
        if m:
            p = doc.add_paragraph(style="List Number")
            add_runs(p, m.group(1)); i += 1; continue
        p = doc.add_paragraph()
        add_runs(p, s)
        i += 1
    if table_buf:
        flush_table(doc, table_buf)
    # title-ish metadata lines at top are fine as normal paragraphs
    doc.save(str(dst))
    print(f"saved {dst}")


if __name__ == "__main__":
    pack = Path(r"E:\Trash_things\llm_codec\docs\advisor pack")
    convert(pack / "paper-survey-report.md", pack / "paper-survey-report.docx")
    convert(pack / "progress-report.md", pack / "progress-report.docx")

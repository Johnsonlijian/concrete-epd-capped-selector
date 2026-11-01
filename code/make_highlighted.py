"""Create the marked manuscript for a revision round.

Usage: python make_highlighted.py OLD_MAIN.tex NEW_MAIN.tex OUT.tex

OLD_MAIN.tex is the previously submitted manuscript and NEW_MAIN.tex the
revised one. Every sentence of NEW_MAIN.tex that does not occur in
OLD_MAIN.tex is coloured blue; unchanged text stays black. Display
equations and table or figure captions are compared in the same way, and an
\\input table whose rows changed is inlined with the changed cells in blue
(the table files are read from the directories of OLD_MAIN.tex and
NEW_MAIN.tex). The clean and marked files therefore contain the same text.
Sentences of OLD_MAIN.tex that no longer occur are listed on stdout so that
deletions can be reviewed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

OPEN = r"{\color{blue}"
OPEN_PAR = r"{\leavevmode\color{blue}"
CLOSE = "}"

STRUCTURAL = (
    "\\section", "\\subsection", "\\subsubsection", "\\label", "\\centering",
    "\\includegraphics", "\\footnotesize", "\\maketitle", "\\bibliography",
    "\\keywords", "\\backmatter", "\\title", "\\author", "\\affil", "\\email",
    "\\newcommand", "\\renewcommand", "\\newtheorem", "\\newcolumntype", "\\graphicspath",
    "\\sisetup", "\\setlength", "\\documentclass", "\\usepackage", "\\input",
    "\\begin", "\\end", "\\par", "\\toprule", "\\midrule", "\\bottomrule", "%", "{\\footnotesize", "}",
)
SENTENCE_BREAK = re.compile(r"(?<!\d)(?<=[.!?])\s+(?=[A-Z\\(])")
PREFIX = re.compile(r"^(\\abstract\{)?(\\textbf\{[^{}]*\}\s*)?")
FLOAT_ENVS = ("table", "table*", "figure", "figure*")
MATH_ENVS = ("equation", "equation*")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_spans(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Conservative sentence spans of a one-line TeX paragraph."""
    starts = [0] + [m.end() for m in SENTENCE_BREAK.finditer(text)]
    ends = starts[1:] + [len(text)]
    spans = []
    for start, end in zip(starts, ends):
        piece = text[start:end]
        stripped = piece.rstrip()
        if stripped.strip():
            lead = len(piece) - len(piece.lstrip())
            spans.append((offset + start + lead, offset + start + len(stripped)))
    return spans


def depth(text: str) -> int:
    """Brace depth change of a TeX fragment, ignoring escaped braces."""
    text = text.replace("\\{", "").replace("\\}", "")
    return text.count("{") - text.count("}")


def paragraph_body(line: str) -> tuple[int, int]:
    """Span of a paragraph line that may be coloured (labels and \\abstract{ excluded)."""
    m = PREFIX.match(line)
    start, end = (m.end() if m else 0), len(line.rstrip())
    while depth(line[start:end]) < 0 and line[start:end].endswith("}"):
        end -= 1
    return start, end


def caption_body(line: str) -> tuple[int, int]:
    start = line.find("\\caption{") + len("\\caption{")
    end = line.rfind("}")
    return start, end


def sentences_of(line: str, body: tuple[int, int]) -> list[tuple[int, int]]:
    start, end = body
    return sentence_spans(line[start:end], start)


def balanced(spans: list[tuple[int, int]], line: str) -> list[tuple[int, int]]:
    """Merge adjacent spans until each one is brace-balanced, so it can be wrapped in a group."""
    out: list[tuple[int, int]] = []
    for start, end in spans:
        if out and depth(line[out[-1][0]:out[-1][1]]) != 0:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    if out and depth(line[out[-1][0]:out[-1][1]]) != 0:
        raise ValueError(f"unbalanced braces in marked span: {line[out[-1][0]:out[-1][1]][:80]}")
    return out


def colour(line: str, spans: list[tuple[int, int]], par_start: int | None = None) -> str:
    """Wrap each span in a colour group; the group keeps the PDF colour stack balanced."""
    out, cursor = [], 0
    for start, end in balanced(spans, line):
        opener = OPEN_PAR if par_start is not None and start == par_start else OPEN
        out += [line[cursor:start], opener, line[start:end], CLOSE]
        cursor = end
    out.append(line[cursor:])
    return "".join(out)


def is_structural(line: str) -> bool:
    stripped = line.lstrip()
    return not stripped or stripped.startswith(STRUCTURAL)


def document_lines(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "\\begin{document}")
    return lines[start:]


def old_inventory(old_lines: list[str]) -> tuple[set[str], set[str], set[str]]:
    sentences: set[str] = set()
    blocks: set[str] = set()
    raw: set[str] = set(norm(ln) for ln in old_lines)
    env, buf = None, []
    for line in old_lines:
        stripped = line.lstrip()
        if env:
            buf.append(line)
            if f"\\end{{{env}}}" in line:
                blocks.add(norm("\n".join(buf)))
                env, buf = None, []
            continue
        m = re.match(r"\\begin\{(equation\*?)\}", stripped)
        if m:
            env, buf = m.group(1), [line]
            continue
        if stripped.startswith("\\caption{"):
            sentences.update(norm(line[s:e]) for s, e in sentences_of(line, caption_body(line)))
        elif "&" in line and line.rstrip().endswith("\\\\"):
            continue
        elif not is_structural(line):
            sentences.update(norm(line[s:e]) for s, e in sentences_of(line, paragraph_body(line)))
    return sentences, blocks, raw


def table_rows(text: str) -> list[str]:
    return [ln for ln in text.splitlines()]


def split_cells(row: str) -> tuple[list[str], str]:
    body, sep, tail = row.rpartition("\\\\")
    if not sep:
        return [row], ""
    return re.split(r"(?<!\\)&", body), sep + tail


def marked_table(old_text: str, new_text: str) -> tuple[list[str], int]:
    old_rows, new_rows = table_rows(old_text), table_rows(new_text)
    out, cells_marked = [], 0
    same_shape = len(old_rows) == len(new_rows)
    old_set = set(norm(r) for r in old_rows)
    for i, row in enumerate(new_rows):
        if norm(row) in old_set or "&" not in row:
            out.append(row)
            continue
        cells, tail = split_cells(row)
        old_cells = split_cells(old_rows[i])[0] if same_shape else []
        new_cells = []
        for j, cell in enumerate(cells):
            changed = not same_shape or j >= len(old_cells) or norm(cell) != norm(old_cells[j])
            if changed and cell.strip():
                lead = cell[: len(cell) - len(cell.lstrip())]
                trail = cell[len(cell.rstrip()):]
                cell = f"{lead}\\textcolor{{blue}}{{{cell.strip()}}}{trail}"
                cells_marked += 1
            new_cells.append(cell)
        out.append("&".join(new_cells) + tail)
    return out, cells_marked


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python make_highlighted.py OLD_MAIN.tex NEW_MAIN.tex OUT.tex")
    old_path, new_path, target = (Path(p) for p in sys.argv[1:4])
    old_text = old_path.read_text(encoding="utf-8")
    new_text = new_path.read_text(encoding="utf-8")
    old_sent, old_blocks, old_raw = old_inventory(document_lines(old_text))
    lines = new_text.splitlines()
    out: list[str] = []
    stats = {"sentences": 0, "captions": 0, "equations": 0, "notes": 0, "table_cells": 0}
    new_sent: set[str] = set()
    env = None
    i = 0
    in_body = False
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped == "\\begin{document}":
            in_body = True
            out.append(line)
            out.append("\\hypersetup{citecolor=black,linkcolor=black,urlcolor=black}")
            i += 1
            continue
        if not in_body:
            out.append(line)
            i += 1
            continue
        m = re.match(r"\\begin\{([a-z*]+)\}", stripped)
        if m and m.group(1) in MATH_ENVS:
            j = i
            while f"\\end{{{m.group(1)}}}" not in lines[j]:
                j += 1
            block = lines[i : j + 1]
            if norm("\n".join(block)) not in old_blocks:
                # the environment group pops the colour at \end{equation}
                block = [block[0], "\\color{blue}"] + block[1:]
                stats["equations"] += 1
            out.extend(block)
            i = j + 1
            continue
        if m and m.group(1) in FLOAT_ENVS:
            env = m.group(1)
        if env and f"\\end{{{env}}}" in line:
            env = None
        if stripped.startswith("\\caption{"):
            spans = sentences_of(line, caption_body(line))
            new_sent.update(norm(line[s:e]) for s, e in spans)
            changed = [(s, e) for s, e in spans if norm(line[s:e]) not in old_sent]
            if changed:
                stats["captions"] += 1
                stats["sentences"] += len(changed)
            out.append(colour(line, changed))
        elif "\\begin{minipage}" in line and norm(line) not in old_raw:
            start = line.find("\\footnotesize") + len("\\footnotesize")
            end = line.rfind("\\end{minipage}")
            out.append(colour(line, [(start + (len(line[start:end]) - len(line[start:end].lstrip())), end)]))
            stats["notes"] += 1
        elif stripped.startswith("\\input{tab_"):
            name = re.match(r"\\input\{([^}]+)\}", stripped).group(1)
            old_tab, new_tab = old_path.parent / name, new_path.parent / name
            if old_tab.exists() and new_tab.exists() and norm(old_tab.read_text(encoding="utf-8")) != norm(new_tab.read_text(encoding="utf-8")):
                rows, n = marked_table(old_tab.read_text(encoding="utf-8"), new_tab.read_text(encoding="utf-8"))
                out.append(f"% inlined from {name} with changed cells in blue")
                out.extend(rows)
                stats["table_cells"] += n
            else:
                out.append(line)
        elif env or is_structural(line):
            out.append(line)
        else:
            body = paragraph_body(line)
            spans = sentences_of(line, body)
            new_sent.update(norm(line[s:e]) for s, e in spans)
            changed = [(s, e) for s, e in spans if norm(line[s:e]) not in old_sent]
            stats["sentences"] += len(changed)
            out.append(colour(line, changed, par_start=body[0]))
        i += 1
    note = ("% Marked manuscript: text added or changed relative to the previous revision is blue; "
            "unchanged text is black. Hyperlinks are black in this file so that blue marks only revised text.\n")
    target.write_text(note + "\n".join(out) + "\n", encoding="utf-8")
    print(stats)
    removed = sorted(s for s in old_sent if s not in new_sent)
    print(f"sentences of the previous version that no longer occur: {len(removed)}")
    for s in removed:
        print("  -", s[:150])


if __name__ == "__main__":
    main()

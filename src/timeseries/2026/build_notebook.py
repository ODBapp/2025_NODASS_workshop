"""把 percent 格式的三個教學腳本依序組裝成 timeseries2026.ipynb（課堂 Colab 用）。

用法:  uv run python build_notebook.py

規則（對應 README「檔案格式約定」）：
- `# %% [markdown]` 格 → notebook markdown cell（去掉每行的 `# ` 前綴）。
- `# %%` 格 → notebook code cell，內容原樣保留。
- 一格對一格、依 01 → 02 → 03 順序串接，不合併任何 cell。

只用標準函式庫；改完 `.py` 後重跑本腳本即可重新產生 notebook。
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = ["01_basics_stl.py", "02_enso_events.py", "03_buoy_upwelling.py"]
OUTPUT = HERE / "timeseries2026.ipynb"

MARKER = re.compile(r"^#\s*%%(.*)$")   # `# %%` 或 `# %% [markdown]`（可帶標題）


def parse_percent_cells(text):
    """把 percent 格式檔案切成 [(cell_type, lines)]，cell_type ∈ {markdown, code}。"""
    cells = []
    current = None
    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            current = (kind, [])
            cells.append(current)
            continue
        if current is None:              # 第一個 marker 之前的內容（正常不會有）
            current = ("code", [])
            cells.append(current)
        current[1].append(line)
    return cells


def strip_comment_prefix(lines):
    """markdown 格的每行去掉 jupytext 的 `# ` 前綴；單獨的 `#` 是空行。"""
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.rstrip() == "#":
            out.append("")
        else:
            out.append(ln)
    return out


def trim_blank_edges(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def make_cell(kind, lines, cell_id):
    source = "\n".join(lines).splitlines(keepends=True)
    cell = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main():
    nb_cells = []
    for name in SOURCES:
        stem = name.split("_")[0]        # "01" / "02" / "03"
        n_md = n_code = 0
        text = (HERE / name).read_text(encoding="utf-8")
        for i, (kind, lines) in enumerate(parse_percent_cells(text)):
            if kind == "markdown":
                lines = strip_comment_prefix(lines)
            if not trim_blank_edges(lines):
                continue                 # 空格子直接略過
            nb_cells.append(make_cell(kind, lines, f"c{stem}-{i:03d}"))
            if kind == "markdown":
                n_md += 1
            else:
                n_code += 1
        print(f"{name}: {n_md} markdown + {n_code} code cells")

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print(f"-> {OUTPUT.name}: {len(nb_cells)} cells")


if __name__ == "__main__":
    main()

"""Convert DATASETS.md (source of truth) into site/datasets.json for the
static datasets page. Re-run and commit whenever DATASETS.md changes:

    python3 benchmarks/build_datasets_json.py
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "DATASETS.md"
OUT = REPO_ROOT / "site" / "datasets.json"


def parse_tables(text):
    """{task: [{"section": name, "columns": [...], "rows": [[...]]}]}"""
    tasks = {}
    current_task = None
    current_section = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## ") and "References" not in line:
            current_task = line[3:].strip()
            tasks[current_task] = []
            current_section = None
        elif line.startswith("### "):
            current_section = line[4:].strip()
        elif line.startswith("|") and current_task is not None:
            columns = [c.strip() for c in line.strip("|").split("|")]
            i += 1  # skip separator row
            rows = []
            while i + 1 < len(lines) and lines[i + 1].startswith("|"):
                i += 1
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
            tasks[current_task].append(
                {"section": current_section, "columns": columns, "rows": rows}
            )
        i += 1
    return tasks


def parse_references(text):
    """{key: bibtex_entry} from the References section's bibtex block."""
    refs = {}
    block = text.split("## References", 1)[1]
    for match in re.finditer(r"@\w+\{([^,]+),", block):
        key = match.group(1).strip()
        start = match.start()
        depth = 0
        for pos in range(start, len(block)):
            if block[pos] == "{":
                depth += 1
            elif block[pos] == "}":
                depth -= 1
                if depth == 0:
                    refs[key] = block[start : pos + 1].strip()
                    break
    return refs


def main():
    text = SRC.read_text()
    data = {"tasks": parse_tables(text), "references": parse_references(text)}
    n_rows = sum(len(t["rows"]) for task in data["tasks"].values() for t in task)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"Wrote {OUT}: {n_rows} datasets, {len(data['references'])} references")


if __name__ == "__main__":
    main()

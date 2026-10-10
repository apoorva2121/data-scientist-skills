"""Mechanical checks for every skill folder in this repo.

Usage:  python tools/validate_skills.py

Checks: frontmatter present; name matches folder and uses [a-z0-9-]; description starts with
"Use when" and fits the 1024-character limit; SKILL.md under 500 lines; every file path the skill
mentions under scripts/ or references/ exists; every ds-* skill it names exists in the repo
(reported as PENDING while the set is being written). Exit code 1 on any hard failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
skills = sorted(p.parent for p in ROOT.glob("*/SKILL.md"))
names = {p.name for p in skills}
failed = False

for d in skills:
    text = (d / "SKILL.md").read_text(encoding="utf-8")
    problems, notes = [], []
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        problems.append("no frontmatter")
    else:
        fm = m.group(1)
        name = re.search(r"^name: (.+)$", fm, re.M)
        desc = re.search(r"^description: (.+)$", fm, re.M)
        if not name or name.group(1).strip() != d.name:
            problems.append("name missing or differs from folder")
        elif not re.fullmatch(r"[a-z0-9-]+", name.group(1).strip()):
            problems.append("name has invalid characters")
        if not desc:
            problems.append("description missing")
        else:
            if not desc.group(1).startswith("Use when"):
                problems.append("description does not start with 'Use when'")
            if len(fm) > 1024:
                problems.append(f"frontmatter {len(fm)} chars > 1024")
    lines = text.count("\n")
    if lines > 500:
        problems.append(f"{lines} lines > 500")
    for ref in sorted(set(re.findall(r"`((?:scripts|references)/[\w./-]+)", text))):
        if not (d / ref).exists():
            problems.append(f"missing file {ref}")
    for other in sorted(set(re.findall(r"\bds-[a-z-]+[a-z]\b", text)) - {d.name}):
        if other not in names:
            notes.append(other)
    status = "FAIL" if problems else "ok"
    failed |= bool(problems)
    print(f"{status:4s} {d.name:24s} {lines:4d} lines  {'; '.join(problems)}")
    if notes:
        print(f"     pending skills referenced: {', '.join(notes)}")

sys.exit(1 if failed else 0)

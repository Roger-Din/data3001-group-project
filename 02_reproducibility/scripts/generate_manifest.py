#!/usr/bin/env python3
"""Create a stable SHA256 manifest for the final-submission package."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST_SHA256.txt"
EXCLUDED_PARTS = {".git", ".venv", ".ipynb_checkpoints", "__pycache__"}
EXCLUDED_FILES = {"MANIFEST_SHA256.txt", "latest_validation.json", ".DS_Store", ".Rhistory"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return not (set(relative.parts) & EXCLUDED_PARTS or path.name in EXCLUDED_FILES)


lines = []
for path in sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path)):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(lines)} checksums to {OUTPUT}")

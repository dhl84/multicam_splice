"""CLI: python -m multicam <config.json>

Probes the media, syncs the angles, builds the remix, writes + validates the
FCPXML named by the config's "output".
"""
from __future__ import annotations

import sys
from pathlib import Path

from .config import load
from .timeline import render


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m multicam <config.json>", file=sys.stderr)
        return 2
    proj = load(argv[0])
    print(f"project: {proj.project_name}")
    print(f"media:   {proj.media_dir}")
    out, ok, msg = render(proj)
    print(f"wrote {out}")
    print(f"validate: {'OK' if ok else 'FAIL'} — {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

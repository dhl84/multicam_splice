"""CLI: python -m multicam <config.json>

Probes the media, syncs the angles, builds the remix, writes + validates the
FCPXML named by the config's "output".
"""
from __future__ import annotations

import sys

from .config import load
from .timeline import render


def main(argv: list[str] | None = None) -> int:
    # argv defaults to the real command line so the installed console script
    # (`multicam-splice <config.json>`, which calls main() with no args) and
    # `python -m multicam <config.json>` share one entry point.
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        print("usage: multicam-splice <config.json>  "
              "(or: python -m multicam <config.json>)", file=sys.stderr)
        return 2
    # Expected user-input problems (bad/missing config, incompatible or missing
    # media, uncovered remix windows) are reported concisely; unexpected
    # exceptions are left to propagate so real bugs stay loud.
    try:
        proj = load(argv[0])
        print(f"project: {proj.project_name}")
        print(f"media:   {proj.media_dir}")
        out, ok, msg = render(proj)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    print(f"validate: {'OK' if ok else 'FAIL'} — {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

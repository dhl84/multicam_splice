"""multicam_splice — splice multiple camera angles of one event into a
flick-between-angles FCPXML remix for Final Cut Pro.

Self-contained (numpy + ffmpeg/ffprobe). See README.md.
"""
from .config import Project, load
from .timeline import Builder, render

__all__ = ["Project", "load", "Builder", "render"]

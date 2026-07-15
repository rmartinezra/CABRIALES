#!/usr/bin/env python3
"""Small terminal-progress helpers shared by CABRIALES entry points."""
from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Return a compact, stable duration for terminal status lines."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"

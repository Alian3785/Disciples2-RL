"""Helpers for consistent UTF-8 console output on Windows."""

from __future__ import annotations

import os
import sys


def setup_utf8_console() -> None:
    """
    Try to force UTF-8 for stdin/stdout/stderr and Windows console code pages.
    Safe to call multiple times.
    """
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if os.name != "nt":
        return

    try:
        os.system("chcp 65001 > nul")
    except Exception:
        pass

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except Exception:
        # Some runners do not expose a real Windows console.
        pass

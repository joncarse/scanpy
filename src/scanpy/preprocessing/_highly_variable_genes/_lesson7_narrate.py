"""Optional Lesson-7 learning narration (env-gated).

Enable with ``HVG_LESSON7_NARRATE=1`` (also ``true`` / ``yes``). No-op otherwise.
Safe to call from client and from distributed workers (they inherit the env).
"""

from __future__ import annotations

import os
from typing import Any


def narrate(where: str, title: str, **vars: Any) -> None:
    flag = os.environ.get("HVG_LESSON7_NARRATE", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return
    pid = os.getpid()
    detail = (" | " + " ".join(f"{k}={v!r}" for k, v in vars.items())) if vars else ""
    print(f"[L7 {where} pid={pid}] {title}{detail}", flush=True)

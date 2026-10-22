# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Low-volume runtime evidence for MXFP QAT experiments."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)
_seen: set[Any] = set()
_lock = threading.Lock()


def warn_once(key: Any, message: str, *args: Any) -> None:
    """Emit a warning once per process for a stable execution-path key."""
    with _lock:
        if key in _seen:
            return
        _seen.add(key)
    logger.warning("[MXFP QAT] " + message, *args)


def reset_warnings() -> None:
    """Reset warning state for isolated tests."""
    with _lock:
        _seen.clear()

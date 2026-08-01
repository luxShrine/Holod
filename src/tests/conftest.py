"""Shared pytest configuration for the holod test suite."""

from __future__ import annotations

import os

from holod.infra.log import set_console_level

# holod logs through a QueueListener, so the Rich console handler writes from a
# background thread. pytest briefly suspends its capture to print each test
# result, and any flush landing in that gap goes straight to the terminal
# instead of the capture buffer -- hence the interleaved, half-garbled log lines
# in the report. Silence the console sink for the session; the JSON file handler
# still records everything at DEBUG in logs/holo_log.jsonl.
# Set HOLOD_CONSOLE_LOG_LEVEL=DEBUG (or INFO) to get the chatter back.
set_console_level(os.environ.get("HOLOD_CONSOLE_LOG_LEVEL", "CRITICAL").upper())

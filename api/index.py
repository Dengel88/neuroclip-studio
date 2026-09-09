"""Serverless entrypoint.

Vercel serves whatever lives under `api/`. The application itself stays in
`main.py` at the project root, so the exact same code runs under uvicorn
locally and inside a function here - this file only fixes up the import path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import app  # noqa: E402  - import must follow the sys.path fix

__all__ = ["app"]

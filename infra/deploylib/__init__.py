"""Provisioning library for the chat-ingestion stack.
Importing this package puts infra/ on sys.path so the standalone helpers
(dbmigrate, dbrestore, admin_endpoint) import identically whether deploy.py is
run as a script or the package is imported from elsewhere.
"""
import sys
from pathlib import Path
_INFRA = Path(__file__).resolve().parent.parent
if str(_INFRA) not in sys.path:
    sys.path.insert(0, str(_INFRA))
from . import config            # noqa: E402
from .stack import Stack        # noqa: E402
__all__ = ["Stack", "config"]
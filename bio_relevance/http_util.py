"""HTTP helpers with optional TLS verify skip (common on locked-down Windows)."""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import Any, Optional


def _context() -> ssl.SSLContext:
    try:
        return ssl.create_default_context()
    except Exception:
        return ssl._create_unverified_context()


def http_get_json(url: str, *, timeout: int = 60, unverified: bool = False) -> Optional[Any]:
    ctx = ssl._create_unverified_context() if unverified else _context()
    req = urllib.request.Request(url, headers={"User-Agent": "bio_relevance/1.0", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ssl.SSLError):
        if not unverified:
            return http_get_json(url, timeout=timeout, unverified=True)
    return None


def http_post_json(url: str, payload: dict, *, timeout: int = 120, unverified: bool = False) -> Optional[Any]:
    ctx = ssl._create_unverified_context() if unverified else _context()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"User-Agent": "bio_relevance/1.0", "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ssl.SSLError):
        if not unverified:
            return http_post_json(url, payload, timeout=timeout, unverified=True)
    return None

"""Shared HTTP helpers for FlashRec benchmark clients."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional


def dumps_json(payload: Mapping[str, Any]) -> bytes:
    try:
        import orjson

        return orjson.dumps(payload)
    except ImportError:
        return json.dumps(payload).encode()


def loads_json(raw: bytes | str) -> Any:
    if isinstance(raw, str):
        raw = raw.encode()
    try:
        import orjson

        return orjson.loads(raw)
    except ImportError:
        return json.loads(raw)


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def post_json(
    url: str,
    body: Mapping[str, Any],
    timeout: float,
    retries: int = 1,
) -> dict[str, Any]:
    data = dumps_json(body)
    last_err: Exception | None = None
    for attempt in range(max(retries, 1)):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = loads_json(resp.read())
            if not isinstance(payload, dict):
                raise json.JSONDecodeError("expected object", "", 0)
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            err_s = str(err)
            if "111" in err_s or "Connection refused" in err_s:
                break
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"request failed after {retries} tries: {last_err}")


def get_text(url: str, timeout: float = 5.0) -> str:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def wait_for_endpoint(
    base_url: str,
    timeout_sec: int = 60,
    paths: tuple[str, ...] = ("/health", "/v1/models", "/get_model_info"),
) -> bool:
    base = base_url.rstrip("/")
    deadline = time.time() + max(timeout_sec, 1)
    while time.time() < deadline:
        for path in paths:
            try:
                get_text(f"{base}{path}", timeout=2.0)
                return True
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
        time.sleep(1.0)
    return False


def start_profile(
    base_url: str,
    *,
    timeout: float = 30.0,
    num_steps: Optional[int] = None,
) -> str:
    body: dict[str, Any] = {}
    if num_steps is not None:
        body["num_steps"] = int(num_steps)
    url = f"{base_url.rstrip('/')}/start_profile"
    try:
        post_json(url, body, timeout=timeout, retries=1)
        return "ok"
    except RuntimeError:
        # FlashRec also accepts GET without a body.
        return get_text(url, timeout=timeout)


def stop_profile(base_url: str, *, timeout: float = 600.0) -> str:
    url = f"{base_url.rstrip('/')}/stop_profile"
    try:
        post_json(url, {}, timeout=timeout, retries=1)
        return "ok"
    except RuntimeError:
        return get_text(url, timeout=timeout)

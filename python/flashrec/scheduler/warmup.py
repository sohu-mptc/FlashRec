"""Pin the shared chat prefix in radix with real KV.

Dummy KV is not inserted (would poison prefix hits). Two dummy user prompts
are tokenized; their longest common prefix is prefilled and locked. A single
short user message must not be locked, because ``add_generation_prompt`` would
pin assistant tokens after the head.

Default probes only share the chat template and system prompt. To also pin a
deployment-specific user-head, pass two texts that share that head via
``warmup_user_a`` / ``warmup_user_b``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# Distinct placeholders: LCP is template + system only.
_DEFAULT_USER_A = "warmup-probe-a"
_DEFAULT_USER_B = "warmup-probe-b"


def _lcp(a: Sequence[int], b: Sequence[int]) -> List[int]:
    n = min(len(a), len(b))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return [int(x) for x in a[:i]]


def strip_trailing_begin(ids: Sequence[int], begin_id: Optional[int]) -> List[int]:
    out = [int(x) for x in ids]
    if begin_id is not None and out and out[-1] == int(begin_id):
        return out[:-1]
    return out


def _probe_users(engine) -> tuple[str, str]:
    cfg = getattr(engine, "config", None)
    a = str(getattr(cfg, "warmup_user_a", None) or "").strip()
    b = str(getattr(cfg, "warmup_user_b", None) or "").strip()
    if a and b and a != b:
        return a, b
    if a or b:
        logger.warning(
            "radix warmup: set both warmup-user-a and warmup-user-b; "
            "using default probes"
        )
    return _DEFAULT_USER_A, _DEFAULT_USER_B


def warmup_shared_prefix(engine) -> int:
    prefix_cache = getattr(engine, "prefix_cache", None)
    system_prompt = getattr(engine, "system_prompt", None)
    if prefix_cache is None or not system_prompt:
        return 0
    device = getattr(engine, "device", None)
    if device is not None and getattr(device, "type", "") != "cuda":
        logger.info("radix warmup skipped (no CUDA)")
        return 0
    begin_id = None
    boundary = getattr(engine, "boundary_ids", None)
    if boundary:
        begin_id = int(boundary[0])
    user_a, user_b = _probe_users(engine)
    try:
        ids_a = engine.tokenize_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_a},
            ]
        )
        ids_b = engine.tokenize_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_b},
            ]
        )
    except Exception:
        logger.warning("radix warmup tokenize failed", exc_info=True)
        return 0
    prefix = strip_trailing_begin(_lcp(ids_a, ids_b), begin_id)
    if len(prefix) <= 1:
        logger.info("radix warmup: shared LCP too short (%d)", len(prefix))
        return 0
    try:
        n = int(engine.fill_prefix_kv(prefix))
    except Exception:
        logger.warning("radix warmup prefill failed", exc_info=True)
        return 0
    logger.info("radix warmup filled %d shared prefix tokens", n)
    return n

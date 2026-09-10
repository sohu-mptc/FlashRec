"""Capability-oriented kernel backend selection.

Each GPU operation (RMSNorm, RoPE, SiLU, etc.) has an ordered preference list
of backends. The registry probes availability once at import time, caches the
resolved backend per capability, and supports env-var overrides:

    FLASHREC_KERNEL_RMSNORM=torch          # force torch fallback
    FLASHREC_KERNEL_SILU_AND_MUL=sgl_kernel  # explicit sgl_kernel

Backends: "sgl_kernel", "flashinfer", "torch" (always available).
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class KernelCapability(Enum):
    RMSNORM = "RMSNORM"
    FUSED_ADD_RMSNORM = "FUSED_ADD_RMSNORM"
    ROPE_INPLACE = "ROPE_INPLACE"
    ROPE_AND_STORE_KV = "ROPE_AND_STORE_KV"
    SILU_AND_MUL = "SILU_AND_MUL"
    FP8_SCALED_MM = "FP8_SCALED_MM"
    PER_TOKEN_QUANT_FP8 = "PER_TOKEN_QUANT_FP8"
    STORE_KV_CACHE = "STORE_KV_CACHE"


_DEFAULT_PREFERENCES: Dict[KernelCapability, List[str]] = {
    KernelCapability.RMSNORM: ["sgl_kernel", "torch"],
    KernelCapability.FUSED_ADD_RMSNORM: ["sgl_kernel", "torch"],
    KernelCapability.ROPE_INPLACE: ["sgl_kernel"],
    KernelCapability.ROPE_AND_STORE_KV: ["sgl_kernel"],
    KernelCapability.SILU_AND_MUL: ["sgl_kernel", "torch"],
    KernelCapability.FP8_SCALED_MM: ["sgl_kernel", "torch"],
    KernelCapability.PER_TOKEN_QUANT_FP8: ["sgl_kernel", "torch"],
    KernelCapability.STORE_KV_CACHE: ["sgl_kernel", "torch"],
}


def _probe_sgl_kernel() -> bool:
    try:
        import sgl_kernel  # noqa: F401

        return True
    except Exception:
        return False


def _probe_flashinfer() -> bool:
    try:
        import flashinfer  # noqa: F401

        return True
    except Exception:
        return False


_BACKEND_PROBES = {
    "sgl_kernel": _probe_sgl_kernel,
    "flashinfer": _probe_flashinfer,
    "torch": lambda: True,
}


class KernelRegistry:
    _instance: Optional["KernelRegistry"] = None

    def __init__(self) -> None:
        self._available: Dict[str, bool] = {}
        for name, probe in _BACKEND_PROBES.items():
            try:
                self._available[name] = probe()
            except Exception:
                self._available[name] = False
        self._resolved: Dict[KernelCapability, Optional[str]] = {}
        for cap in KernelCapability:
            self._resolved[cap] = self._resolve(cap)
        avail = [k for k, v in self._available.items() if v]
        logger.info("KernelRegistry: available=%s", avail)
        overrides = {
            cap.value: backend
            for cap, backend in self._resolved.items()
            if self._env_override(cap) is not None
        }
        if overrides:
            logger.info("KernelRegistry: env overrides=%s", overrides)

    def _env_override(self, cap: KernelCapability) -> Optional[str]:
        val = os.environ.get(f"FLASHREC_KERNEL_{cap.value}")
        if val and val.strip():
            return val.strip()
        return None

    def _resolve(self, cap: KernelCapability) -> Optional[str]:
        override = self._env_override(cap)
        if override is not None:
            if self._available.get(override, False):
                return override
            logger.warning(
                "FLASHREC_KERNEL_%s=%s but backend unavailable, falling back",
                cap.value,
                override,
            )
        prefs = _DEFAULT_PREFERENCES.get(cap, [])
        for backend in prefs:
            if self._available.get(backend, False):
                return backend
        return None

    @classmethod
    def get(cls) -> "KernelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def resolve(self, cap: KernelCapability) -> Optional[str]:
        return self._resolved.get(cap)

    def using_sgl(self, cap: KernelCapability) -> bool:
        return self._resolved.get(cap) == "sgl_kernel"

    def is_available(self, backend: str) -> bool:
        return self._available.get(backend, False)

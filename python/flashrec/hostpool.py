"""CPU thread pool so tokenize / detokenize do not block the GPU worker."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, List, Optional


class HostPool:
    def __init__(self, threads: int):
        n = max(int(threads), 0)
        self._pool: Optional[ThreadPoolExecutor] = None
        if n > 0:
            self._pool = ThreadPoolExecutor(
                max_workers=n, thread_name_prefix="flashrec-host"
            )
        self._pending: List[Future] = []

    @property
    def enabled(self) -> bool:
        return self._pool is not None

    def submit(self, fn: Callable, *args, **kwargs) -> Optional[Future]:
        if self._pool is None:
            fn(*args, **kwargs)
            return None
        fut = self._pool.submit(fn, *args, **kwargs)
        self._pending.append(fut)
        return fut

    def flush(self) -> None:
        pending = self._pending
        self._pending = []
        for fut in pending:
            fut.result()

    def shutdown(self) -> None:
        self.flush()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

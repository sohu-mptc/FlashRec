"""Serving runtime for catalog-constrained wide-beam search."""

from typing import TYPE_CHECKING

from flashrec.version import __git_commit__, __version__

if TYPE_CHECKING:
    from flashrec.config import BeamRecConfig, FlashRecConfig
    from flashrec.scheduler.scheduler import BeamRecEngine, FlashRecEngine

__all__ = [
    "FlashRecConfig",
    "FlashRecEngine",
    "BeamRecConfig",
    "BeamRecEngine",
    "__git_commit__",
    "__version__",
]

# Resolved on first attribute access rather than at import time: pulling the
# engine in eagerly would drag torch and numpy into `import flashrec`,
# which breaks reading __version__ and running check_env on exactly the
# half-installed environments those two exist to diagnose.
_LAZY = {
    "FlashRecConfig": "flashrec.config",
    "FlashRecEngine": "flashrec.scheduler.scheduler",
    "BeamRecConfig": "flashrec.config",
    "BeamRecEngine": "flashrec.scheduler.scheduler",
}


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # subsequent lookups skip __getattr__
    return value


def __dir__() -> list:
    return sorted(__all__)

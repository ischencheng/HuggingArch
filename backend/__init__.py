"""
Backend — HuggingFace Model Architecture Analyzer

Uses lazy imports to avoid loading torch/transformers at import time.
"""

__version__ = "0.2.0"

import os as _os

if _os.environ.get("HA_CODE_TRACE") == "functions":
    # Recording starts before any engine module is imported, so what runs at
    # import is in the import-time trace (`code_trace.provider`, rule 2).
    from .code_trace import provider as _provider
    _provider.install()


def __getattr__(name):
    if name == "analyze_model":
        from .analyzer import analyze_model
        return analyze_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["analyze_model"]

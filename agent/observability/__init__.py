"""Unified observability module for tracing and instrumentation.

This module provides a consistent API for all tracing backends (LangSmith, MLflow, etc.)
and makes it easy to add new backends in the future.

Usage:
    from agent.observability import get_tracer, init_observability

    # Initialize all configured backends at startup
    results = init_observability()
    for backend, success in results.items():
        if success:
            print(f"{backend} tracing initialized")

    # Get trace URLs for a run
    tracer = get_tracer()
    urls = tracer.get_trace_urls(run_id)
    # urls = {"LangSmith": "https://...", "MLflow": "https://..."}
"""

from .config import ObservabilityConfig
from .tracer import Tracer, get_tracer, init_observability

__all__ = [
    "ObservabilityConfig",
    "Tracer",
    "get_tracer",
    "init_observability",
]

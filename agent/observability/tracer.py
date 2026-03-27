"""Unified tracer that delegates to configured backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .backends.langsmith import LangSmithBackend
from .config import ObservabilityConfig

if TYPE_CHECKING:
    from .backends.base import TracingBackend

logger = logging.getLogger(__name__)


class Tracer:
    """Unified tracer that delegates to configured backends.

    This class provides a single interface for all tracing operations,
    automatically discovering and initializing available backends.
    """

    _instance: Tracer | None = None

    def __init__(self, config: ObservabilityConfig | None = None) -> None:
        """Initialize the tracer.

        Args:
            config: The observability configuration. If None, loads from environment.
        """
        self._config = config or ObservabilityConfig.from_env()
        self._backends: list[TracingBackend] = []
        self._initialized = False
        self._load_backends()

    def _load_backends(self) -> None:
        """Load all available backends."""
        # Always load LangSmith (core dependency)
        self._backends.append(LangSmithBackend(self._config))

        # Try to load MLflow (optional)
        try:
            from .backends.mlflow import MLflowBackend

            self._backends.append(MLflowBackend(self._config))
        except ImportError:
            logger.debug("MLflow backend not available (package not installed)")

    def initialize(self) -> dict[str, bool]:
        """Initialize all enabled backends.

        Returns:
            A dictionary mapping backend names to their initialization status.
        """
        results: dict[str, bool] = {}
        for backend in self._backends:
            if backend.is_available() and backend.is_enabled():
                try:
                    success = backend.initialize()
                    results[backend.name] = success
                except Exception:
                    logger.exception(
                        "Failed to initialize backend: %s", backend.name
                    )
                    results[backend.name] = False
        self._initialized = True
        return results

    def get_trace_urls(self, run_id: str) -> dict[str, str]:
        """Get trace URLs from all initialized backends.

        Args:
            run_id: The run ID to get URLs for.

        Returns:
            A dictionary mapping backend names to their trace URLs.
        """
        urls: dict[str, str] = {}
        for backend in self._backends:
            if backend.is_enabled():
                try:
                    url = backend.get_trace_url(run_id)
                    if url:
                        urls[backend.name] = url
                except Exception:
                    logger.exception(
                        "Failed to get trace URL from backend: %s", backend.name
                    )
        return urls

    @property
    def backends(self) -> list[TracingBackend]:
        """Get the list of loaded backends."""
        return self._backends

    @property
    def enabled_backends(self) -> list[TracingBackend]:
        """Get the list of enabled backends."""
        return [b for b in self._backends if b.is_enabled()]

    @classmethod
    def get_instance(cls) -> Tracer:
        """Get or create the singleton tracer instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None


def get_tracer() -> Tracer:
    """Get the global tracer instance.

    Returns:
        The singleton Tracer instance.
    """
    return Tracer.get_instance()


def init_observability(config: ObservabilityConfig | None = None) -> dict[str, bool]:
    """Initialize all observability backends.

    This is a convenience function that creates a tracer and initializes it.

    Args:
        config: Optional configuration. If None, loads from environment.

    Returns:
        A dictionary mapping backend names to their initialization status.
    """
    tracer = get_tracer()
    if config is not None:
        # Replace the tracer with one using the provided config
        Tracer.reset_instance()
        Tracer._instance = Tracer(config)
        tracer = Tracer._instance
    return tracer.initialize()

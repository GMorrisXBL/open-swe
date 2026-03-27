"""Base protocol for tracing backends."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Generator


class TracingBackend(Protocol):
    """Protocol that all tracing backends must implement."""

    @property
    def name(self) -> str:
        """Human-readable backend name (e.g., 'LangSmith', 'MLflow')."""
        ...

    def is_available(self) -> bool:
        """Check if backend dependencies are installed."""
        ...

    def is_enabled(self) -> bool:
        """Check if backend is configured/enabled via environment."""
        ...

    def initialize(self) -> bool:
        """Initialize the backend. Returns True on success."""
        ...

    def get_trace_url(self, run_id: str) -> str | None:
        """Get the trace URL for a given run ID."""
        ...

    @contextmanager
    def create_span(
        self, name: str, **attributes: Any
    ) -> Generator[Any, None, None]:
        """Create a tracing span (optional, for manual instrumentation).

        Args:
            name: The name of the span.
            **attributes: Additional attributes to attach to the span.

        Yields:
            The span object, or None if tracing is not available.
        """
        ...

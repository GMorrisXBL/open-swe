"""LangSmith tracing backend (always available)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from agent.observability.config import ObservabilityConfig

logger = logging.getLogger(__name__)


class LangSmithBackend:
    """LangSmith tracing backend.

    LangSmith is a core dependency and is always available.
    Auto-initializes via langchain - no explicit setup needed.
    """

    name = "LangSmith"

    def __init__(self, config: ObservabilityConfig) -> None:
        """Initialize the LangSmith backend.

        Args:
            config: The observability configuration.
        """
        self._config = config

    def is_available(self) -> bool:
        """LangSmith is always available as a core dependency."""
        return True

    def is_enabled(self) -> bool:
        """Check if LangSmith is configured via environment variables."""
        return bool(
            self._config.langsmith_tenant_id and self._config.langsmith_project_id
        )

    def initialize(self) -> bool:
        """Initialize LangSmith tracing.

        LangSmith auto-initializes via langchain environment variables,
        so there's nothing to do here.
        """
        logger.debug("LangSmith backend initialized (auto-configured via langchain)")
        return True

    def get_trace_url(self, run_id: str) -> str | None:
        """Build the LangSmith trace URL for a given run ID.

        Args:
            run_id: The run ID to build the URL for.

        Returns:
            The full trace URL, or None if configuration is missing.
        """
        try:
            if not self._config.langsmith_tenant_id or not self._config.langsmith_project_id:
                logger.debug(
                    "LangSmith trace URL unavailable: missing tenant_id or project_id"
                )
                return None

            url_base = (
                f"{self._config.langsmith_url}"
                f"/o/{self._config.langsmith_tenant_id}"
                f"/projects/p/{self._config.langsmith_project_id}/r"
            )
            return f"{url_base}/{run_id}?poll=true"
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to build LangSmith trace URL for run %s",
                run_id,
                exc_info=True,
            )
            return None

    @contextmanager
    def create_span(
        self, name: str, **attributes: Any
    ) -> Generator[Any, None, None]:
        """Create a tracing span.

        For LangSmith, spans are auto-captured via langchain instrumentation.
        This method is a no-op placeholder for API consistency.

        Args:
            name: The name of the span.
            **attributes: Additional attributes (ignored).

        Yields:
            None (LangSmith uses automatic instrumentation).
        """
        # LangSmith uses automatic instrumentation via langchain
        # Manual span creation is not needed for the current use case
        yield None

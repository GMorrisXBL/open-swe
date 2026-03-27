"""MLflow/Databricks tracing backend (optional)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from agent.observability.config import ObservabilityConfig

logger = logging.getLogger(__name__)

# Check availability at module level
MLFLOW_AVAILABLE = False
try:
    import mlflow  # noqa: F401
    import mlflow.langchain  # noqa: F401

    MLFLOW_AVAILABLE = True
except ImportError:
    pass


class MLflowBackend:
    """MLflow/Databricks tracing backend.

    This backend is optional - MLflow must be installed separately.
    Supports both standard MLflow and Databricks-managed MLflow.
    """

    name = "MLflow"

    def __init__(self, config: ObservabilityConfig) -> None:
        """Initialize the MLflow backend.

        Args:
            config: The observability configuration.
        """
        self._config = config
        self._initialized = False

    def is_available(self) -> bool:
        """Check if MLflow is installed."""
        return MLFLOW_AVAILABLE

    def is_enabled(self) -> bool:
        """Check if MLflow is configured via environment variables."""
        return bool(
            self._config.mlflow_tracking_uri or self._config.databricks_host
        )

    def initialize(self) -> bool:
        """Initialize MLflow tracing with Databricks or standard MLflow backend.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if not self.is_available():
            logger.debug("MLflow not available (package not installed)")
            return False

        if not self.is_enabled():
            logger.debug(
                "MLflow not configured (no MLFLOW_TRACKING_URI or DATABRICKS_HOST)"
            )
            return False

        try:
            import mlflow
            import mlflow.langchain

            # Configure tracking URI
            if self._config.databricks_host:
                # Databricks-managed MLflow
                tracking_uri = "databricks"
                if self._config.databricks_token:
                    os.environ.setdefault(
                        "DATABRICKS_TOKEN", self._config.databricks_token
                    )
                if self._config.databricks_host:
                    os.environ.setdefault(
                        "DATABRICKS_HOST", self._config.databricks_host
                    )
                logger.info(
                    "Configuring MLflow with Databricks backend: %s",
                    self._config.databricks_host,
                )
            else:
                tracking_uri = self._config.mlflow_tracking_uri
                logger.info(
                    "Configuring MLflow with tracking URI: %s", tracking_uri
                )

            mlflow.set_tracking_uri(tracking_uri)

            # Set experiment
            if self._config.mlflow_experiment_name:
                mlflow.set_experiment(self._config.mlflow_experiment_name)
                logger.info(
                    "MLflow experiment set to: %s",
                    self._config.mlflow_experiment_name,
                )

            # Enable LangChain autolog for automatic trace capture
            if self._config.mlflow_enable_autolog:
                mlflow.langchain.autolog(
                    log_models=self._config.mlflow_log_models,
                    log_input_examples=self._config.mlflow_log_input_examples,
                    silent=True,
                )
                logger.info(
                    "MLflow LangChain autolog enabled (log_models=%s, log_input_examples=%s)",
                    self._config.mlflow_log_models,
                    self._config.mlflow_log_input_examples,
                )

            self._initialized = True
            return True

        except Exception:
            logger.exception("Failed to initialize MLflow tracing")
            return False

    def get_trace_url(self, run_id: str) -> str | None:
        """Build the MLflow run URL for a given run ID.

        Args:
            run_id: The run ID to build the URL for.

        Returns:
            The full URL to the MLflow run, or None if not configured.
        """
        if self._config.databricks_host:
            # Databricks MLflow URL format
            experiment_name = (
                self._config.mlflow_experiment_name or "open-swe-agent"
            )
            return (
                f"{self._config.databricks_host}"
                f"/#mlflow/experiments/{experiment_name}/runs/{run_id}"
            )

        if self._config.mlflow_tracking_uri:
            # Standard MLflow URL format
            return f"{self._config.mlflow_tracking_uri}/#/experiments/runs/{run_id}"

        return None

    @contextmanager
    def create_span(
        self, name: str, **attributes: Any
    ) -> Generator[Any, None, None]:
        """Create a tracing span using MLflow.

        Args:
            name: The name of the span.
            **attributes: Additional attributes to attach to the span.

        Yields:
            The MLflow run object, or None if MLflow is not available/configured.
        """
        if not self.is_available() or not self.is_enabled():
            yield None
            return

        if not self._initialized:
            if not self.initialize():
                yield None
                return

        try:
            import mlflow

            tags = {k: str(v) for k, v in attributes.items()} if attributes else None
            with mlflow.start_run(run_name=name, tags=tags) as run:
                logger.debug("Started MLflow run: %s", run.info.run_id)
                yield run
        except Exception:
            logger.exception("Error creating MLflow span")
            yield None

"""MLflow/Databricks observability integration for traces and LLM evaluation logs."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid startup overhead when MLflow is not configured
_mlflow = None
_mlflow_langchain = None


def _get_mlflow():
    """Lazy import of mlflow."""
    global _mlflow
    if _mlflow is None:
        import mlflow

        _mlflow = mlflow
    return _mlflow


def _get_mlflow_langchain():
    """Lazy import of mlflow.langchain."""
    global _mlflow_langchain
    if _mlflow_langchain is None:
        import mlflow.langchain

        _mlflow_langchain = mlflow.langchain
    return _mlflow_langchain


@dataclass
class MLflowConfig:
    """Configuration for MLflow/Databricks integration."""

    tracking_uri: str | None = None
    experiment_name: str | None = None
    databricks_host: str | None = None
    databricks_token: str | None = None
    enable_autolog: bool = True
    log_models: bool = False  # Disabled by default to reduce overhead
    log_input_examples: bool = True

    @classmethod
    def from_env(cls) -> MLflowConfig:
        """Create config from environment variables."""
        return cls(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            experiment_name=os.environ.get(
                "MLFLOW_EXPERIMENT_NAME", "open-swe-agent"
            ),
            databricks_host=os.environ.get("DATABRICKS_HOST"),
            databricks_token=os.environ.get("DATABRICKS_TOKEN"),
            enable_autolog=os.environ.get("MLFLOW_ENABLE_AUTOLOG", "true").lower()
            == "true",
            log_models=os.environ.get("MLFLOW_LOG_MODELS", "false").lower() == "true",
            log_input_examples=os.environ.get(
                "MLFLOW_LOG_INPUT_EXAMPLES", "true"
            ).lower()
            == "true",
        )


def is_mlflow_enabled() -> bool:
    """Check if MLflow tracing is enabled via environment configuration."""
    config = MLflowConfig.from_env()
    return bool(config.tracking_uri or config.databricks_host)


def initialize_mlflow_tracing(config: MLflowConfig | None = None) -> bool:
    """Initialize MLflow tracing with Databricks or standard MLflow backend.

    Args:
        config: MLflow configuration. If None, loads from environment.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    if config is None:
        config = MLflowConfig.from_env()

    if not config.tracking_uri and not config.databricks_host:
        logger.debug("MLflow tracing not configured (no MLFLOW_TRACKING_URI or DATABRICKS_HOST)")
        return False

    try:
        mlflow = _get_mlflow()

        # Configure tracking URI
        if config.databricks_host:
            # Databricks-managed MLflow
            tracking_uri = "databricks"
            if config.databricks_token:
                os.environ.setdefault("DATABRICKS_TOKEN", config.databricks_token)
            if config.databricks_host:
                os.environ.setdefault("DATABRICKS_HOST", config.databricks_host)
            logger.info("Configuring MLflow with Databricks backend: %s", config.databricks_host)
        else:
            tracking_uri = config.tracking_uri
            logger.info("Configuring MLflow with tracking URI: %s", tracking_uri)

        mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        if config.experiment_name:
            mlflow.set_experiment(config.experiment_name)
            logger.info("MLflow experiment set to: %s", config.experiment_name)

        # Enable LangChain autolog for automatic trace capture
        if config.enable_autolog:
            mlflow_langchain = _get_mlflow_langchain()
            mlflow_langchain.autolog(
                log_models=config.log_models,
                log_input_examples=config.log_input_examples,
                silent=True,
            )
            logger.info(
                "MLflow LangChain autolog enabled (log_models=%s, log_input_examples=%s)",
                config.log_models,
                config.log_input_examples,
            )

        return True

    except Exception:
        logger.exception("Failed to initialize MLflow tracing")
        return False


@dataclass
class LLMEvaluationLog:
    """Structured log for LLM evaluation."""

    run_id: str
    thread_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_id: str | None = None
    input_prompt: str | None = None
    output_completion: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_input: int | None = None
    tokens_output: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MLflowEvaluationLogger:
    """Logger for capturing LLM interactions for evaluation."""

    def __init__(self, run_id: str | None = None):
        """Initialize the evaluation logger.

        Args:
            run_id: Optional MLflow run ID. If None, uses active run or creates new one.
        """
        self._run_id = run_id
        self._logs: list[LLMEvaluationLog] = []
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Ensure MLflow is initialized."""
        if self._initialized:
            return True

        if not is_mlflow_enabled():
            return False

        self._initialized = initialize_mlflow_tracing()
        return self._initialized

    def log_llm_interaction(
        self,
        run_id: str,
        *,
        thread_id: str | None = None,
        model_id: str | None = None,
        input_prompt: str | None = None,
        output_completion: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an LLM interaction for later evaluation.

        Args:
            run_id: Unique identifier for this agent run.
            thread_id: Thread/conversation ID.
            model_id: Model identifier (e.g., "anthropic:claude-opus-4-6").
            input_prompt: The input prompt sent to the model.
            output_completion: The model's output completion.
            tool_calls: List of tool calls made by the model.
            tokens_input: Number of input tokens.
            tokens_output: Number of output tokens.
            latency_ms: Response latency in milliseconds.
            error: Error message if the call failed.
            metadata: Additional metadata to log.
        """
        log_entry = LLMEvaluationLog(
            run_id=run_id,
            thread_id=thread_id,
            model_id=model_id,
            input_prompt=input_prompt,
            output_completion=output_completion,
            tool_calls=tool_calls or [],
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            error=error,
            metadata=metadata or {},
        )
        self._logs.append(log_entry)

        # Attempt immediate flush to MLflow if enabled
        if self._ensure_initialized():
            self._flush_log(log_entry)

    def _flush_log(self, log: LLMEvaluationLog) -> None:
        """Flush a single log entry to MLflow."""
        try:
            mlflow = _get_mlflow()

            # Log as a table row for evaluation
            mlflow.log_table(
                data={
                    "run_id": [log.run_id],
                    "thread_id": [log.thread_id],
                    "timestamp": [log.timestamp],
                    "model_id": [log.model_id],
                    "input_prompt": [log.input_prompt],
                    "output_completion": [log.output_completion],
                    "tool_calls": [str(log.tool_calls)],
                    "tokens_input": [log.tokens_input],
                    "tokens_output": [log.tokens_output],
                    "latency_ms": [log.latency_ms],
                    "error": [log.error],
                },
                artifact_file="llm_evaluation_logs.json",
            )

            # Also log key metrics
            if log.tokens_input is not None:
                mlflow.log_metric("tokens_input", log.tokens_input)
            if log.tokens_output is not None:
                mlflow.log_metric("tokens_output", log.tokens_output)
            if log.latency_ms is not None:
                mlflow.log_metric("latency_ms", log.latency_ms)

        except Exception:
            logger.exception("Failed to flush log to MLflow: %s", log.run_id)

    def flush_all(self) -> int:
        """Flush all pending logs to MLflow.

        Returns:
            Number of logs flushed.
        """
        if not self._ensure_initialized():
            logger.warning("MLflow not initialized, cannot flush logs")
            return 0

        count = 0
        for log in self._logs:
            try:
                self._flush_log(log)
                count += 1
            except Exception:
                logger.exception("Failed to flush log: %s", log.run_id)

        self._logs.clear()
        return count


# Global evaluation logger instance
_evaluation_logger: MLflowEvaluationLogger | None = None


def get_evaluation_logger() -> MLflowEvaluationLogger:
    """Get or create the global evaluation logger instance."""
    global _evaluation_logger
    if _evaluation_logger is None:
        _evaluation_logger = MLflowEvaluationLogger()
    return _evaluation_logger


@contextmanager
def mlflow_trace_context(
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[Any, None, None]:
    """Context manager for MLflow tracing.

    Creates an MLflow run context for tracing agent execution.

    Args:
        run_name: Optional name for the MLflow run.
        tags: Optional tags to add to the run.

    Yields:
        The MLflow run object, or None if MLflow is not configured.

    Example:
        with mlflow_trace_context(run_name="agent-execution", tags={"thread_id": "123"}):
            # Agent execution code here
            pass
    """
    if not is_mlflow_enabled():
        logger.debug("MLflow not enabled, yielding None context")
        yield None
        return

    if not initialize_mlflow_tracing():
        logger.warning("Failed to initialize MLflow, yielding None context")
        yield None
        return

    try:
        mlflow = _get_mlflow()
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            logger.info("Started MLflow run: %s", run.info.run_id)
            yield run
    except Exception:
        logger.exception("Error in MLflow trace context")
        yield None


def get_mlflow_run_url(run_id: str) -> str | None:
    """Build the MLflow run URL for a given run ID.

    Args:
        run_id: The MLflow run ID.

    Returns:
        The full URL to the MLflow run, or None if not configured.
    """
    config = MLflowConfig.from_env()

    if config.databricks_host:
        # Databricks MLflow URL format
        experiment_name = config.experiment_name or "open-swe-agent"
        return f"{config.databricks_host}/#mlflow/experiments/{experiment_name}/runs/{run_id}"

    if config.tracking_uri:
        # Standard MLflow URL format
        return f"{config.tracking_uri}/#/experiments/runs/{run_id}"

    return None

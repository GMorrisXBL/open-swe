"""Unified observability configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ObservabilityConfig:
    """Unified observability configuration loaded from environment."""

    # LangSmith (always available)
    langsmith_api_key: str | None = None
    langsmith_url: str = "https://smith.langchain.com"
    langsmith_tenant_id: str | None = None
    langsmith_project_id: str | None = None

    # MLflow (optional)
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "open-swe-agent"
    databricks_host: str | None = None
    databricks_token: str | None = None
    mlflow_enable_autolog: bool = True
    mlflow_log_models: bool = False
    mlflow_log_input_examples: bool = True

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        """Load configuration from environment variables."""
        return cls(
            # LangSmith configuration
            langsmith_api_key=os.environ.get("LANGSMITH_API_KEY_PROD"),
            langsmith_url=os.environ.get(
                "LANGSMITH_URL_PROD", "https://smith.langchain.com"
            ),
            langsmith_tenant_id=os.environ.get("LANGSMITH_TENANT_ID_PROD"),
            langsmith_project_id=os.environ.get("LANGSMITH_TRACING_PROJECT_ID_PROD"),
            # MLflow configuration
            mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            mlflow_experiment_name=os.environ.get(
                "MLFLOW_EXPERIMENT_NAME", "open-swe-agent"
            ),
            databricks_host=os.environ.get("DATABRICKS_HOST"),
            databricks_token=os.environ.get("DATABRICKS_TOKEN"),
            mlflow_enable_autolog=os.environ.get(
                "MLFLOW_ENABLE_AUTOLOG", "true"
            ).lower()
            == "true",
            mlflow_log_models=os.environ.get("MLFLOW_LOG_MODELS", "false").lower()
            == "true",
            mlflow_log_input_examples=os.environ.get(
                "MLFLOW_LOG_INPUT_EXAMPLES", "true"
            ).lower()
            == "true",
        )

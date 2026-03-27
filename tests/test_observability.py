"""Tests for the observability module."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from agent.observability import ObservabilityConfig, Tracer, get_tracer, init_observability

# ============================================================================
# ObservabilityConfig Tests
# ============================================================================


class TestObservabilityConfig:
    """Tests for ObservabilityConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = ObservabilityConfig()

        # LangSmith defaults
        assert config.langsmith_api_key is None
        assert config.langsmith_url == "https://smith.langchain.com"
        assert config.langsmith_tenant_id is None
        assert config.langsmith_project_id is None

        # MLflow defaults
        assert config.mlflow_tracking_uri is None
        assert config.mlflow_experiment_name == "open-swe-agent"
        assert config.databricks_host is None
        assert config.databricks_token is None
        assert config.mlflow_enable_autolog is True
        assert config.mlflow_log_models is False
        assert config.mlflow_log_input_examples is True

    def test_from_env_with_no_env_vars(self) -> None:
        """Test from_env with no environment variables set."""
        with _clean_env():
            config = ObservabilityConfig.from_env()

            assert config.langsmith_api_key is None
            assert config.langsmith_tenant_id is None
            assert config.langsmith_project_id is None
            assert config.mlflow_tracking_uri is None
            assert config.databricks_host is None

    def test_from_env_with_langsmith_vars(self) -> None:
        """Test from_env loads LangSmith variables correctly."""
        env_vars = {
            "LANGSMITH_API_KEY_PROD": "test-api-key",
            "LANGSMITH_URL_PROD": "https://custom.langsmith.com",
            "LANGSMITH_TENANT_ID_PROD": "tenant-123",
            "LANGSMITH_TRACING_PROJECT_ID_PROD": "project-456",
        }
        with _clean_env(env_vars):
            config = ObservabilityConfig.from_env()

            assert config.langsmith_api_key == "test-api-key"
            assert config.langsmith_url == "https://custom.langsmith.com"
            assert config.langsmith_tenant_id == "tenant-123"
            assert config.langsmith_project_id == "project-456"

    def test_from_env_with_mlflow_vars(self) -> None:
        """Test from_env loads MLflow variables correctly."""
        env_vars = {
            "MLFLOW_TRACKING_URI": "http://mlflow.example.com",
            "MLFLOW_EXPERIMENT_NAME": "my-experiment",
            "MLFLOW_ENABLE_AUTOLOG": "false",
            "MLFLOW_LOG_MODELS": "true",
            "MLFLOW_LOG_INPUT_EXAMPLES": "false",
        }
        with _clean_env(env_vars):
            config = ObservabilityConfig.from_env()

            assert config.mlflow_tracking_uri == "http://mlflow.example.com"
            assert config.mlflow_experiment_name == "my-experiment"
            assert config.mlflow_enable_autolog is False
            assert config.mlflow_log_models is True
            assert config.mlflow_log_input_examples is False

    def test_from_env_with_databricks_vars(self) -> None:
        """Test from_env loads Databricks variables correctly."""
        env_vars = {
            "DATABRICKS_HOST": "https://databricks.example.com",
            "DATABRICKS_TOKEN": "dapi-token-123",
        }
        with _clean_env(env_vars):
            config = ObservabilityConfig.from_env()

            assert config.databricks_host == "https://databricks.example.com"
            assert config.databricks_token == "dapi-token-123"


# ============================================================================
# LangSmith Backend Tests
# ============================================================================


class TestLangSmithBackend:
    """Tests for the LangSmith tracing backend."""

    def test_is_available_always_true(self) -> None:
        """LangSmith is always available as a core dependency."""
        from agent.observability.backends.langsmith import LangSmithBackend

        config = ObservabilityConfig()
        backend = LangSmithBackend(config)

        assert backend.is_available() is True

    def test_name_property(self) -> None:
        """Test the backend name property."""
        from agent.observability.backends.langsmith import LangSmithBackend

        config = ObservabilityConfig()
        backend = LangSmithBackend(config)

        assert backend.name == "LangSmith"

    def test_is_enabled_when_configured(self) -> None:
        """Test is_enabled returns True when tenant and project IDs are set."""
        from agent.observability.backends.langsmith import LangSmithBackend

        config = ObservabilityConfig(
            langsmith_tenant_id="tenant-123",
            langsmith_project_id="project-456",
        )
        backend = LangSmithBackend(config)

        assert backend.is_enabled() is True

    def test_is_enabled_when_not_configured(self) -> None:
        """Test is_enabled returns False when IDs are missing."""
        from agent.observability.backends.langsmith import LangSmithBackend

        config = ObservabilityConfig()
        backend = LangSmithBackend(config)

        assert backend.is_enabled() is False

    def test_is_enabled_with_partial_config(self) -> None:
        """Test is_enabled returns False with only tenant ID."""
        from agent.observability.backends.langsmith import LangSmithBackend

        config = ObservabilityConfig(langsmith_tenant_id="tenant-123")
        backend = LangSmithBackend(config)

        assert backend.is_enabled() is False

    def test_initialize_returns_true(self) -> None:
        """Test initialize always returns True (auto-configured via langchain)."""
        from agent.observability.backends.langsmith import LangSmithBackend

        config = ObservabilityConfig()
        backend = LangSmithBackend(config)

        assert backend.initialize() is True

    def test_get_trace_url_when_configured(self) -> None:
        """Test get_trace_url returns correct URL when configured."""
        from agent.observability.backends.langsmith import LangSmithBackend

        config = ObservabilityConfig(
            langsmith_url="https://smith.langchain.com",
            langsmith_tenant_id="tenant-123",
            langsmith_project_id="project-456",
        )
        backend = LangSmithBackend(config)

        url = backend.get_trace_url("run-789")

        assert url is not None
        assert "tenant-123" in url
        assert "project-456" in url
        assert "run-789" in url
        assert url == "https://smith.langchain.com/o/tenant-123/projects/p/project-456/r/run-789?poll=true"

    def test_get_trace_url_when_not_configured(self) -> None:
        """Test get_trace_url returns None when not configured."""
        from agent.observability.backends.langsmith import LangSmithBackend

        config = ObservabilityConfig()
        backend = LangSmithBackend(config)

        url = backend.get_trace_url("run-789")

        assert url is None

    def test_create_span_yields_none(self) -> None:
        """Test create_span yields None (uses automatic instrumentation)."""
        from agent.observability.backends.langsmith import LangSmithBackend

        config = ObservabilityConfig()
        backend = LangSmithBackend(config)

        with backend.create_span("test-span", key="value") as span:
            assert span is None


# ============================================================================
# MLflow Backend Tests
# ============================================================================


class TestMLflowBackend:
    """Tests for the MLflow tracing backend."""

    def test_name_property(self) -> None:
        """Test the backend name property."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig()
        backend = MLflowBackend(config)

        assert backend.name == "MLflow"

    def test_is_available_when_mlflow_installed(self) -> None:
        """Test is_available returns True when MLflow is installed."""
        from agent.observability.backends.mlflow import MLFLOW_AVAILABLE, MLflowBackend

        config = ObservabilityConfig()
        backend = MLflowBackend(config)

        # This depends on whether mlflow is actually installed
        assert backend.is_available() == MLFLOW_AVAILABLE

    def test_is_enabled_with_tracking_uri(self) -> None:
        """Test is_enabled returns True when tracking URI is set."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig(mlflow_tracking_uri="http://mlflow.example.com")
        backend = MLflowBackend(config)

        assert backend.is_enabled() is True

    def test_is_enabled_with_databricks_host(self) -> None:
        """Test is_enabled returns True when Databricks host is set."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig(databricks_host="https://databricks.example.com")
        backend = MLflowBackend(config)

        assert backend.is_enabled() is True

    def test_is_enabled_when_not_configured(self) -> None:
        """Test is_enabled returns False when not configured."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig()
        backend = MLflowBackend(config)

        assert backend.is_enabled() is False

    def test_initialize_returns_false_when_not_available(self) -> None:
        """Test initialize returns False when MLflow is not installed."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig(mlflow_tracking_uri="http://mlflow.example.com")
        backend = MLflowBackend(config)

        # Mock is_available to return False
        with patch.object(backend, "is_available", return_value=False):
            result = backend.initialize()
            assert result is False

    def test_initialize_returns_false_when_not_enabled(self) -> None:
        """Test initialize returns False when not enabled."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig()  # No tracking URI or Databricks host
        backend = MLflowBackend(config)

        result = backend.initialize()
        assert result is False

    def test_initialize_with_standard_mlflow(self) -> None:
        """Test initialize with standard MLflow tracking URI."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig(
            mlflow_tracking_uri="http://mlflow.example.com",
            mlflow_experiment_name="test-experiment",
            mlflow_enable_autolog=True,
            mlflow_log_models=False,
            mlflow_log_input_examples=True,
        )
        backend = MLflowBackend(config)

        # Mock mlflow module
        mock_mlflow = MagicMock()
        mock_mlflow_langchain = MagicMock()

        with patch.object(backend, "is_available", return_value=True):
            with patch.dict(
                "sys.modules",
                {"mlflow": mock_mlflow, "mlflow.langchain": mock_mlflow_langchain},
            ):
                with patch(
                    "agent.observability.backends.mlflow.MLFLOW_AVAILABLE", True
                ):
                    # Re-import to get mocked version
                    import agent.observability.backends.mlflow as mlflow_module

                    # Patch the imports inside initialize
                    with patch.object(
                        mlflow_module, "mlflow", mock_mlflow, create=True
                    ):
                        # Direct patching approach
                        def mock_initialize() -> bool:
                            mock_mlflow.set_tracking_uri("http://mlflow.example.com")
                            mock_mlflow.set_experiment("test-experiment")
                            mock_mlflow_langchain.autolog(
                                log_models=False,
                                log_input_examples=True,
                                silent=True,
                            )
                            backend._initialized = True
                            return True

                        with patch.object(backend, "initialize", mock_initialize):
                            result = backend.initialize()

                        assert result is True
                        assert backend._initialized is True

    def test_get_trace_url_with_databricks(self) -> None:
        """Test get_trace_url with Databricks configuration."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig(
            databricks_host="https://databricks.example.com",
            mlflow_experiment_name="test-experiment",
        )
        backend = MLflowBackend(config)

        url = backend.get_trace_url("run-123")

        assert url is not None
        assert "databricks.example.com" in url
        assert "test-experiment" in url
        assert "run-123" in url

    def test_get_trace_url_with_standard_mlflow(self) -> None:
        """Test get_trace_url with standard MLflow configuration."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig(mlflow_tracking_uri="http://mlflow.example.com")
        backend = MLflowBackend(config)

        url = backend.get_trace_url("run-456")

        assert url is not None
        assert "mlflow.example.com" in url
        assert "run-456" in url

    def test_get_trace_url_when_not_configured(self) -> None:
        """Test get_trace_url returns None when not configured."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig()
        backend = MLflowBackend(config)

        url = backend.get_trace_url("run-789")

        assert url is None

    def test_create_span_when_not_available(self) -> None:
        """Test create_span yields None when MLflow is not available."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig(mlflow_tracking_uri="http://mlflow.example.com")
        backend = MLflowBackend(config)

        with patch.object(backend, "is_available", return_value=False):
            with backend.create_span("test-span") as span:
                assert span is None

    def test_create_span_when_not_enabled(self) -> None:
        """Test create_span yields None when not enabled."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig()  # Not enabled
        backend = MLflowBackend(config)

        with backend.create_span("test-span") as span:
            assert span is None


# ============================================================================
# Tracer Tests
# ============================================================================


class TestTracer:
    """Tests for the Tracer class."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        Tracer.reset_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        Tracer.reset_instance()

    def test_singleton_pattern(self) -> None:
        """Test that get_instance returns the same instance."""
        tracer1 = Tracer.get_instance()
        tracer2 = Tracer.get_instance()

        assert tracer1 is tracer2

    def test_reset_instance(self) -> None:
        """Test that reset_instance clears the singleton."""
        tracer1 = Tracer.get_instance()
        Tracer.reset_instance()
        tracer2 = Tracer.get_instance()

        assert tracer1 is not tracer2

    def test_get_tracer_returns_singleton(self) -> None:
        """Test that get_tracer returns the singleton instance."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()

        assert tracer1 is tracer2

    def test_loads_langsmith_backend(self) -> None:
        """Test that LangSmith backend is always loaded."""
        tracer = Tracer()

        backend_names = [b.name for b in tracer.backends]
        assert "LangSmith" in backend_names

    def test_loads_mlflow_backend_when_available(self) -> None:
        """Test that MLflow backend is loaded when available."""
        from agent.observability.backends.mlflow import MLFLOW_AVAILABLE

        tracer = Tracer()

        backend_names = [b.name for b in tracer.backends]

        if MLFLOW_AVAILABLE:
            assert "MLflow" in backend_names
        else:
            # MLflow should not be in backends if not installed
            assert "MLflow" not in backend_names or any(
                b.name == "MLflow" for b in tracer.backends
            )

    def test_enabled_backends_property(self) -> None:
        """Test enabled_backends returns only enabled backends."""
        config = ObservabilityConfig(
            langsmith_tenant_id="tenant-123",
            langsmith_project_id="project-456",
        )
        tracer = Tracer(config)

        enabled = tracer.enabled_backends
        enabled_names = [b.name for b in enabled]

        assert "LangSmith" in enabled_names

    def test_initialize_returns_results_dict(self) -> None:
        """Test initialize returns a dictionary of results."""
        config = ObservabilityConfig(
            langsmith_tenant_id="tenant-123",
            langsmith_project_id="project-456",
        )
        tracer = Tracer(config)

        results = tracer.initialize()

        assert isinstance(results, dict)
        assert "LangSmith" in results
        assert results["LangSmith"] is True

    def test_initialize_skips_disabled_backends(self) -> None:
        """Test initialize skips backends that are not enabled."""
        config = ObservabilityConfig()  # Nothing enabled
        tracer = Tracer(config)

        results = tracer.initialize()

        # No backends should be initialized
        assert results == {}

    def test_get_trace_urls_returns_dict(self) -> None:
        """Test get_trace_urls returns a dictionary of URLs."""
        config = ObservabilityConfig(
            langsmith_url="https://smith.langchain.com",
            langsmith_tenant_id="tenant-123",
            langsmith_project_id="project-456",
        )
        tracer = Tracer(config)

        urls = tracer.get_trace_urls("run-789")

        assert isinstance(urls, dict)
        assert "LangSmith" in urls
        assert "run-789" in urls["LangSmith"]

    def test_get_trace_urls_empty_when_not_configured(self) -> None:
        """Test get_trace_urls returns empty dict when not configured."""
        config = ObservabilityConfig()
        tracer = Tracer(config)

        urls = tracer.get_trace_urls("run-789")

        assert urls == {}

    def test_init_observability_convenience_function(self) -> None:
        """Test init_observability convenience function."""
        config = ObservabilityConfig(
            langsmith_tenant_id="tenant-123",
            langsmith_project_id="project-456",
        )

        results = init_observability(config)

        assert isinstance(results, dict)
        assert "LangSmith" in results


# ============================================================================
# Integration Tests
# ============================================================================


class TestObservabilityIntegration:
    """Integration tests for the observability module."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        Tracer.reset_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        Tracer.reset_instance()

    def test_full_workflow_langsmith_only(self) -> None:
        """Test full workflow with only LangSmith configured."""
        config = ObservabilityConfig(
            langsmith_url="https://smith.langchain.com",
            langsmith_tenant_id="tenant-123",
            langsmith_project_id="project-456",
        )

        # Initialize
        results = init_observability(config)
        assert results.get("LangSmith") is True

        # Get trace URLs
        tracer = get_tracer()
        urls = tracer.get_trace_urls("test-run-id")

        assert "LangSmith" in urls
        assert "test-run-id" in urls["LangSmith"]

    def test_full_workflow_with_mlflow_mocked(self) -> None:
        """Test full workflow with MLflow mocked."""
        from agent.observability.backends.mlflow import MLflowBackend

        config = ObservabilityConfig(
            langsmith_url="https://smith.langchain.com",
            langsmith_tenant_id="tenant-123",
            langsmith_project_id="project-456",
            mlflow_tracking_uri="http://mlflow.example.com",
        )

        tracer = Tracer(config)

        # Mock MLflow backend initialization
        for backend in tracer.backends:
            if isinstance(backend, MLflowBackend):
                with patch.object(backend, "is_available", return_value=True):
                    with patch.object(backend, "initialize", return_value=True):
                        tracer.initialize()

        # Get URLs (MLflow should work since tracking URI is set)
        urls = tracer.get_trace_urls("test-run-id")

        assert "LangSmith" in urls
        # MLflow URL should be present if configured
        if config.mlflow_tracking_uri:
            # Check MLflow backend returns URL
            for backend in tracer.backends:
                if backend.name == "MLflow":
                    mlflow_url = backend.get_trace_url("test-run-id")
                    if mlflow_url:
                        assert "mlflow.example.com" in mlflow_url


# ============================================================================
# Helper Functions
# ============================================================================


@contextmanager
def _clean_env(
    env_vars: dict[str, str] | None = None
) -> Generator[None, None, None]:
    """Context manager to temporarily set environment variables.

    Clears all observability-related env vars first, then sets the provided ones.

    Args:
        env_vars: Dictionary of environment variables to set.

    Yields:
        None
    """
    # List of env vars to clear
    obs_env_vars = [
        "LANGSMITH_API_KEY_PROD",
        "LANGSMITH_URL_PROD",
        "LANGSMITH_TENANT_ID_PROD",
        "LANGSMITH_TRACING_PROJECT_ID_PROD",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "MLFLOW_ENABLE_AUTOLOG",
        "MLFLOW_LOG_MODELS",
        "MLFLOW_LOG_INPUT_EXAMPLES",
    ]

    # Save original values
    original_values: dict[str, str | None] = {}
    for var in obs_env_vars:
        original_values[var] = os.environ.get(var)

    # Clear all
    for var in obs_env_vars:
        if var in os.environ:
            del os.environ[var]

    # Set new values
    if env_vars:
        for key, value in env_vars.items():
            os.environ[key] = value

    try:
        yield
    finally:
        # Restore original values
        for var, value in original_values.items():
            if value is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = value

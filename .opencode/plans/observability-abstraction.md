# Observability Abstraction Plan

## Goal

Create a unified observability module that abstracts instrumentation configuration and initialization, making it easy to:
1. Add new tracing backends (MLflow, Datadog, etc.)
2. Keep MLflow as an optional dependency for upstream PR
3. Provide a consistent `Tracer` API for trace operations

---

## Current State

### OpenTelemetry Packages Installed (via `langsmith[otel]`)
- `opentelemetry-api` (v1.39.1)
- `opentelemetry-sdk` (v1.39.1)
- `opentelemetry-exporter-otlp-proto-http` (v1.39.1)
- `opentelemetry-instrumentation-aiohttp-client` (v0.60b1)

### Current Instrumentation Files
| File | Purpose |
|------|---------|
| `agent/utils/langsmith.py` | LangSmith trace URL generation |
| `agent/utils/mlflow_tracing.py` | MLflow initialization + evaluation logging |
| `agent/server.py:60-75` | MLflow initialization at import time |
| `agent/utils/slack.py` | Posts trace URLs to Slack |
| `agent/utils/linear.py` | Posts trace URLs to Linear |

### Problems
- Tracing setup is scattered across multiple files
- Each backend has its own initialization pattern
- Adding new backends requires touching multiple files
- MLflow imports fail if package not installed

---

## Proposed Architecture

### Directory Structure

```
agent/
├── observability/
│   ├── __init__.py           # Public API: Tracer, get_tracer(), init_observability()
│   ├── config.py             # ObservabilityConfig dataclass
│   ├── tracer.py             # Tracer class implementation
│   └── backends/
│       ├── __init__.py       # Backend registry
│       ├── base.py           # TracingBackend protocol
│       ├── langsmith.py      # LangSmith backend (always available)
│       └── mlflow.py         # MLflow backend (optional)
```

---

## Key Components

### 1. `ObservabilityConfig` (config.py)

```python
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
        ...
```

### 2. `TracingBackend` Protocol (backends/base.py)

```python
from typing import Protocol, ContextManager

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
        """Check if backend is configured/enabled."""
        ...
    
    def initialize(self) -> bool:
        """Initialize the backend. Returns True on success."""
        ...
    
    def get_trace_url(self, run_id: str) -> str | None:
        """Get the trace URL for a given run ID."""
        ...
    
    def create_span(self, name: str, **attributes) -> ContextManager:
        """Create a tracing span (optional, for manual instrumentation)."""
        ...
```

### 3. `Tracer` Class (tracer.py)

```python
class Tracer:
    """Unified tracer that delegates to configured backends."""
    
    _instance: "Tracer | None" = None
    
    def __init__(self, config: ObservabilityConfig | None = None):
        self._config = config or ObservabilityConfig.from_env()
        self._backends: list[TracingBackend] = []
        self._initialized = False
        self._load_backends()
    
    def _load_backends(self) -> None:
        """Load all available backends."""
        # Always try to load LangSmith
        from .backends.langsmith import LangSmithBackend
        self._backends.append(LangSmithBackend(self._config))
        
        # Try to load MLflow (optional)
        try:
            from .backends.mlflow import MLflowBackend
            self._backends.append(MLflowBackend(self._config))
        except ImportError:
            pass  # MLflow not installed
    
    def initialize(self) -> dict[str, bool]:
        """Initialize all enabled backends. Returns {name: success}."""
        results = {}
        for backend in self._backends:
            if backend.is_available() and backend.is_enabled():
                results[backend.name] = backend.initialize()
        self._initialized = True
        return results
    
    def get_trace_urls(self, run_id: str) -> dict[str, str]:
        """Get trace URLs from all initialized backends."""
        urls = {}
        for backend in self._backends:
            if backend.is_enabled():
                url = backend.get_trace_url(run_id)
                if url:
                    urls[backend.name] = url
        return urls
    
    @classmethod
    def get_instance(cls) -> "Tracer":
        """Get or create the singleton tracer instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    return Tracer.get_instance()
```

### 4. Backend Implementations

#### LangSmith Backend (backends/langsmith.py)
```python
class LangSmithBackend:
    """LangSmith tracing backend (always available)."""
    
    name = "LangSmith"
    
    def __init__(self, config: ObservabilityConfig):
        self._config = config
    
    def is_available(self) -> bool:
        return True  # LangSmith is a core dependency
    
    def is_enabled(self) -> bool:
        return bool(
            self._config.langsmith_tenant_id 
            and self._config.langsmith_project_id
        )
    
    def initialize(self) -> bool:
        # LangSmith auto-initializes via langchain, nothing to do
        return True
    
    def get_trace_url(self, run_id: str) -> str | None:
        # Existing logic from agent/utils/langsmith.py
        ...
```

#### MLflow Backend (backends/mlflow.py)
```python
# Check availability at module level
MLFLOW_AVAILABLE = False
try:
    import mlflow
    import mlflow.langchain
    MLFLOW_AVAILABLE = True
except ImportError:
    pass


class MLflowBackend:
    """MLflow/Databricks tracing backend (optional)."""
    
    name = "MLflow"
    
    def __init__(self, config: ObservabilityConfig):
        self._config = config
        self._initialized = False
    
    def is_available(self) -> bool:
        return MLFLOW_AVAILABLE
    
    def is_enabled(self) -> bool:
        return bool(
            self._config.mlflow_tracking_uri 
            or self._config.databricks_host
        )
    
    def initialize(self) -> bool:
        if not self.is_available():
            return False
        # Existing logic from agent/utils/mlflow_tracing.py
        ...
    
    def get_trace_url(self, run_id: str) -> str | None:
        # Existing logic
        ...
```

---

## Updated Usage

### In `agent/server.py`
```python
# Before
from .utils.mlflow_tracing import initialize_mlflow_tracing, is_mlflow_enabled

if is_mlflow_enabled():
    if initialize_mlflow_tracing():
        logger.info("MLflow tracing initialized successfully")

# After
from .observability import get_tracer

tracer = get_tracer()
results = tracer.initialize()
for backend, success in results.items():
    if success:
        logger.info("%s tracing initialized successfully", backend)
    else:
        logger.warning("%s tracing initialization failed", backend)
```

### In `agent/utils/slack.py`
```python
# Before
from agent.utils.langsmith import get_langsmith_trace_url
from agent.utils.mlflow_tracing import get_mlflow_run_url, is_mlflow_enabled

async def post_slack_trace_reply(channel_id: str, thread_ts: str, run_id: str) -> None:
    trace_url = get_langsmith_trace_url(run_id)
    mlflow_url = get_mlflow_run_url(run_id) if is_mlflow_enabled() else None
    # ... build links manually

# After
from agent.observability import get_tracer

async def post_slack_trace_reply(channel_id: str, thread_ts: str, run_id: str) -> None:
    trace_urls = get_tracer().get_trace_urls(run_id)
    
    if trace_urls:
        links = [f"<{url}|{name}>" for name, url in trace_urls.items()]
        await post_slack_thread_reply(
            channel_id, thread_ts, f"Working on it! {' | '.join(links)}"
        )
```

### In `agent/utils/linear.py`
```python
# After
from agent.observability import get_tracer

async def post_linear_trace_comment(issue_id: str, run_id: str, triggering_comment_id: str) -> None:
    trace_urls = get_tracer().get_trace_urls(run_id)
    
    if trace_urls:
        links = [f"[{name}]({url})" for name, url in trace_urls.items()]
        await comment_on_linear_issue(
            issue_id,
            f"On it! {' | '.join(links)}",
            parent_id=triggering_comment_id or None,
        )
```

---

## Implementation Checklist

### Phase 1: Create Observability Module
- [ ] Create `agent/observability/` directory
- [ ] Create `agent/observability/__init__.py` with public exports
- [ ] Create `agent/observability/config.py` with `ObservabilityConfig`
- [ ] Create `agent/observability/backends/__init__.py`
- [ ] Create `agent/observability/backends/base.py` with `TracingBackend` protocol
- [ ] Create `agent/observability/backends/langsmith.py` (move from `utils/langsmith.py`)
- [ ] Create `agent/observability/backends/mlflow.py` (move from `utils/mlflow_tracing.py`)
- [ ] Create `agent/observability/tracer.py` with `Tracer` class

### Phase 2: Update Consumers
- [ ] Update `agent/server.py` to use `get_tracer().initialize()`
- [ ] Update `agent/utils/slack.py` to use `get_tracer().get_trace_urls()`
- [ ] Update `agent/utils/linear.py` to use `get_tracer().get_trace_urls()`
- [ ] Update `agent/webapp.py` if needed

### Phase 3: Cleanup & Dependencies
- [ ] Move MLflow dependencies to optional extras in `pyproject.toml`
- [ ] Delete or deprecate `agent/utils/mlflow_tracing.py`
- [ ] Keep `agent/utils/langsmith.py` as re-export for backwards compatibility (optional)
- [ ] Update `INSTALLATION.md` with optional MLflow install instructions

### Phase 4: Testing
- [ ] Verify agent works without MLflow installed
- [ ] Verify agent works with MLflow installed and configured
- [ ] Verify LangSmith trace URLs still work
- [ ] Verify MLflow trace URLs work when enabled
- [ ] Run linter (`ruff check`)

---

## Environment Variables

### Existing (LangSmith)
```bash
LANGSMITH_API_KEY_PROD=""
LANGSMITH_URL_PROD="https://smith.langchain.com"
LANGSMITH_TENANT_ID_PROD=""
LANGSMITH_TRACING_PROJECT_ID_PROD=""
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT=""
```

### New (MLflow - Optional)
```bash
# Standard MLflow
MLFLOW_TRACKING_URI=""
MLFLOW_EXPERIMENT_NAME="open-swe-agent"

# Databricks MLflow
DATABRICKS_HOST=""
DATABRICKS_TOKEN=""

# Autolog settings
MLFLOW_ENABLE_AUTOLOG="true"
MLFLOW_LOG_MODELS="false"
MLFLOW_LOG_INPUT_EXAMPLES="true"
```

---

## pyproject.toml Changes

```toml
[project]
dependencies = [
    # ... existing deps ...
    # Remove mlflow and databricks-sdk from here
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
]
mlflow = [
    "mlflow>=2.15.0",
    "databricks-sdk>=0.30.0",
]
```

**Install with:** `uv sync --extra mlflow` or `pip install open-swe-agent[mlflow]`

---

## Future Extensibility

To add a new backend (e.g., Datadog):

1. Create `agent/observability/backends/datadog.py`
2. Implement the `TracingBackend` protocol
3. Add to the backend loading logic in `Tracer._load_backends()`
4. Add optional dependencies to `pyproject.toml`

```python
# Example: agent/observability/backends/datadog.py
DATADOG_AVAILABLE = False
try:
    from ddtrace import tracer as dd_tracer
    DATADOG_AVAILABLE = True
except ImportError:
    pass


class DatadogBackend:
    name = "Datadog"
    
    def is_available(self) -> bool:
        return DATADOG_AVAILABLE
    
    # ... implement other methods
```

---

## Open Questions

1. **Should we keep `agent/utils/langsmith.py` for backwards compatibility?**
   - Option A: Delete it, update all imports
   - Option B: Keep it as a re-export shim

2. **Should the `Tracer` be a singleton or instantiated per-request?**
   - Current plan: Singleton (simpler, matches current behavior)

3. **Should we expose OpenTelemetry spans directly?**
   - Current plan: Not initially, but the `create_span()` method is in the protocol for future use

4. **MLflow Evaluation Logger - keep or remove?**
   - The current `mlflow_tracing.py` has `MLflowEvaluationLogger` for logging LLM interactions
   - Should this be part of the `MLflowBackend` or a separate concern?

---

## Notes

- Edit this file to adjust the plan before implementation
- When ready, exit plan mode and I'll implement the changes

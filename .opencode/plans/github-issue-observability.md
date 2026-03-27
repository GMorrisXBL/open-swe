# GitHub Issue: Add Pluggable Observability/Tracing Backend Support

Copy the content below to create a GitHub issue in the open-swe repository.

---

## Issue Title

**Feature Request: Add pluggable observability backend support (MLflow, Datadog, etc.)**

---

## Issue Body

### Summary

Add support for pluggable observability/tracing backends beyond LangSmith, starting with **Databricks MLflow**. This would allow teams using different observability stacks to emit traces and LLM evaluation logs to their preferred platform.

### Motivation

Currently, open-swe has excellent LangSmith integration for tracing, but many enterprise teams use other observability platforms:

- **Databricks MLflow** - Common in ML/data teams already using Databricks
- **Datadog** - Popular enterprise APM
- **New Relic**, **Honeycomb**, etc.

Since LangSmith already uses OpenTelemetry under the hood (`langsmith[otel]` brings in `opentelemetry-api`, `opentelemetry-sdk`, etc.), we can leverage this to support multiple backends through a unified abstraction.

### Proposed Solution

#### 1. Create a unified `agent/observability/` module

```
agent/
├── observability/
│   ├── __init__.py           # Public API: get_tracer()
│   ├── config.py             # ObservabilityConfig dataclass
│   ├── tracer.py             # Tracer class
│   └── backends/
│       ├── base.py           # TracingBackend protocol
│       ├── langsmith.py      # LangSmith (always available)
│       └── mlflow.py         # MLflow (optional)
```

#### 2. Define a `TracingBackend` protocol

```python
class TracingBackend(Protocol):
    name: str
    
    def is_available(self) -> bool:
        """Check if dependencies are installed."""
        ...
    
    def is_enabled(self) -> bool:
        """Check if backend is configured."""
        ...
    
    def initialize(self) -> bool:
        """Initialize the backend."""
        ...
    
    def get_trace_url(self, run_id: str) -> str | None:
        """Get trace URL for a run."""
        ...
```

#### 3. Provide a unified `Tracer` API

```python
from agent.observability import get_tracer

# In server.py - single initialization point
tracer = get_tracer()
tracer.initialize()

# In slack.py/linear.py - get all trace URLs
trace_urls = get_tracer().get_trace_urls(run_id)
# Returns: {"LangSmith": "https://...", "MLflow": "https://..."}
```

#### 4. Make additional backends optional dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
mlflow = ["mlflow>=2.15.0", "databricks-sdk>=0.30.0"]
```

Install with: `pip install open-swe[mlflow]`

### Benefits

1. **No breaking changes** - LangSmith continues to work as-is
2. **Optional dependencies** - MLflow only installed when needed
3. **Extensible** - Easy to add new backends (Datadog, etc.)
4. **Cleaner architecture** - Consolidates scattered tracing code
5. **Enterprise-friendly** - Teams can use their existing observability stack

### Implementation Details

#### Environment Variables (MLflow)

```bash
# Standard MLflow
MLFLOW_TRACKING_URI=""
MLFLOW_EXPERIMENT_NAME="open-swe-agent"

# Databricks MLflow
DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
DATABRICKS_TOKEN=""
```

#### Graceful Degradation

- If MLflow isn't installed, the backend is silently skipped
- If MLflow is installed but not configured, it's disabled
- LangSmith remains the default and always-available option

#### Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add `mlflow` optional extras |
| `agent/observability/*` | New module (see structure above) |
| `agent/server.py` | Use `get_tracer().initialize()` |
| `agent/utils/slack.py` | Use `get_tracer().get_trace_urls()` |
| `agent/utils/linear.py` | Use `get_tracer().get_trace_urls()` |
| `INSTALLATION.md` | Document optional MLflow setup |

### Alternatives Considered

1. **Just add MLflow directly** - Less maintainable, harder to add future backends
2. **Use OpenTelemetry exporters only** - Loses backend-specific features like trace URLs
3. **External plugin system** - Over-engineered for this use case

### Additional Context

I have a working implementation locally and am happy to submit a PR. The implementation:

- Uses lazy imports to avoid overhead when MLflow isn't used
- Includes an `MLflowEvaluationLogger` for structured LLM evaluation logging
- Passes `ruff` linting
- Maintains backwards compatibility

### Checklist

- [ ] Maintainers approve the general approach
- [ ] Decide on module location (`agent/observability/` vs `agent/tracing/`)
- [ ] Decide whether to keep `agent/utils/langsmith.py` for backwards compat
- [ ] PR ready for review

---

## Labels (suggested)

- `enhancement`
- `observability`
- `good first issue` (if maintainers want to implement themselves)

---

## Notes for You

- Edit this issue as needed before posting
- You can reference the detailed implementation plan at `.opencode/plans/observability-abstraction.md`
- Consider linking to your fork/branch if you've pushed the implementation

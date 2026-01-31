# Structured Logging and Circuit Breaker Documentation

## Overview

This document describes the structured logging and circuit breaker patterns implemented in the RAG Orchestrator for production resilience.

## Structured Logging

### Features

- **JSON formatting in production**: Structured logs with consistent fields
- **Context binding**: Add request-specific context to all log messages
- **Log levels**: DEBUG, INFO, WARNING, ERROR with appropriate filtering
- **Third-party noise reduction**: Suppresses verbose logs from libraries

### Usage

```python
from src.core.logging import get_logger, setup_logging

# Initialize logging (typically in app startup)
setup_logging()

# Get a logger
logger = get_logger(__name__)

# Basic logging
logger.info("operation_completed", duration_ms=45.2)
logger.error("operation_failed", error=str(e))

# Context binding
request_logger = logger.bind(request_id="abc123", user="user@example.com")
request_logger.info("request_processed", status_code=200)
```

### Log Format

**Development** (readable format):
```
2026-01-31 00:27:41,410 | INFO | src.api.app | application_starting
```

**Production** (JSON format):
```json
{
  "timestamp": "2026-01-31T00:27:41.410Z",
  "level": "INFO",
  "logger": "src.api.app",
  "message": "application_starting",
  "module": "app",
  "function": "lifespan",
  "line": 25,
  "environment": "production",
  "debug": false
}
```

### Configuration

Logging behavior is controlled by environment variables:
- `ENVIRONMENT=production`: Enables JSON logging
- `DEBUG=true`: Sets log level to DEBUG (otherwise INFO)

## Circuit Breaker

### Features

- **Three states**: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- **Automatic failure tracking**: Opens after configurable threshold
- **Auto-recovery**: Transitions to HALF_OPEN after timeout, then CLOSED after successful calls
- **Thread-safe registry**: Centralized management of multiple circuit breakers
- **Statistics tracking**: Failures, successes, and state information

### Usage

#### Direct Usage

```python
from src.infrastructure.resilience import CircuitBreaker

# Create a circuit breaker
breaker = CircuitBreaker(
    name="external_service",
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=30.0,    # Try recovery after 30 seconds
    half_open_max_calls=3,    # Confirm recovery with 3 successes
)

# Wrap calls
async def call_external_service():
    return await breaker.call(my_async_function, arg1, arg2)
```

#### Using Registry

```python
from src.infrastructure.resilience.registry import CircuitBreakerRegistry

# Get or create a breaker (recommended for shared services)
breaker = CircuitBreakerRegistry.get_or_create(
    name="openai",
    failure_threshold=5,
    recovery_timeout=30.0,
)

# Use the breaker
result = await breaker.call(async_function)
```

### Circuit States

1. **CLOSED** (Normal Operation)
   - All requests pass through
   - Failures are counted
   - Opens when failure threshold is reached

2. **OPEN** (Service Failing)
   - All requests are rejected immediately
   - Prevents cascading failures
   - Transitions to HALF_OPEN after recovery timeout

3. **HALF_OPEN** (Testing Recovery)
   - Limited requests are allowed
   - Success: Counts toward recovery
   - Failure: Returns to OPEN immediately
   - After N successes: Transitions to CLOSED

### Monitoring Endpoints

#### Get Circuit Breaker Status

```bash
GET /api/v1/monitoring/circuits
```

Response:
```json
{
  "circuit_breakers": [
    {
      "name": "openai",
      "state": "closed",
      "failures": 0,
      "successes": 142,
      "failure_threshold": 5,
      "recovery_timeout": 30.0
    },
    {
      "name": "qdrant",
      "state": "open",
      "failures": 5,
      "successes": 98,
      "failure_threshold": 5,
      "recovery_timeout": 30.0
    }
  ],
  "timestamp": "2026-01-31T00:27:41.410Z"
}
```

#### Reset Circuit Breaker

```bash
POST /api/v1/monitoring/circuits/{name}/reset
```

Response:
```json
{
  "status": "reset",
  "circuit": "openai"
}
```

## Implementation Details

### Services with Circuit Breakers

1. **OpenAI Client** (`src/infrastructure/llm/openai_client.py`)
   - Circuit breaker: `openai`
   - Protects: All LLM generation calls
   - Logs: Request start, completion, failures

2. **Qdrant Client** (`src/infrastructure/vector/qdrant_client.py`)
   - Circuit breaker: `qdrant`
   - Protects: Vector search operations
   - Logs: Search operations, collection management

### Request Correlation

All API requests are assigned a unique `request_id` that:
- Appears in all logs for that request
- Is returned in the `X-Request-ID` header
- Enables tracing requests through the system

Example:
```
2026-01-31 00:27:41,410 | INFO | src.api.app | request_started
2026-01-31 00:27:41,415 | INFO | src.infrastructure.llm.openai_client | generate_started
2026-01-31 00:27:41,520 | INFO | src.infrastructure.llm.openai_client | generate_completed
2026-01-31 00:27:41,521 | INFO | src.api.app | request_completed
```

All logs include: `request_id=abc12345`

## Benefits

### 1. Production Observability
- Structured logs are easily parsed by log aggregation tools (ELK, Splunk, etc.)
- Request correlation enables distributed tracing
- Consistent format across all services

### 2. Failure Isolation
- Circuit breakers prevent cascading failures
- Fast-fail when services are down
- Automatic recovery when services stabilize

### 3. Operational Insights
- Circuit breaker metrics show service health
- Log context provides debugging information
- Performance metrics in every request

### 4. Developer Experience
- Easy-to-use logging API
- Readable logs in development
- Automatic context propagation

## Configuration Examples

### Development Environment
```bash
ENVIRONMENT=development
DEBUG=true
OPENAI_API_KEY=sk-...
```

Logs will be readable and verbose.

### Production Environment
```bash
ENVIRONMENT=production
DEBUG=false
OPENAI_API_KEY=sk-...
```

Logs will be JSON-formatted and filtered to INFO level.

## Best Practices

1. **Use context binding** for request-specific information
2. **Log at appropriate levels**:
   - DEBUG: Detailed diagnostic information
   - INFO: Significant events (requests, completions)
   - WARNING: Recoverable issues
   - ERROR: Failures requiring attention

3. **Monitor circuit breaker metrics** to detect service issues
4. **Set appropriate thresholds** based on service SLAs
5. **Use structured fields** instead of string formatting in log messages

## Troubleshooting

### Circuit Breaker Stuck Open?
1. Check service health directly
2. Review recent error logs for the service
3. Manually reset via monitoring endpoint if needed
4. Adjust threshold/timeout if too sensitive

### Logs Not Appearing?
1. Verify `setup_logging()` is called at startup
2. Check log level configuration
3. Ensure third-party log filtering isn't too aggressive

### Performance Impact?
- Logging: Minimal overhead (~1ms per log)
- Circuit breakers: Microseconds for state checks
- Both are production-optimized and well-tested

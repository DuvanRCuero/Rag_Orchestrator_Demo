"""Circuit Breaker implementation for resilience."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_requests: int = 0
    total_failures: int = 0


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """Circuit Breaker for protecting against cascading failures."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if not self._can_execute():
                logger.warning(
                    "circuit_breaker_rejected",
                    name=self.name,
                    state=self.state.value,
                )
                raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure(e)
            raise

    def _can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

        return False

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.stats.last_failure_time is None:
            return True
        elapsed = time.time() - self.stats.last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        logger.info(
            "circuit_breaker_half_open",
            name=self.name,
            previous_state=self.state.value,
        )
        self.state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._half_open_successes = 0

    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self.stats.successes += 1
            self.stats.total_requests += 1
            self.stats.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to_closed()

    async def _record_failure(self, error: Exception) -> None:
        """Record failed call."""
        async with self._lock:
            self.stats.failures += 1
            self.stats.total_failures += 1
            self.stats.total_requests += 1
            self.stats.last_failure_time = time.time()

            logger.warning(
                "circuit_breaker_failure",
                name=self.name,
                state=self.state.value,
                error=str(error),
                failure_count=self.stats.failures,
            )

            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.stats.failures >= self.config.failure_threshold:
                self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        logger.error(
            "circuit_breaker_opened",
            name=self.name,
            failures=self.stats.failures,
            threshold=self.config.failure_threshold,
        )
        self.state = CircuitState.OPEN
        self.stats.failures = 0

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        logger.info(
            "circuit_breaker_closed",
            name=self.name,
            successes=self._half_open_successes,
        )
        self.state = CircuitState.CLOSED
        self.stats.failures = 0
        self._half_open_calls = 0
        self._half_open_successes = 0

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "stats": {
                "failures": self.stats.failures,
                "successes": self.stats.successes,
                "total_requests": self.stats.total_requests,
                "total_failures": self.stats.total_failures,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
            },
        }

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        logger.info("circuit_breaker_manual_reset", name=self.name)
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._half_open_calls = 0
        self._half_open_successes = 0


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator to wrap async functions with circuit breaker."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator

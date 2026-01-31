"""Circuit Breaker pattern implementation."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


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


class CircuitBreaker:
    """
    Circuit Breaker for external service calls.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def _check_state(self):
        """Check if circuit should transition states."""
        if self._state == CircuitState.OPEN:
            if self._stats.last_failure_time:
                elapsed = time.time() - self._stats.last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        "circuit_half_open",
                        circuit=self.name,
                        elapsed_seconds=elapsed,
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0

    async def _record_success(self):
        """Record a successful call."""
        async with self._lock:
            self._stats.successes += 1
            self._stats.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    logger.info(
                        "circuit_closed",
                        circuit=self.name,
                        reason="recovery_confirmed",
                    )
                    self._state = CircuitState.CLOSED
                    self._stats.failures = 0

    async def _record_failure(self, error: Exception):
        """Record a failed call."""
        async with self._lock:
            self._stats.failures += 1
            self._stats.last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "circuit_opened",
                    circuit=self.name,
                    reason="half_open_failure",
                    error=str(error),
                )
                self._state = CircuitState.OPEN

            elif self._state == CircuitState.CLOSED:
                if self._stats.failures >= self.failure_threshold:
                    logger.warning(
                        "circuit_opened",
                        circuit=self.name,
                        reason="failure_threshold",
                        failures=self._stats.failures,
                        threshold=self.failure_threshold,
                    )
                    self._state = CircuitState.OPEN

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function through the circuit breaker."""
        await self._check_state()

        if self._state == CircuitState.OPEN:
            logger.warning(
                "circuit_rejected",
                circuit=self.name,
                state=self._state.value,
            )
            raise CircuitOpenError(f"Circuit {self.name} is open")

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

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failures": self._stats.failures,
            "successes": self._stats.successes,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
):
    """Decorator to add circuit breaker to a function."""
    breaker = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        wrapper._circuit_breaker = breaker
        return wrapper

    return decorator

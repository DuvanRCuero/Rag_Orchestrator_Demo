"""Circuit Breaker implementation for resilience."""

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
class CircuitBreaker:
    """Circuit Breaker for external service calls."""
    
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    
    # State
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    half_open_calls: int = field(default=0)
    
    def __post_init__(self):
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            await self._check_state()
            
            if self.state == CircuitState.OPEN:
                logger.warning(
                    "circuit_breaker_open",
                    breaker=self.name,
                    retry_after=self._time_until_retry(),
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open. Retry after {self._time_until_retry():.1f}s"
                )

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise

    async def _check_state(self) -> None:
        """Check and potentially update circuit state."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(
                    "circuit_breaker_half_open",
                    breaker=self.name,
                )
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    logger.info(
                        "circuit_breaker_closed",
                        breaker=self.name,
                        after_successes=self.half_open_calls,
                    )
                    self._reset()
            else:
                self.failure_count = 0

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(
                "circuit_breaker_failure",
                breaker=self.name,
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
                error=str(error),
            )
            
            if self.state == CircuitState.HALF_OPEN:
                self._trip()
            elif self.failure_count >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        """Open the circuit breaker."""
        self.state = CircuitState.OPEN
        logger.error(
            "circuit_breaker_tripped",
            breaker=self.name,
            failure_count=self.failure_count,
            recovery_timeout=self.recovery_timeout,
        )

    def _reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _time_until_retry(self) -> float:
        """Get time remaining until retry is allowed."""
        if self.last_failure_time is None:
            return 0
        elapsed = time.time() - self.last_failure_time
        return max(0, self.recovery_timeout - elapsed)

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
):
    """Decorator to apply circuit breaker to async functions."""
    breaker = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator

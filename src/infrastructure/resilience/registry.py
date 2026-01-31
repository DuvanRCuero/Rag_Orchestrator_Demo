"""Circuit Breaker Registry for managing multiple breakers."""

from typing import Dict
from src.infrastructure.resilience.circuit_breaker import CircuitBreaker, CircuitState, CircuitStats


class CircuitBreakerRegistry:
    """Registry for managing circuit breakers."""

    _breakers: Dict[str, CircuitBreaker] = {}

    @classmethod
    def get_or_create(
        cls,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        if name not in cls._breakers:
            cls._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return cls._breakers[name]

    @classmethod
    def get_all_stats(cls) -> list:
        return [breaker.get_stats() for breaker in cls._breakers.values()]

    @classmethod
    def reset(cls, name: str):
        if name in cls._breakers:
            cls._breakers[name]._state = CircuitState.CLOSED
            cls._breakers[name]._stats = CircuitStats()

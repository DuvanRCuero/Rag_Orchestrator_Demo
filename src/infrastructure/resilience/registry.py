"""Circuit Breaker Registry for managing multiple breakers."""

from typing import Dict
from src.infrastructure.resilience.circuit_breaker import CircuitBreaker


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
        """Get existing or create new circuit breaker."""
        if name not in cls._breakers:
            cls._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return cls._breakers[name]
    
    @classmethod
    def get(cls, name: str) -> CircuitBreaker:
        """Get a circuit breaker by name."""
        return cls._breakers.get(name)
    
    @classmethod
    def all_stats(cls) -> dict:
        """Get stats for all circuit breakers."""
        return {name: cb.stats for name, cb in cls._breakers.items()}
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers."""
        for cb in cls._breakers.values():
            cb._reset()

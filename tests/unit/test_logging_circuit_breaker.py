"""Test structured logging and circuit breaker functionality."""

import asyncio
import logging
import sys
import pytest

from src.core.logging import get_logger, setup_logging
from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
)


class TestStructuredLogging:
    """Test structured logging functionality."""

    def test_get_logger(self):
        """Test getting a logger instance."""
        import structlog
        logger = get_logger("test")
        # structlog returns a lazy proxy, so we just check it's a structlog logger
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'error')

    def test_setup_logging(self):
        """Test logging setup."""
        setup_logging("INFO")
        logger = get_logger("test")
        assert logger is not None


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test successful calls through circuit breaker."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.stats.successes == 1
        assert breaker.stats.failures == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        
        async def failing_func():
            raise ValueError("test error")
        
        # Fail 3 times to open circuit
        for _ in range(3):
            try:
                await breaker.call(failing_func)
            except ValueError:
                pass
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.total_failures == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects calls when open."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))
        
        async def failing_func():
            raise ValueError("test error")
        
        # Open the circuit
        for _ in range(2):
            try:
                await breaker.call(failing_func)
            except ValueError:
                pass
        
        # Next call should be rejected
        with pytest.raises(CircuitOpenError):
            await breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker transitions to half-open."""
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=0.1,  # Short timeout for testing
            )
        )
        
        async def failing_func():
            raise ValueError("test error")
        
        # Open the circuit
        for _ in range(2):
            try:
                await breaker.call(failing_func)
            except ValueError:
                pass
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Try to execute - should transition to half-open
        async def success_func():
            return "success"
        
        await breaker.call(success_func)
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovers after successful calls."""
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=0.1,
                half_open_max_calls=2,
                success_threshold=2,
            )
        )
        
        async def success_func():
            return "success"
        
        async def failing_func():
            raise ValueError("test error")
        
        # Open the circuit
        for _ in range(2):
            try:
                await breaker.call(failing_func)
            except ValueError:
                pass
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Make successful calls to close circuit
        for _ in range(2):
            await breaker.call(success_func)
        
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_get_status(self):
        """Test getting circuit breaker status."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))
        
        status = breaker.get_status()
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["stats"]["failures"] == 0
        assert status["stats"]["successes"] == 0
        assert status["config"]["failure_threshold"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

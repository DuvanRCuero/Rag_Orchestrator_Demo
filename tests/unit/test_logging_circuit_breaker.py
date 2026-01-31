"""Test structured logging and circuit breaker functionality."""

import asyncio
import logging
import sys
import pytest

from src.core.logging import StructuredLogger, JSONFormatter, get_logger, setup_logging
from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
)


class TestStructuredLogging:
    """Test structured logging functionality."""

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test")
        assert isinstance(logger, StructuredLogger)
        assert logger.logger.name == "test"

    def test_logger_bind(self):
        """Test logger context binding."""
        logger = get_logger("test")
        bound_logger = logger.bind(request_id="123", user="test")
        
        assert bound_logger._context["request_id"] == "123"
        assert bound_logger._context["user"] == "test"

    def test_json_formatter(self):
        """Test JSON formatter."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        
        formatted = formatter.format(record)
        assert "test message" in formatted
        assert "INFO" in formatted
        assert "timestamp" in formatted


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test successful calls through circuit breaker."""
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker._stats.successes == 1
        assert breaker._stats.failures == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        async def failing_func():
            raise ValueError("test error")
        
        # Fail 3 times to open circuit
        for _ in range(3):
            try:
                await breaker.call(failing_func)
            except ValueError:
                pass
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open
        assert breaker._stats.failures == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects calls when open."""
        breaker = CircuitBreaker("test", failure_threshold=2)
        
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
            failure_threshold=2,
            recovery_timeout=0.1,  # Short timeout for testing
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
        
        # Check state should transition to half-open
        await breaker._check_state()
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovers after successful calls."""
        breaker = CircuitBreaker(
            "test",
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2,
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
        await breaker._check_state()
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Make successful calls to close circuit
        for _ in range(2):
            await breaker.call(success_func)
        
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_get_stats(self):
        """Test getting circuit breaker statistics."""
        breaker = CircuitBreaker("test", failure_threshold=5)
        
        stats = breaker.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failures"] == 0
        assert stats["successes"] == 0
        assert stats["failure_threshold"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

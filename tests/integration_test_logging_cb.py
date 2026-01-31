#!/usr/bin/env python3
"""Simple integration test for logging and circuit breaker."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock settings to avoid dependency issues
os.environ['OPENAI_API_KEY'] = 'test-key'

from src.core.logging import get_logger, setup_logging
from src.infrastructure.resilience import CircuitBreaker, CircuitState
from src.infrastructure.resilience.registry import CircuitBreakerRegistry


async def test_logging():
    """Test structured logging."""
    print("\n=== Testing Structured Logging ===")
    
    # Setup logging
    setup_logging()
    
    # Get logger
    logger = get_logger("test")
    
    # Test basic logging
    logger.info("test_started", component="integration_test")
    logger.debug("debug_message", data="test")
    logger.warning("warning_message", code=123)
    logger.error("error_message", error="test error")
    
    # Test context binding
    bound_logger = logger.bind(request_id="abc123", user="test_user")
    bound_logger.info("request_processed", duration_ms=45.2)
    
    print("✅ Structured logging tests passed")


async def test_circuit_breaker():
    """Test circuit breaker."""
    print("\n=== Testing Circuit Breaker ===")
    
    # Create circuit breaker
    breaker = CircuitBreaker("test_service", failure_threshold=3, recovery_timeout=1.0)
    
    # Test successful calls
    async def success_func():
        return "success"
    
    result = await breaker.call(success_func)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
    print(f"✅ Circuit breaker in CLOSED state after success: {breaker.get_stats()}")
    
    # Test failures to open circuit
    async def failing_func():
        raise ValueError("Test error")
    
    for i in range(3):
        try:
            await breaker.call(failing_func)
        except ValueError:
            pass
    
    assert breaker.state == CircuitState.OPEN
    print(f"✅ Circuit breaker OPENED after {breaker._stats.failures} failures: {breaker.get_stats()}")
    
    # Test that calls are rejected when open
    try:
        await breaker.call(success_func)
        assert False, "Should have raised CircuitOpenError"
    except Exception as e:
        assert "Circuit" in str(e) and "open" in str(e)
        print(f"✅ Circuit breaker correctly rejects calls when OPEN")
    
    # Test recovery to half-open
    await asyncio.sleep(1.1)
    await breaker._check_state()
    assert breaker.state == CircuitState.HALF_OPEN
    print(f"✅ Circuit breaker transitioned to HALF_OPEN after timeout: {breaker.get_stats()}")
    
    # Test recovery to closed
    breaker2 = CircuitBreaker("test_service2", failure_threshold=2, recovery_timeout=0.5, half_open_max_calls=2)
    
    # Open it
    for _ in range(2):
        try:
            await breaker2.call(failing_func)
        except ValueError:
            pass
    
    assert breaker2.state == CircuitState.OPEN
    
    # Wait and check state
    await asyncio.sleep(0.6)
    await breaker2._check_state()
    assert breaker2.state == CircuitState.HALF_OPEN
    
    # Make successful calls to close
    for _ in range(2):
        await breaker2.call(success_func)
    
    assert breaker2.state == CircuitState.CLOSED
    print(f"✅ Circuit breaker recovered to CLOSED after successful calls: {breaker2.get_stats()}")


async def test_circuit_breaker_registry():
    """Test circuit breaker registry."""
    print("\n=== Testing Circuit Breaker Registry ===")
    
    # Get or create breakers
    breaker1 = CircuitBreakerRegistry.get_or_create("service1")
    breaker2 = CircuitBreakerRegistry.get_or_create("service2")
    
    assert breaker1.name == "service1"
    assert breaker2.name == "service2"
    
    # Get all stats
    all_stats = CircuitBreakerRegistry.get_all_stats()
    assert len(all_stats) >= 2
    print(f"✅ Registry tracking {len(all_stats)} circuit breakers")
    
    # Reset a breaker
    breaker1._stats.failures = 5
    breaker1._state = CircuitState.OPEN
    CircuitBreakerRegistry.reset("service1")
    assert breaker1.state == CircuitState.CLOSED
    assert breaker1._stats.failures == 0
    print(f"✅ Circuit breaker reset works")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Integration Test: Logging + Circuit Breaker")
    print("=" * 60)
    
    try:
        await test_logging()
        await test_circuit_breaker()
        await test_circuit_breaker_registry()
        
        print("\n" + "=" * 60)
        print("✅ All integration tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

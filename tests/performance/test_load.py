import asyncio
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import aiohttp
import pytest

# Note: These tests require the API to be running
# They are marked as "slow" and should be run separately


@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance and load tests."""

    API_BASE_URL = "http://localhost:8000"

    @pytest.fixture
    def test_queries(self) -> List[Dict[str, Any]]:
        """Test queries for performance testing."""
        return [
            {"query": "How to optimize LangChain?", "top_k": 3},
            {"query": "What is RAG?", "top_k": 2},
            {"query": "Best vector database for production?", "top_k": 5},
            {"query": "How to reduce hallucination in LLMs?", "top_k": 4},
            {"query": "Caching strategies for RAG systems?", "top_k": 3},
            {"query": "Monitoring tools for AI applications?", "top_k": 2},
            {"query": "Cost optimization techniques?", "top_k": 3},
            {"query": "Security best practices for RAG?", "top_k": 4},
            {"query": "Scaling RAG systems horizontally?", "top_k": 5},
            {"query": "Async vs sync operations in LangChain?", "top_k": 3},
        ]

    async def make_async_request(
        self, session, query_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make an async request to the API."""
        url = f"{self.API_BASE_URL}/api/v1/query/ask"
        
        start_time = time.time()  # Track time manually
        
        try:
            async with session.post(url, json=query_data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                elapsed = time.time() - start_time
                data = await response.json() if response.status == 200 else None
                
                return {
                    "success": response.status == 200,
                    "response_time": elapsed,  # Use manual timing
                    "data": data,
                    "status": response.status,
                }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "data": None,
                "timeout": True,
            }
        except Exception as e:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "error": str(e),
            }

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_single_request_latency(self, test_queries):
        """Test latency of single requests."""
        async with aiohttp.ClientSession() as session:
            results = []

            for query_data in test_queries[:3]:  # Test first 3 queries
                start_time = time.time()
                result = await self.make_async_request(session, query_data)
                end_time = time.time()

                actual_time = end_time - start_time
                results.append(
                    {
                        "query": query_data["query"],
                        "api_time": result["response_time"],
                        "total_time": actual_time,
                        "success": result["success"],
                    }
                )

            # Analyze results
            api_times = [r["api_time"] for r in results]
            total_times = [r["total_time"] for r in results]

            print(f"\nSingle Request Performance:")
            print(f"Average API time: {statistics.mean(api_times):.3f}s")
            print(f"Average total time: {statistics.mean(total_times):.3f}s")
            print(f"Min API time: {min(api_times):.3f}s")
            print(f"Max API time: {max(api_times):.3f}s")

            # Assert performance requirements (adjust as needed)
            assert statistics.mean(api_times) < 5.0  # Should be under 5 seconds
            assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_requests(self, test_queries):
        """Test performance under concurrent load."""
        async with aiohttp.ClientSession() as session:
            # Create tasks for concurrent execution
            tasks = []
            for query_data in test_queries:
                task = self.make_async_request(session, query_data)
                tasks.append(task)

            # Execute concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            total_time = end_time - start_time
            successful = sum(1 for r in results if r["success"])
            response_times = [r["response_time"] for r in results if r["success"]]

            print(f"\nConcurrent Request Performance:")
            print(f"Total requests: {len(test_queries)}")
            print(f"Successful requests: {successful}")
            print(f"Total execution time: {total_time:.3f}s")
            print(f"Average response time: {statistics.mean(response_times):.3f}s")
            print(f"Throughput: {len(test_queries) / total_time:.2f} requests/second")

            # Assertions
            assert successful == len(test_queries)  # All should succeed
            assert statistics.mean(response_times) < 10.0  # Should be under 10 seconds
            assert len(test_queries) / total_time > 0.5  # At least 0.5 requests/second

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_load(self, test_queries):
        """Test sustained load over time."""
        async with aiohttp.ClientSession() as session:
            durations = []
            all_successful = True

            # Run for 30 seconds
            end_time = time.time() + 30
            request_count = 0

            while time.time() < end_time:
                # Use round-robin through queries
                query_data = test_queries[request_count % len(test_queries)]

                start_time = time.time()
                result = await self.make_async_request(session, query_data)
                duration = time.time() - start_time

                durations.append(duration)
                request_count += 1

                if not result["success"]:
                    all_successful = False

                # Small delay to avoid overwhelming
                await asyncio.sleep(0.1)

            print(f"\nSustained Load Performance:")
            print(f"Total requests: {request_count}")
            print(f"Test duration: 30 seconds")
            print(f"Average response time: {statistics.mean(durations):.3f}s")
            print(f"Max response time: {max(durations):.3f}s")
            print(f"Min response time: {min(durations):.3f}s")
            print(f"Throughput: {request_count / 30:.2f} requests/second")

            # Assertions
            assert all_successful
            assert statistics.mean(durations) < 3.0  # Average under 3 seconds
            assert request_count / 30 > 1.0  # At least 1 request/second sustained

    def test_memory_usage_single_request(self):
        """Test memory usage for single request (simplified)."""
        # This would require more sophisticated monitoring
        # For now, just a placeholder tests
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate some work
        _ = [i for i in range(1000000)]

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        print(f"\nMemory Usage Test:")
        print(f"Memory before: {memory_before:.2f} MB")
        print(f"Memory after: {memory_after:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")

        # Memory shouldn't increase too much
        assert memory_increase < 100  # Less than 100MB increase

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_retrieval_performance(self):
        """Test retrieval performance specifically."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.API_BASE_URL}/api/v1/query/retrieve-only"

            query_data = {
                "query": "LangChain production optimization",
                "top_k": "5",  # String instead of int
                "use_multi_query": "false",  # String instead of bool
                "use_reranking": "false",  # String instead of bool
            }

            # Run multiple times
            times = []
            for _ in range(10):
                start_time = time.time()
                async with session.post(url, params=query_data) as response:
                    await response.json()
                    end_time = time.time()
                    times.append(end_time - start_time)

            avg_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0

            print(f"\nRetrieval Performance:")
            print(f"Average time: {avg_time:.3f}s")
            print(f"Standard deviation: {std_dev:.3f}s")
            print(f"Min time: {min(times):.3f}s")
            print(f"Max time: {max(times):.3f}s")

            assert avg_time < 2.0  # Retrieval should be fast
            assert std_dev < avg_time * 0.5  # Reasonable consistency

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_error_handling_under_load(self):
        """Test error handling when system is under heavy load."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.API_BASE_URL}/api/v1/query/ask"

            # Create many concurrent requests
            tasks = []
            for i in range(50):  # 50 concurrent requests
                query_data = {"query": f"Test query {i}", "top_k": 5}
                task = session.post(url, json=query_data, timeout=30)
                tasks.append(task)

            # Execute with timeout
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Analyze results
                success_count = 0
                error_count = 0
                timeout_count = 0

                for response in responses:
                    if isinstance(response, Exception):
                        error_count += 1
                        if "timeout" in str(response).lower():
                            timeout_count += 1
                    elif response.status == 200:
                        success_count += 1
                    else:
                        error_count += 1

                print(f"\nError Handling Under Load:")
                print(f"Total requests: 50")
                print(f"Successful: {success_count}")
                print(f"Errors: {error_count}")
                print(f"Timeouts: {timeout_count}")

                # System should handle load gracefully
                assert (
                    success_count > 25
                )  # At least 50% should succeed under heavy load
                assert timeout_count < 10  # Not too many timeouts

            except Exception as e:
                print(f"Load tests failed: {e}")
                # Even if tests fails, we want to know why
                raise

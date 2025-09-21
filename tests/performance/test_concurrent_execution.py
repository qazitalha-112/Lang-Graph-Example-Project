"""
Performance tests for concurrent subagent execution.

This module contains performance tests that validate the system's ability
to handle concurrent subagent execution, measure performance characteristics,
and ensure scalability under load.
"""

import pytest
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple
import statistics

from src.workflow import SupervisorWorkflow
from src.workflow_simple import SimpleWorkflow
from src.config import AgentConfig
from src.models.data_models import TaskStatus, Task, TaskResult


class TestConcurrentExecution:
    """Performance tests for concurrent subagent execution."""

    @pytest.fixture
    def performance_config(self):
        """Create a configuration optimized for performance testing."""
        config = Mock(spec=AgentConfig)
        config.llm_model = "gpt-4"
        config.openai_api_key = "test-key"
        config.max_iterations = 20
        config.max_subagents = 10  # Allow more concurrent subagents
        config.tool_timeout = 60
        config.langsmith_project = "performance-test"
        config.enable_tracing = False  # Disable tracing for better performance
        config.tavily_api_key = None
        config.firecrawl_api_key = None
        config.virtual_fs_root = "/virtual"
        config.max_file_size = 10485760  # 10MB for performance tests
        return config

    @pytest.fixture
    def workflow_pool(self, performance_config):
        """Create a pool of workflow instances for concurrent testing."""
        workflows = []
        for i in range(5):  # Create 5 workflow instances
            workflow = SimpleWorkflow(performance_config)
            workflows.append(workflow)
        return workflows

    def test_concurrent_task_execution_performance(self, performance_config):
        """Test performance of concurrent task execution within a single workflow."""
        workflow = SimpleWorkflow(performance_config)
        objective = "Execute multiple concurrent tasks for performance testing"

        with patch.object(workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate concurrent task execution with varying execution times
            def simulate_concurrent_task(*args, **kwargs):
                # Simulate different task execution times
                task_id = args[0] if args else "concurrent_task"
                execution_time = 5.0 + (hash(task_id) % 10)  # 5-15 seconds

                # Simulate actual work with sleep
                time.sleep(0.1)  # Small delay to simulate processing

                return {
                    "task_id": task_id,
                    "status": TaskStatus.COMPLETED.value,
                    "output": f"Concurrent task {task_id} completed",
                    "artifacts": [f"result_{task_id}.txt"],
                    "execution_time": execution_time,
                    "tool_usage": {"execute_code": 2},
                }

            mock_task_tool.side_effect = simulate_concurrent_task

            # Measure execution time
            start_time = time.time()
            result = workflow.run(objective)
            total_time = time.time() - start_time

            # Validate performance characteristics
            assert result.success is True
            assert result.completed_tasks >= 1
            assert total_time < 30.0  # Should complete within reasonable time

            # Performance metrics
            print(f"Concurrent execution completed in {total_time:.2f}s")
            print(f"Tasks completed: {result.completed_tasks}")
            print(f"Iterations: {result.iterations}")

    def test_multiple_workflow_instances_concurrent_execution(self, workflow_pool):
        """Test concurrent execution of multiple workflow instances."""
        objectives = [
            "Research task for workflow 1",
            "Analysis task for workflow 2",
            "Testing task for workflow 3",
            "Documentation task for workflow 4",
            "Security audit for workflow 5",
        ]

        def execute_workflow(workflow_objective_pair):
            workflow, objective = workflow_objective_pair

            with patch.object(workflow.supervisor, "task_tool") as mock_task_tool:
                mock_task_tool.return_value = {
                    "task_id": f"concurrent_{hash(objective) % 1000}",
                    "status": TaskStatus.COMPLETED.value,
                    "output": f"Completed: {objective}",
                    "artifacts": [f"result_{hash(objective) % 1000}.txt"],
                    "execution_time": 8.0,
                }

                start_time = time.time()
                result = workflow.run(objective)
                execution_time = time.time() - start_time

                return {
                    "objective": objective,
                    "result": result,
                    "execution_time": execution_time,
                }

        # Execute workflows concurrently
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=5) as executor:
            workflow_objective_pairs = list(zip(workflow_pool, objectives))
            future_to_pair = {
                executor.submit(execute_workflow, pair): pair
                for pair in workflow_objective_pairs
            }

            results = []
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Workflow failed for {pair[1]}: {e}")

        total_concurrent_time = time.time() - start_time

        # Validate concurrent execution results
        assert len(results) >= 3  # At least 3 workflows should complete

        # All workflows should complete successfully
        for result_data in results:
            assert result_data["result"].success is True
            assert result_data["execution_time"] < 20.0  # Individual workflow time

        # Concurrent execution should be faster than sequential
        individual_times = [r["execution_time"] for r in results]
        sequential_time_estimate = sum(individual_times)

        print(f"Concurrent execution time: {total_concurrent_time:.2f}s")
        print(f"Sequential time estimate: {sequential_time_estimate:.2f}s")
        print(
            f"Concurrency speedup: {sequential_time_estimate / total_concurrent_time:.2f}x"
        )

        # Concurrent execution should provide some speedup
        assert total_concurrent_time < sequential_time_estimate * 0.8

    def test_workflow_scalability_under_load(self, performance_config):
        """Test workflow scalability with increasing load."""
        workflow = SimpleWorkflow(performance_config)

        # Test with different numbers of simulated concurrent tasks
        load_levels = [1, 3, 5, 8, 10]
        performance_metrics = []

        for num_tasks in load_levels:
            objective = f"Handle {num_tasks} concurrent tasks for scalability testing"

            with patch.object(workflow.supervisor, "task_tool") as mock_task_tool:
                # Simulate multiple tasks with realistic execution times
                task_results = []
                for i in range(num_tasks):
                    task_results.append(
                        {
                            "task_id": f"scale_task_{i}",
                            "status": TaskStatus.COMPLETED.value,
                            "output": f"Scalability task {i} completed",
                            "artifacts": [f"scale_result_{i}.txt"],
                            "execution_time": 3.0
                            + (i * 0.5),  # Increasing execution time
                            "tool_usage": {"execute_code": 1},
                        }
                    )

                mock_task_tool.side_effect = task_results

                # Measure performance at this load level
                start_time = time.time()
                result = workflow.run(objective)
                execution_time = time.time() - start_time

                performance_metrics.append(
                    {
                        "num_tasks": num_tasks,
                        "execution_time": execution_time,
                        "completed_tasks": result.completed_tasks,
                        "success": result.success,
                        "iterations": result.iterations,
                    }
                )

            # Reset workflow for next test
            workflow.reset()

        # Analyze scalability characteristics
        print("\nScalability Test Results:")
        print("Tasks | Time (s) | Completed | Success | Iterations")
        print("-" * 50)

        for metrics in performance_metrics:
            print(
                f"{metrics['num_tasks']:5d} | {metrics['execution_time']:8.2f} | "
                f"{metrics['completed_tasks']:9d} | {metrics['success']:7} | "
                f"{metrics['iterations']:10d}"
            )

        # Validate scalability characteristics
        # All load levels should complete successfully
        for metrics in performance_metrics:
            assert metrics["success"] is True
            assert metrics["completed_tasks"] >= 1

        # Execution time should scale reasonably (not exponentially)
        execution_times = [m["execution_time"] for m in performance_metrics]

        # Time should increase with load but not exponentially
        time_ratios = []
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i - 1]
            time_ratios.append(ratio)

        # Average time ratio should be reasonable (< 3x per load increase)
        avg_time_ratio = statistics.mean(time_ratios)
        assert avg_time_ratio < 3.0, (
            f"Performance degrades too quickly: {avg_time_ratio:.2f}x per load increase"
        )

    def test_memory_usage_under_concurrent_load(self, performance_config):
        """Test memory usage characteristics under concurrent load."""
        workflow = SimpleWorkflow(performance_config)
        objective = "Test memory usage under concurrent load"

        # Create a large number of tasks with substantial data
        num_tasks = 15
        large_data_size = 1000  # Characters per artifact

        with patch.object(workflow.supervisor, "task_tool") as mock_task_tool:

            def create_memory_intensive_task(*args, **kwargs):
                task_id = args[0] if args else "memory_task"

                # Create substantial output data
                large_output = "x" * large_data_size
                large_artifacts = [
                    f"large_artifact_{i}_{task_id}.txt" for i in range(5)
                ]

                # Simulate file creation in VFS
                for artifact in large_artifacts:
                    workflow.vfs.write_file(artifact, large_output)

                return {
                    "task_id": task_id,
                    "status": TaskStatus.COMPLETED.value,
                    "output": large_output,
                    "artifacts": large_artifacts,
                    "execution_time": 5.0,
                    "tool_usage": {"execute_code": 3},
                }

            mock_task_tool.side_effect = create_memory_intensive_task

            # Measure memory usage (approximate via file system size)
            initial_files = len(workflow.list_files())
            initial_vfs_size = sum(
                len(content) for content in workflow.vfs.files.values()
            )

            start_time = time.time()
            result = workflow.run(objective)
            execution_time = time.time() - start_time

            final_files = len(workflow.list_files())
            final_vfs_size = sum(
                len(content) for content in workflow.vfs.files.values()
            )

            # Validate memory usage characteristics
            assert result.success is True

            # Memory usage should be reasonable
            memory_growth = final_vfs_size - initial_vfs_size
            files_created = final_files - initial_files

            print(f"Memory usage test results:")
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Files created: {files_created}")
            print(f"  Memory growth: {memory_growth:,} bytes")
            print(
                f"  Average file size: {memory_growth / max(files_created, 1):,.0f} bytes"
            )

            # Memory growth should be proportional to work done
            expected_memory_growth = (
                num_tasks * 5 * large_data_size
            )  # 5 artifacts per task
            assert memory_growth <= expected_memory_growth * 2  # Allow some overhead

    def test_error_handling_under_concurrent_load(self, performance_config):
        """Test error handling and recovery under concurrent load."""
        workflow = SimpleWorkflow(performance_config)
        objective = "Test error handling under concurrent load"

        with patch.object(workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate mixed success/failure scenarios
            task_results = [
                # Successful tasks
                {
                    "task_id": "success_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Successful task 1",
                    "artifacts": ["success_1.txt"],
                    "execution_time": 5.0,
                },
                {
                    "task_id": "success_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Successful task 2",
                    "artifacts": ["success_2.txt"],
                    "execution_time": 6.0,
                },
                # Error scenarios
                {"error": "Simulated network timeout"},
                {"error": "Tool execution failure"},
                {"error": "Resource temporarily unavailable"},
                # Recovery tasks
                {
                    "task_id": "recovery_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Recovery task 1 completed",
                    "artifacts": ["recovery_1.txt"],
                    "execution_time": 7.0,
                },
                {
                    "task_id": "recovery_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Recovery task 2 completed",
                    "artifacts": ["recovery_2.txt"],
                    "execution_time": 8.0,
                },
            ]

            mock_task_tool.side_effect = task_results

            start_time = time.time()
            result = workflow.run(objective)
            execution_time = time.time() - start_time

            # Validate error handling under load
            assert result.success is True  # Should recover from errors
            assert (
                result.completed_tasks >= 2
            )  # Should complete some tasks despite errors
            assert execution_time < 60.0  # Should not hang due to errors

            print(f"Error handling test results:")
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Tasks completed: {result.completed_tasks}")
            print(f"  Iterations: {result.iterations}")

    def test_throughput_measurement(self, performance_config):
        """Test and measure workflow throughput characteristics."""
        workflow = SimpleWorkflow(performance_config)

        # Test different objective complexities
        test_cases = [
            {
                "name": "Simple",
                "objective": "Create a simple report",
                "expected_tasks": 2,
            },
            {
                "name": "Medium",
                "objective": "Analyze data and create comprehensive report with visualizations",
                "expected_tasks": 5,
            },
            {
                "name": "Complex",
                "objective": "Conduct security audit, performance analysis, and create detailed recommendations with implementation guide",
                "expected_tasks": 8,
            },
        ]

        throughput_results = []

        for test_case in test_cases:
            with patch.object(workflow.supervisor, "task_tool") as mock_task_tool:
                # Create appropriate number of tasks for complexity
                task_results = []
                for i in range(test_case["expected_tasks"]):
                    task_results.append(
                        {
                            "task_id": f"{test_case['name'].lower()}_task_{i}",
                            "status": TaskStatus.COMPLETED.value,
                            "output": f"Task {i} for {test_case['name']} objective completed",
                            "artifacts": [
                                f"{test_case['name'].lower()}_result_{i}.txt"
                            ],
                            "execution_time": 4.0 + (i * 0.5),
                            "tool_usage": {"execute_code": 2},
                        }
                    )

                mock_task_tool.side_effect = task_results

                # Measure throughput
                start_time = time.time()
                result = workflow.run(test_case["objective"])
                execution_time = time.time() - start_time

                # Calculate throughput metrics
                tasks_per_second = result.completed_tasks / execution_time
                iterations_per_second = result.iterations / execution_time

                throughput_results.append(
                    {
                        "complexity": test_case["name"],
                        "execution_time": execution_time,
                        "completed_tasks": result.completed_tasks,
                        "iterations": result.iterations,
                        "tasks_per_second": tasks_per_second,
                        "iterations_per_second": iterations_per_second,
                    }
                )

            # Reset for next test
            workflow.reset()

        # Display throughput results
        print("\nThroughput Test Results:")
        print("Complexity | Time (s) | Tasks | Iterations | Tasks/s | Iter/s")
        print("-" * 65)

        for result in throughput_results:
            print(
                f"{result['complexity']:10s} | {result['execution_time']:8.2f} | "
                f"{result['completed_tasks']:5d} | {result['iterations']:10d} | "
                f"{result['tasks_per_second']:7.2f} | {result['iterations_per_second']:6.2f}"
            )

        # Validate throughput characteristics
        for result in throughput_results:
            assert result["tasks_per_second"] > 0.1  # Minimum throughput
            assert result["iterations_per_second"] > 0.1  # Minimum iteration rate

    @pytest.mark.asyncio
    async def test_async_concurrent_execution_performance(self, performance_config):
        """Test asynchronous concurrent execution performance."""
        with patch("src.workflow.ChatOpenAI") as mock_llm:
            mock_llm.return_value = Mock()
            workflow = SupervisorWorkflow(performance_config)

        objectives = [
            "Async task 1: Research and analysis",
            "Async task 2: Code review and testing",
            "Async task 3: Documentation generation",
        ]

        async def execute_async_workflow(objective):
            with patch.object(workflow.supervisor, "task_tool") as mock_task_tool:
                mock_task_tool.return_value = {
                    "task_id": f"async_{hash(objective) % 1000}",
                    "status": TaskStatus.COMPLETED.value,
                    "output": f"Async completed: {objective}",
                    "artifacts": [f"async_result_{hash(objective) % 1000}.txt"],
                    "execution_time": 6.0,
                }

                # Mock the graph's async invoke
                with patch.object(workflow.graph, "ainvoke") as mock_ainvoke:
                    mock_final_state = {
                        "user_objective": objective,
                        "final_result": f"Async workflow completed: {objective}",
                        "iteration_count": 2,
                        "completed_tasks": [
                            {
                                "task_id": f"async_{hash(objective) % 1000}",
                                "status": "completed",
                            }
                        ],
                        "artifacts": {"async_result": "async_data"},
                        "file_system": {},
                    }
                    mock_ainvoke.return_value = mock_final_state

                    start_time = time.time()
                    result = await workflow.arun(objective)
                    execution_time = time.time() - start_time

                    return {
                        "objective": objective,
                        "result": result,
                        "execution_time": execution_time,
                    }

        # Execute async workflows concurrently
        start_time = time.time()

        tasks = [execute_async_workflow(obj) for obj in objectives]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_async_time = time.time() - start_time

        # Validate async concurrent execution
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 2  # At least 2 should complete

        # Async execution should be efficient
        individual_times = [r["execution_time"] for r in successful_results]
        sequential_estimate = sum(individual_times)

        print(f"Async concurrent execution time: {total_async_time:.2f}s")
        print(f"Sequential time estimate: {sequential_estimate:.2f}s")
        print(f"Async speedup: {sequential_estimate / total_async_time:.2f}x")

        # Async should provide good concurrency
        assert total_async_time < sequential_estimate * 0.7

    def test_resource_cleanup_after_concurrent_execution(self, performance_config):
        """Test that resources are properly cleaned up after concurrent execution."""
        workflow = SimpleWorkflow(performance_config)

        # Record initial state
        initial_files = len(workflow.list_files())
        initial_stats = workflow.get_workflow_stats()

        # Execute multiple workflows to create resources
        objectives = [
            "Create temporary files and data",
            "Generate analysis reports",
            "Process and store results",
        ]

        for objective in objectives:
            with patch.object(workflow.supervisor, "task_tool") as mock_task_tool:
                mock_task_tool.return_value = {
                    "task_id": f"cleanup_task_{hash(objective) % 1000}",
                    "status": TaskStatus.COMPLETED.value,
                    "output": f"Cleanup test: {objective}",
                    "artifacts": [f"cleanup_result_{hash(objective) % 1000}.txt"],
                    "execution_time": 5.0,
                }

                result = workflow.run(objective)
                assert result.success is True

            # Reset workflow (should clean up resources)
            workflow.reset()

        # Check resource cleanup
        final_files = len(workflow.list_files())
        final_stats = workflow.get_workflow_stats()

        print(f"Resource cleanup test:")
        print(f"  Initial files: {initial_files}")
        print(f"  Final files: {final_files}")
        print(f"  Files difference: {final_files - initial_files}")

        # Resources should be cleaned up properly
        # Allow some files to remain (e.g., configuration files)
        assert final_files <= initial_files + 5  # Small tolerance for persistent files

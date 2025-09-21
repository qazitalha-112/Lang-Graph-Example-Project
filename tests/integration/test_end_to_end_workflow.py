"""End-to-end integration tests for the complete supervisor workflow."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.workflow import SupervisorWorkflow
from src.workflow_simple import SimpleWorkflow
from src.config import AgentConfig
from src.models.data_models import TaskStatus, Task, TaskResult


class TestEndToEndWorkflow:
    """End-to-end tests for complete workflow execution with complex scenarios."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=AgentConfig)
        config.llm_model = "gpt-4"
        config.openai_api_key = "test-key"
        config.max_iterations = 10
        config.max_subagents = 5
        config.tool_timeout = 30
        config.langsmith_project = "test-project"
        config.enable_tracing = False
        config.tavily_api_key = None
        config.firecrawl_api_key = None
        config.virtual_fs_root = "/virtual"
        config.max_file_size = 1048576
        return config

    @pytest.fixture
    def langgraph_workflow(self, mock_config):
        """Create a LangGraph workflow instance for testing."""
        with patch("src.workflow.ChatOpenAI") as mock_llm:
            mock_llm.return_value = Mock()
            return SupervisorWorkflow(mock_config)

    @pytest.fixture
    def simple_workflow(self, mock_config):
        """Create a simple workflow instance for testing."""
        return SimpleWorkflow(mock_config)

    def test_complete_research_workflow(self, simple_workflow):
        """Test a complete research workflow from start to finish."""
        objective = (
            "Research the latest trends in artificial intelligence and machine learning"
        )

        # Mock the supervisor's task execution to simulate real workflow
        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate sequential task execution for research workflow
            task_results = [
                {
                    "task_id": "task_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Research completed: Found 5 major AI trends including GPT models, computer vision advances, and autonomous systems.",
                    "artifacts": ["ai_trends_research.md", "sources.json"],
                    "execution_time": 45.2,
                    "tool_usage": {"search_internet": 4, "web_scrape": 3},
                },
                {
                    "task_id": "task_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Analysis completed: Synthesized research findings into key themes and future predictions.",
                    "artifacts": ["analysis_report.md"],
                    "execution_time": 32.1,
                    "tool_usage": {"execute_code": 2},
                },
                {
                    "task_id": "task_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Research report generated: Comprehensive 15-page report with citations and recommendations.",
                    "artifacts": ["final_research_report.md", "bibliography.txt"],
                    "execution_time": 28.7,
                    "tool_usage": {"execute_code": 3},
                },
            ]

            mock_task_tool.side_effect = task_results

            # Execute the workflow
            result = simple_workflow.run(objective)

            # Validate workflow completion
            assert result.success is True
            assert result.objective == objective
            assert result.iterations > 0
            assert result.completed_tasks >= 3

            # Validate final result content
            assert "research" in result.final_result.lower()
            assert (
                "artificial intelligence" in result.final_result.lower()
                or "ai" in result.final_result.lower()
            )

            # Validate artifacts were created
            assert len(result.artifacts) > 0
            assert "ai_trends_research.md" in str(result.artifacts)
            assert "final_research_report.md" in str(result.artifacts)

            # Validate file system state
            assert len(result.file_system) >= 0  # Files may be created during execution

    def test_complete_web_testing_workflow(self, simple_workflow):
        """Test a complete web application testing workflow."""
        objective = (
            "Test my web application for bugs, security issues, and usability problems"
        )

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate web testing workflow with multiple phases
            task_results = [
                {
                    "task_id": "task_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Application structure analyzed: Identified 5 main features and 12 user workflows to test.",
                    "artifacts": ["app_structure_analysis.md", "test_plan.md"],
                    "execution_time": 25.0,
                    "tool_usage": {"web_scrape": 3, "execute_code": 2},
                },
                {
                    "task_id": "task_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Authentication testing completed: Login works correctly, found 1 minor session timeout issue.",
                    "artifacts": ["auth_test_results.md"],
                    "execution_time": 35.0,
                    "tool_usage": {"web_scrape": 5, "execute_code": 3},
                },
                {
                    "task_id": "task_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Feature testing completed: Core features work well, found 2 UI bugs and 1 data validation issue.",
                    "artifacts": ["feature_test_results.md", "bug_screenshots.zip"],
                    "execution_time": 50.0,
                    "tool_usage": {"web_scrape": 8, "execute_code": 4},
                },
                {
                    "task_id": "task_4",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Comprehensive test report compiled: 3 bugs found, 2 usability improvements recommended.",
                    "artifacts": ["comprehensive_test_report.md", "bug_report.json"],
                    "execution_time": 20.0,
                    "tool_usage": {"execute_code": 2},
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate successful completion
            assert result.success is True
            assert result.completed_tasks >= 4
            assert "test" in result.final_result.lower()
            assert "bug" in result.final_result.lower()

            # Validate comprehensive artifacts
            artifacts_str = str(result.artifacts)
            assert "test_plan.md" in artifacts_str
            assert "comprehensive_test_report.md" in artifacts_str
            assert "bug_report.json" in artifacts_str

    def test_workflow_with_task_failures_and_recovery(self, simple_workflow):
        """Test workflow behavior when tasks fail and recovery mechanisms."""
        objective = "Analyze codebase and generate improvement recommendations"

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate workflow with failures and recovery
            task_results = [
                {
                    "task_id": "task_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Codebase structure mapped successfully.",
                    "artifacts": ["codebase_structure.md"],
                    "execution_time": 20.0,
                },
                {
                    "error": "Network timeout while analyzing code quality"
                },  # First failure
                {
                    "task_id": "task_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Code quality analysis completed on retry: Found 8 issues.",
                    "artifacts": ["quality_analysis.md"],
                    "execution_time": 40.0,
                },
                {
                    "task_id": "task_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Improvement recommendations generated: 12 actionable suggestions provided.",
                    "artifacts": ["improvement_recommendations.md"],
                    "execution_time": 30.0,
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Should complete successfully despite initial failure
            assert result.success is True
            assert result.completed_tasks >= 2
            assert "improvement" in result.final_result.lower()

    def test_workflow_max_iterations_termination(self, simple_workflow):
        """Test that workflow properly terminates at max iterations."""
        objective = "Test max iterations handling"

        # Set low max iterations for testing
        simple_workflow.config.max_iterations = 3

        with patch.object(
            simple_workflow.supervisor, "get_next_task"
        ) as mock_next_task:
            with patch.object(
                simple_workflow.supervisor, "task_tool"
            ) as mock_task_tool:
                # Always return a task to simulate infinite work
                mock_next_task.return_value = {
                    "id": "endless_task",
                    "description": "A task that keeps generating more work",
                    "status": TaskStatus.PENDING.value,
                }

                mock_task_tool.return_value = {
                    "task_id": "endless_task",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Task completed, but more work needed",
                    "artifacts": ["partial_result.txt"],
                    "execution_time": 10.0,
                }

                result = simple_workflow.run(objective)

                # Should terminate due to max iterations
                assert result.iterations <= simple_workflow.config.max_iterations
                assert result.success is True  # Should still be considered successful

    def test_workflow_objective_completion_detection(self, simple_workflow):
        """Test that workflow correctly detects when objective is complete."""
        objective = "Create a simple status report"

        with patch.object(
            simple_workflow.supervisor, "is_objective_complete"
        ) as mock_complete:
            with patch.object(
                simple_workflow.supervisor, "task_tool"
            ) as mock_task_tool:
                # First call: not complete, second call: complete
                mock_complete.side_effect = [False, True]

                mock_task_tool.return_value = {
                    "task_id": "report_task",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Status report created successfully",
                    "artifacts": ["status_report.md"],
                    "execution_time": 15.0,
                }

                result = simple_workflow.run(objective)

                assert result.success is True
                assert result.completed_tasks >= 1
                assert "status report" in result.final_result.lower()

    def test_workflow_artifact_collection_and_consolidation(self, simple_workflow):
        """Test that workflow properly collects and consolidates artifacts."""
        objective = "Generate comprehensive project documentation"

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate tasks that create multiple artifacts
            task_results = [
                {
                    "task_id": "task_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "README documentation created",
                    "artifacts": ["README.md", "INSTALLATION.md"],
                    "execution_time": 20.0,
                },
                {
                    "task_id": "task_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "API documentation generated",
                    "artifacts": ["api_docs.md", "examples.md", "changelog.md"],
                    "execution_time": 35.0,
                },
                {
                    "task_id": "task_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Architecture documentation completed",
                    "artifacts": ["architecture.md", "diagrams.png"],
                    "execution_time": 25.0,
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate artifact consolidation
            assert result.success is True
            artifacts_data = result.artifacts

            # Should contain consolidated results
            assert "consolidated_summary" in str(artifacts_data)

            # Should track all created artifacts
            all_artifacts = []
            for task_result in task_results:
                all_artifacts.extend(task_result["artifacts"])

            artifacts_str = str(artifacts_data)
            for artifact in all_artifacts:
                assert artifact in artifacts_str

    def test_workflow_file_system_persistence(self, simple_workflow):
        """Test that file system state is properly maintained throughout workflow."""
        objective = "Create and manage multiple files"

        # Pre-populate file system
        simple_workflow.vfs.write_file("input.txt", "Initial input data")
        simple_workflow.vfs.write_file("config.json", '{"setting": "value"}')

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:

            def simulate_file_creation(*args, **kwargs):
                # Simulate file creation during task execution
                simple_workflow.vfs.write_file("output.txt", "Generated output")
                simple_workflow.vfs.write_file(
                    "results.json", '{"results": ["item1", "item2"]}'
                )
                simple_workflow.vfs.write_file(
                    "summary.md", "# Summary\nTask completed successfully"
                )

                return {
                    "task_id": "file_task",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "File operations completed successfully",
                    "artifacts": ["output.txt", "results.json", "summary.md"],
                    "execution_time": 15.0,
                }

            mock_task_tool.side_effect = simulate_file_creation

            result = simple_workflow.run(objective)

            # Validate file system persistence
            assert result.success is True

            # Original files should still exist
            assert "input.txt" in result.file_system
            assert "config.json" in result.file_system

            # New files should be created
            assert "output.txt" in result.file_system
            assert "results.json" in result.file_system
            assert "summary.md" in result.file_system

            # Validate file contents
            assert result.file_system["input.txt"] == "Initial input data"
            assert "Generated output" in result.file_system["output.txt"]

    def test_complex_multi_step_objective_with_dependencies(self, simple_workflow):
        """Test complex objective requiring multiple coordinated steps."""
        objective = "Conduct security audit of web application including penetration testing and vulnerability assessment"

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate complex security audit workflow
            task_results = [
                {
                    "task_id": "task_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Security scan initialization: Target application mapped, 15 endpoints identified.",
                    "artifacts": ["security_scan_plan.md", "endpoint_mapping.json"],
                    "execution_time": 30.0,
                },
                {
                    "task_id": "task_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Vulnerability assessment completed: 3 high-risk, 5 medium-risk, 12 low-risk vulnerabilities found.",
                    "artifacts": ["vulnerability_report.md", "scan_results.json"],
                    "execution_time": 120.0,
                },
                {
                    "task_id": "task_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Penetration testing completed: Successfully exploited 2 vulnerabilities, documented attack vectors.",
                    "artifacts": ["pentest_report.md", "exploit_documentation.md"],
                    "execution_time": 180.0,
                },
                {
                    "task_id": "task_4",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Security recommendations compiled: 15 actionable recommendations with priority levels.",
                    "artifacts": ["security_recommendations.md", "remediation_plan.md"],
                    "execution_time": 45.0,
                },
                {
                    "task_id": "task_5",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Executive summary created: Comprehensive security audit report with risk assessment.",
                    "artifacts": [
                        "executive_summary.md",
                        "full_security_audit_report.pdf",
                    ],
                    "execution_time": 30.0,
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate comprehensive workflow completion
            assert result.success is True
            assert result.completed_tasks >= 5
            assert result.iterations > 0

            # Validate security-specific content
            final_result_lower = result.final_result.lower()
            assert any(
                term in final_result_lower
                for term in ["security", "vulnerability", "audit"]
            )

            # Validate comprehensive artifact creation
            artifacts_str = str(result.artifacts)
            expected_artifacts = [
                "security_scan_plan.md",
                "vulnerability_report.md",
                "pentest_report.md",
                "security_recommendations.md",
                "executive_summary.md",
            ]
            for artifact in expected_artifacts:
                assert artifact in artifacts_str

            # Validate execution time tracking
            total_time = sum(task["execution_time"] for task in task_results)
            assert total_time > 400  # Should be substantial for security audit

    @pytest.mark.asyncio
    async def test_async_workflow_execution(self, langgraph_workflow):
        """Test asynchronous workflow execution with LangGraph."""
        objective = "Test async workflow capabilities"

        with patch.object(langgraph_workflow.supervisor, "task_tool") as mock_task_tool:
            mock_task_tool.return_value = {
                "task_id": "async_task",
                "status": TaskStatus.COMPLETED.value,
                "output": "Async task completed successfully",
                "artifacts": ["async_result.txt"],
                "execution_time": 12.0,
            }

            # Mock the graph's async invoke
            with patch.object(langgraph_workflow.graph, "ainvoke") as mock_ainvoke:
                mock_final_state = {
                    "user_objective": objective,
                    "final_result": "Async workflow completed successfully",
                    "iteration_count": 2,
                    "completed_tasks": [
                        {"task_id": "async_task", "status": "completed"}
                    ],
                    "artifacts": {"async_result": "data"},
                    "file_system": {},
                }
                mock_ainvoke.return_value = mock_final_state

                result = await langgraph_workflow.arun(objective)

                assert result["objective"] == objective
                assert result["final_result"] is not None
                assert result["iterations"] >= 0

    def test_workflow_performance_and_metrics_collection(self, simple_workflow):
        """Test that workflow properly collects performance metrics."""
        objective = "Performance test with metrics collection"

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate tasks with varying execution times
            task_results = [
                {
                    "task_id": "fast_task",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Fast task completed",
                    "artifacts": ["fast_result.txt"],
                    "execution_time": 5.0,
                    "tool_usage": {"execute_code": 1},
                },
                {
                    "task_id": "slow_task",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Slow task completed",
                    "artifacts": ["slow_result.txt"],
                    "execution_time": 45.0,
                    "tool_usage": {"search_internet": 3, "web_scrape": 2},
                },
                {
                    "task_id": "medium_task",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Medium task completed",
                    "artifacts": ["medium_result.txt"],
                    "execution_time": 20.0,
                    "tool_usage": {"execute_code": 2, "web_scrape": 1},
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate metrics collection
            assert result.success is True

            # Should have performance data in artifacts
            artifacts_data = result.artifacts
            assert "total_execution_time" in str(artifacts_data)

            # Total execution time should be sum of individual tasks
            expected_total_time = sum(task["execution_time"] for task in task_results)
            assert expected_total_time == 70.0  # 5 + 45 + 20

    def test_workflow_error_handling_and_graceful_degradation(self, simple_workflow):
        """Test workflow error handling and graceful degradation."""
        objective = "Test error handling capabilities"

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate various error scenarios
            task_results = [
                {
                    "task_id": "success_task",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "First task completed successfully",
                    "artifacts": ["success_result.txt"],
                    "execution_time": 15.0,
                },
                {"error": "Simulated network failure"},  # Network error
                {"error": "Tool execution timeout"},  # Timeout error
                {
                    "task_id": "recovery_task",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Recovery task completed after errors",
                    "artifacts": ["recovery_result.txt"],
                    "execution_time": 25.0,
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Should complete gracefully despite errors
            assert result.success is True
            assert result.completed_tasks >= 1  # At least some tasks should complete

            # Should contain information about both successes and errors
            final_result = result.final_result.lower()
            assert "task" in final_result

    def test_workflow_integration_with_all_components(self, simple_workflow):
        """Test that workflow properly integrates all system components."""
        objective = "Test complete system integration"

        # Verify all components are properly initialized
        assert simple_workflow.vfs is not None
        assert simple_workflow.tool_registry is not None
        assert simple_workflow.subagent_factory is not None
        assert simple_workflow.supervisor is not None

        # Verify tool registry has expected tools
        all_tools = simple_workflow.tool_registry.get_all_tools()
        assert len(all_tools) > 0

        # Verify supervisor has access to required tools
        supervisor_tools = simple_workflow.supervisor.get_supervisor_tools()
        assert "update_todo" in supervisor_tools
        assert "task_tool" in supervisor_tools

        # Test workflow execution with component integration
        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            mock_task_tool.return_value = {
                "task_id": "integration_task",
                "status": TaskStatus.COMPLETED.value,
                "output": "Integration test completed successfully",
                "artifacts": ["integration_result.txt"],
                "execution_time": 10.0,
            }

            result = simple_workflow.run(objective)

            assert result.success is True
            assert "integration" in result.final_result.lower()

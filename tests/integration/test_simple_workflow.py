"""Integration tests for the simplified workflow implementation."""

import pytest
from unittest.mock import Mock

from src.workflow_simple import SimpleWorkflow, WorkflowResult
from src.config import AgentConfig


class TestSimpleWorkflow:
    """Integration tests for the simplified workflow."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=AgentConfig)
        config.llm_model = "gpt-4"
        config.openai_api_key = "test-key"
        config.max_iterations = 5
        config.max_subagents = 3
        config.tool_timeout = 30
        config.langsmith_project = "test-project"
        config.enable_tracing = False
        config.tavily_api_key = None
        config.firecrawl_api_key = None
        config.virtual_fs_root = "/virtual"
        config.max_file_size = 1048576
        return config

    @pytest.fixture
    def workflow(self, mock_config):
        """Create a workflow instance for testing."""
        return SimpleWorkflow(mock_config)

    def test_workflow_initialization(self, workflow):
        """Test that workflow initializes correctly with all components."""
        assert workflow.config is not None
        assert workflow.vfs is not None
        assert workflow.tool_registry is not None
        assert workflow.subagent_factory is not None
        assert workflow.supervisor is not None

    def test_simple_objective_execution(self, workflow):
        """Test execution of a simple objective."""
        objective = "Create a simple test report"

        result = workflow.run(objective)

        assert isinstance(result, WorkflowResult)
        assert result.objective == objective
        assert result.success == True
        assert result.iterations > 0
        assert result.final_result is not None
        assert isinstance(result.artifacts, dict)
        assert isinstance(result.file_system, dict)

    def test_research_objective_execution(self, workflow):
        """Test execution of a research-based objective."""
        objective = "Research the latest trends in AI"

        result = workflow.run(objective)

        assert result.objective == objective
        assert result.success == True
        assert (
            "research" in result.final_result.lower()
            or "ai" in result.final_result.lower()
        )

    def test_web_testing_objective_execution(self, workflow):
        """Test execution of a web testing objective."""
        objective = "Test my web application for bugs"

        result = workflow.run(objective)

        assert result.objective == objective
        assert result.success == True
        assert result.completed_tasks >= 1
        assert "test" in result.final_result.lower()

    def test_code_analysis_objective_execution(self, workflow):
        """Test execution of a code analysis objective."""
        objective = "Analyze this Python codebase for improvements"

        result = workflow.run(objective)

        assert result.objective == objective
        assert result.success == True
        assert "analy" in result.final_result.lower()

    def test_workflow_max_iterations(self, workflow):
        """Test that workflow respects max iterations limit."""
        objective = "Test max iterations handling"

        result = workflow.run(objective)

        assert result.iterations <= workflow.config.max_iterations

    def test_workflow_stats(self, workflow):
        """Test workflow statistics collection."""
        objective = "Test workflow statistics"

        # Run workflow
        result = workflow.run(objective)

        # Get stats
        stats = workflow.get_workflow_stats()

        assert "objective" in stats
        assert "total_tasks" in stats
        assert "completed_tasks" in stats
        assert "pending_tasks" in stats
        assert "failed_tasks" in stats
        assert "iterations" in stats
        assert stats["objective"] == objective

    def test_task_details(self, workflow):
        """Test task details retrieval."""
        objective = "Test task details"

        # Run workflow
        result = workflow.run(objective)

        # Get task details
        tasks = workflow.get_task_details()

        assert isinstance(tasks, list)
        assert len(tasks) > 0

        # Check task structure
        for task in tasks:
            assert "id" in task
            assert "description" in task
            assert "status" in task

    def test_file_operations(self, workflow):
        """Test file operations in workflow."""
        objective = "Create and manage files"

        # Run workflow
        result = workflow.run(objective)

        # List files
        files = workflow.list_files()
        assert isinstance(files, list)

        # If files were created, test reading them
        if files:
            for file_path in files[:3]:  # Test first 3 files
                content = workflow.get_file_contents(file_path)
                assert content is not None or content == ""  # Could be empty file

    def test_workflow_reset(self, workflow):
        """Test workflow reset functionality."""
        objective = "Test workflow reset"

        # Run workflow
        result1 = workflow.run(objective)

        # Get initial stats
        stats1 = workflow.get_workflow_stats()

        # Reset workflow
        workflow.reset()

        # Check that state is cleared
        stats2 = workflow.get_workflow_stats()
        assert "error" in stats2 or stats2.get("total_tasks", 0) == 0

    def test_error_handling(self, workflow):
        """Test error handling in workflow execution."""
        # Test with empty objective
        result = workflow.run("")

        # Should still complete but may have minimal results
        assert isinstance(result, WorkflowResult)
        assert result.objective == ""

    def test_multiple_objectives(self, workflow):
        """Test running multiple objectives in sequence."""
        objectives = ["Create a simple report", "Analyze data", "Generate summary"]

        results = []
        for objective in objectives:
            workflow.reset()  # Reset between runs
            result = workflow.run(objective)
            results.append(result)

        # All should complete successfully
        for i, result in enumerate(results):
            assert result.objective == objectives[i]
            assert result.success == True

    def test_workflow_with_complex_objective(self, workflow):
        """Test workflow with a complex multi-step objective."""
        objective = "Analyze my codebase, identify issues, and create improvement recommendations"

        result = workflow.run(objective)

        assert result.objective == objective
        assert result.success == True
        assert result.completed_tasks >= 2  # Should break into multiple tasks
        assert "analy" in result.final_result.lower()
        assert "recommend" in result.final_result.lower()

    def test_workflow_result_structure(self, workflow):
        """Test that workflow result has correct structure."""
        objective = "Test result structure"

        result = workflow.run(objective)

        # Check all required fields are present
        assert hasattr(result, "objective")
        assert hasattr(result, "final_result")
        assert hasattr(result, "iterations")
        assert hasattr(result, "completed_tasks")
        assert hasattr(result, "artifacts")
        assert hasattr(result, "file_system")
        assert hasattr(result, "success")
        assert hasattr(result, "error_message")

        # Check types
        assert isinstance(result.objective, str)
        assert isinstance(result.final_result, str)
        assert isinstance(result.iterations, int)
        assert isinstance(result.completed_tasks, int)
        assert isinstance(result.artifacts, dict)
        assert isinstance(result.file_system, dict)
        assert isinstance(result.success, bool)

    def test_workflow_component_integration(self, workflow):
        """Test that all workflow components work together correctly."""
        objective = "Test component integration"

        # Check initial state
        assert workflow.supervisor.get_state() is None

        # Run workflow
        result = workflow.run(objective)

        # Check that components were used
        assert workflow.supervisor.get_state() is not None
        assert len(workflow.tool_registry.get_all_tools()) > 0

        # Check that virtual file system may have been used
        files = workflow.list_files()
        assert isinstance(files, list)  # May be empty, but should be a list

    def test_workflow_performance(self, workflow):
        """Test workflow performance characteristics."""
        import time

        objective = "Quick performance test"

        start_time = time.time()
        result = workflow.run(objective)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete reasonably quickly (under 10 seconds for simple test)
        assert execution_time < 10.0
        assert result.success == True

"""Unit tests for the Supervisor agent."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any

from src.agents.supervisor import SupervisorAgent
from src.models.data_models import Task, TaskResult, TaskStatus, AgentState
from src.config import AgentConfig
from src.tools.tool_registry import ToolRegistry
from src.models.virtual_file_system import VirtualFileSystem
from src.agents.subagent_factory import SubAgentFactory


class TestSupervisorAgent:
    """Test cases for SupervisorAgent class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=AgentConfig)
        config.max_iterations = 10
        config.max_subagents = 5
        config.tool_timeout = 30
        config.llm_model = "gpt-4"
        return config

    @pytest.fixture
    def mock_vfs(self):
        """Create a mock virtual file system."""
        vfs = Mock(spec=VirtualFileSystem)
        vfs.files = {}
        vfs.list_files.return_value = []
        return vfs

    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock tool registry."""
        registry = Mock(spec=ToolRegistry)
        registry.create_tool_whitelist.return_value = {
            "agent_id": "supervisor",
            "whitelisted_tools": ["read_file", "write_file", "edit_file"],
            "validation_results": {},
            "shared_tools_included": ["read_file", "write_file", "edit_file"],
            "assignable_tools_included": [],
        }
        registry.get_agent_tools.return_value = {
            "read_file": Mock(),
            "write_file": Mock(),
            "edit_file": Mock(),
        }
        return registry

    @pytest.fixture
    def mock_subagent_factory(self):
        """Create a mock subagent factory."""
        factory = Mock(spec=SubAgentFactory)
        return factory

    @pytest.fixture
    def supervisor(
        self, mock_config, mock_tool_registry, mock_vfs, mock_subagent_factory
    ):
        """Create a SupervisorAgent instance for testing."""
        return SupervisorAgent(
            config=mock_config,
            tool_registry=mock_tool_registry,
            vfs=mock_vfs,
            subagent_factory=mock_subagent_factory,
        )

    def test_initialization(
        self,
        supervisor,
        mock_config,
        mock_tool_registry,
        mock_vfs,
        mock_subagent_factory,
    ):
        """Test supervisor initialization."""
        assert supervisor.config == mock_config
        assert supervisor.tool_registry == mock_tool_registry
        assert supervisor.vfs == mock_vfs
        assert supervisor.subagent_factory == mock_subagent_factory
        assert supervisor.current_state is None

    def test_initialize_state(self, supervisor):
        """Test state initialization."""
        objective = "Test my web application"

        state = supervisor.initialize_state(objective)

        assert state["user_objective"] == objective
        assert state["todo_list"] == []
        assert state["completed_tasks"] == []
        assert state["current_task"] is None
        assert state["artifacts"] == {}
        assert state["subagent_logs"] == []
        assert state["iteration_count"] == 0
        assert state["final_result"] is None
        assert supervisor.current_state == state

    def test_decompose_objective_web_testing(self, supervisor):
        """Test objective decomposition for web testing scenario."""
        objective = "Test my web application for bugs"

        tasks = supervisor.decompose_objective(objective)

        assert len(tasks) == 4
        assert all(isinstance(task, Task) for task in tasks)

        # Check task sequence and dependencies
        assert "analyze" in tasks[0].description.lower()
        assert tasks[0].task_type == "analysis"
        assert tasks[0].dependencies == []

        assert "authentication" in tasks[1].description.lower()
        assert tasks[1].task_type == "web_testing"
        assert tasks[1].dependencies == [tasks[0].id]

        assert "core" in tasks[2].description.lower()
        assert tasks[2].task_type == "web_testing"
        assert tasks[2].dependencies == [tasks[1].id]

        assert "compile" in tasks[3].description.lower()
        assert tasks[3].task_type == "analysis"
        assert tasks[3].dependencies == [tasks[2].id]

    def test_decompose_objective_research(self, supervisor):
        """Test objective decomposition for research scenario."""
        objective = "Research the latest trends in AI"

        tasks = supervisor.decompose_objective(objective)

        assert len(tasks) == 3
        assert "research" in tasks[0].description.lower()
        assert tasks[0].task_type == "research"
        assert "search_internet" in tasks[0].required_tools

        assert "analyze" in tasks[1].description.lower()
        assert tasks[1].task_type == "analysis"
        assert tasks[1].dependencies == [tasks[0].id]

        assert "report" in tasks[2].description.lower()
        assert tasks[2].task_type == "analysis"
        assert tasks[2].dependencies == [tasks[1].id]

    def test_decompose_objective_code_analysis(self, supervisor):
        """Test objective decomposition for code analysis scenario."""
        objective = "Analyze this Python codebase for improvements"

        tasks = supervisor.decompose_objective(objective)

        assert len(tasks) == 3
        assert "examine" in tasks[0].description.lower()
        assert tasks[0].task_type == "analysis"

        assert "perform" in tasks[1].description.lower()
        assert tasks[1].task_type == "code_execution"
        assert tasks[1].dependencies == [tasks[0].id]

        assert "generate" in tasks[2].description.lower()
        assert tasks[2].task_type == "analysis"
        assert tasks[2].dependencies == [tasks[1].id]

    def test_decompose_objective_general(self, supervisor):
        """Test objective decomposition for general scenario."""
        objective = "Help me organize my files"

        tasks = supervisor.decompose_objective(objective)

        assert len(tasks) == 1
        assert tasks[0].task_type == "general"
        assert "execute_code" in tasks[0].required_tools

    def test_update_todo_tool_create(self, supervisor):
        """Test update_todo tool with create action."""
        supervisor.initialize_state("test objective")

        result = supervisor.update_todo_tool(
            action="create", objective="Test my application"
        )

        assert result["action"] == "create"
        assert result["tasks_created"] > 0
        assert "todo_list" in result
        assert supervisor.current_state["user_objective"] == "Test my application"
        assert len(supervisor.current_state["todo_list"]) > 0

    def test_update_todo_tool_add_task(self, supervisor):
        """Test update_todo tool with add_task action."""
        supervisor.initialize_state("test objective")

        task_data = {
            "description": "New test task",
            "task_type": "analysis",
            "priority": 1,
            "success_criteria": "Complete the task",
        }

        result = supervisor.update_todo_tool(action="add_task", task=task_data)

        assert result["action"] == "add_task"
        assert "task_added" in result
        assert len(supervisor.current_state["todo_list"]) == 1
        assert (
            supervisor.current_state["todo_list"][0]["description"] == "New test task"
        )

    def test_update_todo_tool_update_task(self, supervisor):
        """Test update_todo tool with update_task action."""
        supervisor.initialize_state("test objective")

        # Add a task first
        task_data = {
            "id": "test_task_1",
            "description": "Original description",
            "task_type": "analysis",
            "priority": 1,
        }
        supervisor.update_todo_tool(action="add_task", task=task_data)

        # Update the task
        updates = {"description": "Updated description", "priority": 2}
        result = supervisor.update_todo_tool(
            action="update_task", task_id="test_task_1", updates=updates
        )

        assert result["action"] == "update_task"
        assert result["task_updated"] == "test_task_1"
        assert (
            supervisor.current_state["todo_list"][0]["description"]
            == "Updated description"
        )
        assert supervisor.current_state["todo_list"][0]["priority"] == 2

    def test_update_todo_tool_remove_task(self, supervisor):
        """Test update_todo tool with remove_task action."""
        supervisor.initialize_state("test objective")

        # Add tasks first
        task1_data = {
            "id": "task_1",
            "description": "Task 1",
            "task_type": "analysis",
            "priority": 1,
        }
        task2_data = {
            "id": "task_2",
            "description": "Task 2",
            "task_type": "analysis",
            "priority": 2,
            "dependencies": ["task_1"],
        }

        supervisor.update_todo_tool(action="add_task", task=task1_data)
        supervisor.update_todo_tool(action="add_task", task=task2_data)

        # Remove task_1
        result = supervisor.update_todo_tool(action="remove_task", task_id="task_1")

        assert result["action"] == "remove_task"
        assert result["task_removed"] == "task_1"
        assert len(supervisor.current_state["todo_list"]) == 1
        assert supervisor.current_state["todo_list"][0]["id"] == "task_2"
        # Check that dependency was removed
        assert "task_1" not in supervisor.current_state["todo_list"][0]["dependencies"]

    def test_update_todo_tool_reorder(self, supervisor):
        """Test update_todo tool with reorder action."""
        supervisor.initialize_state("test objective")

        # Add tasks
        task1_data = {
            "id": "task_1",
            "description": "Task 1",
            "task_type": "analysis",
            "priority": 1,
        }
        task2_data = {
            "id": "task_2",
            "description": "Task 2",
            "task_type": "analysis",
            "priority": 2,
        }
        task3_data = {
            "id": "task_3",
            "description": "Task 3",
            "task_type": "analysis",
            "priority": 3,
        }

        supervisor.update_todo_tool(action="add_task", task=task1_data)
        supervisor.update_todo_tool(action="add_task", task=task2_data)
        supervisor.update_todo_tool(action="add_task", task=task3_data)

        # Reorder tasks
        new_order = ["task_3", "task_1", "task_2"]
        result = supervisor.update_todo_tool(action="reorder", task_order=new_order)

        assert result["action"] == "reorder"
        assert result["tasks_reordered"] == 3

        # Check new order
        task_ids = [task["id"] for task in supervisor.current_state["todo_list"]]
        assert task_ids == new_order

    def test_update_todo_tool_errors(self, supervisor):
        """Test update_todo tool error handling."""
        # Test without state
        result = supervisor.update_todo_tool(action="create")
        assert "error" in result

        supervisor.initialize_state("test objective")

        # Test unknown action
        result = supervisor.update_todo_tool(action="unknown_action")
        assert "error" in result
        assert "Unknown action" in result["error"]

        # Test create without objective
        result = supervisor.update_todo_tool(action="create")
        assert "error" in result
        assert "Objective is required" in result["error"]

        # Test add_task without task data
        result = supervisor.update_todo_tool(action="add_task")
        assert "error" in result
        assert "Task data is required" in result["error"]

    def test_are_dependencies_completed(self, supervisor):
        """Test dependency checking logic."""
        supervisor.initialize_state("test objective")

        # Add completed task result
        completed_result = {
            "task_id": "task_1",
            "status": TaskStatus.COMPLETED.value,
            "output": "Completed",
            "artifacts": [],
            "execution_time": 10.0,
            "tool_usage": {},
            "error_message": None,
            "completed_at": datetime.now().isoformat(),
        }
        supervisor.current_state["completed_tasks"].append(completed_result)

        # Test task with no dependencies
        task_no_deps = Task(
            id="task_2", description="No dependencies", task_type="general", priority=1
        )
        assert supervisor._are_dependencies_completed(task_no_deps) is True

        # Test task with completed dependency
        task_with_completed_dep = Task(
            id="task_3",
            description="Has completed dependency",
            task_type="general",
            priority=1,
            dependencies=["task_1"],
        )
        assert supervisor._are_dependencies_completed(task_with_completed_dep) is True

        # Test task with incomplete dependency
        task_with_incomplete_dep = Task(
            id="task_4",
            description="Has incomplete dependency",
            task_type="general",
            priority=1,
            dependencies=["task_nonexistent"],
        )
        assert supervisor._are_dependencies_completed(task_with_incomplete_dep) is False

    def test_task_tool_success(self, supervisor, mock_subagent_factory):
        """Test successful task execution with task_tool."""
        supervisor.initialize_state("test objective")

        # Add a task
        task_data = {
            "id": "test_task",
            "description": "Test task",
            "task_type": "analysis",
            "priority": 1,
            "status": TaskStatus.PENDING.value,
        }
        supervisor.current_state["todo_list"].append(task_data)

        # Mock subagent creation
        mock_subagent = Mock()
        mock_subagent.id = "agent_123"
        mock_subagent.available_tools = ["execute_code"]
        mock_subagent.created_at = datetime.now()
        mock_subagent_factory.create_agent.return_value = mock_subagent
        mock_subagent_factory.cleanup_agent.return_value = True

        result = supervisor.task_tool("test_task")

        assert result["task_id"] == "test_task"
        assert result["agent_id"] == "agent_123"
        assert result["status"] == TaskStatus.COMPLETED.value
        assert "output" in result
        assert "artifacts" in result

        # Check state updates
        assert len(supervisor.current_state["completed_tasks"]) == 1
        assert len(supervisor.current_state["subagent_logs"]) == 1
        assert supervisor.current_state["current_task"] is None

        # Verify subagent factory calls
        mock_subagent_factory.create_agent.assert_called_once()
        mock_subagent_factory.cleanup_agent.assert_called_once_with("agent_123")

    def test_task_tool_task_not_found(self, supervisor):
        """Test task_tool with non-existent task."""
        supervisor.initialize_state("test objective")

        result = supervisor.task_tool("nonexistent_task")

        assert "error" in result
        assert "not found" in result["error"]

    def test_task_tool_dependencies_not_completed(self, supervisor):
        """Test task_tool with incomplete dependencies."""
        supervisor.initialize_state("test objective")

        # Add a task with dependencies
        task_data = {
            "id": "dependent_task",
            "description": "Task with dependencies",
            "task_type": "analysis",
            "priority": 1,
            "dependencies": ["missing_task"],
            "status": TaskStatus.PENDING.value,
        }
        supervisor.current_state["todo_list"].append(task_data)

        result = supervisor.task_tool("dependent_task")

        assert "error" in result
        assert "dependencies not completed" in result["error"]
        assert result["dependencies"] == ["missing_task"]

    def test_get_next_task(self, supervisor):
        """Test getting the next ready task."""
        supervisor.initialize_state("test objective")

        # Add tasks with different priorities and dependencies
        tasks = [
            {
                "id": "task_1",
                "description": "High priority task",
                "task_type": "general",
                "priority": 1,
                "status": TaskStatus.PENDING.value,
                "dependencies": [],
            },
            {
                "id": "task_2",
                "description": "Low priority task",
                "task_type": "general",
                "priority": 3,
                "status": TaskStatus.PENDING.value,
                "dependencies": [],
            },
            {
                "id": "task_3",
                "description": "Dependent task",
                "task_type": "general",
                "priority": 2,
                "status": TaskStatus.PENDING.value,
                "dependencies": ["nonexistent"],
            },
        ]

        supervisor.current_state["todo_list"] = tasks

        next_task = supervisor.get_next_task()

        assert next_task is not None
        assert next_task["id"] == "task_1"  # Highest priority ready task

    def test_get_next_task_no_ready_tasks(self, supervisor):
        """Test getting next task when none are ready."""
        supervisor.initialize_state("test objective")

        # Add task with incomplete dependencies
        task_data = {
            "id": "blocked_task",
            "description": "Blocked task",
            "task_type": "general",
            "priority": 1,
            "status": TaskStatus.PENDING.value,
            "dependencies": ["missing_dependency"],
        }
        supervisor.current_state["todo_list"].append(task_data)

        next_task = supervisor.get_next_task()

        assert next_task is None

    def test_is_objective_complete(self, supervisor):
        """Test objective completion detection."""
        supervisor.initialize_state("test objective")

        # No tasks - should be complete
        assert supervisor.is_objective_complete() is True

        # Add pending task - should not be complete
        task_data = {"id": "pending_task", "status": TaskStatus.PENDING.value}
        supervisor.current_state["todo_list"].append(task_data)
        assert supervisor.is_objective_complete() is False

        # Complete the task - should be complete
        supervisor.current_state["todo_list"][0]["status"] = TaskStatus.COMPLETED.value
        assert supervisor.is_objective_complete() is True

        # Add failed task - should still be complete
        failed_task = {"id": "failed_task", "status": TaskStatus.FAILED.value}
        supervisor.current_state["todo_list"].append(failed_task)
        assert supervisor.is_objective_complete() is True

    def test_collect_results(self, supervisor):
        """Test result collection and consolidation."""
        supervisor.initialize_state("test objective")

        # Add completed task results
        results = [
            {
                "task_id": "task_1",
                "status": TaskStatus.COMPLETED.value,
                "output": "First task completed successfully",
                "artifacts": ["report1.md"],
                "execution_time": 15.5,
                "tool_usage": {"execute_code": 2},
                "error_message": None,
                "completed_at": datetime.now().isoformat(),
            },
            {
                "task_id": "task_2",
                "status": TaskStatus.COMPLETED.value,
                "output": "Second task completed successfully",
                "artifacts": ["report2.md", "data.json"],
                "execution_time": 22.3,
                "tool_usage": {"web_scrape": 1, "execute_code": 1},
                "error_message": None,
                "completed_at": datetime.now().isoformat(),
            },
        ]

        supervisor.current_state["completed_tasks"] = results

        consolidated = supervisor.collect_results()

        assert consolidated["objective"] == "test objective"
        assert consolidated["completed_tasks"] == 2
        assert consolidated["total_execution_time"] == 37.8
        assert consolidated["artifacts_created"] == [
            "report1.md",
            "report2.md",
            "data.json",
        ]
        assert consolidated["tool_usage_summary"] == {
            "execute_code": 3,
            "web_scrape": 1,
        }
        assert "task_1" in consolidated["consolidated_summary"]
        assert "task_2" in consolidated["consolidated_summary"]

    def test_collect_results_no_tasks(self, supervisor):
        """Test result collection with no completed tasks."""
        supervisor.initialize_state("test objective")

        result = supervisor.collect_results()

        assert result["completed_count"] == 0
        assert result["artifacts"] == []
        assert "No completed tasks" in result["message"]

    def test_update_plan_from_results_failed_task(self, supervisor):
        """Test plan updates for failed tasks."""
        supervisor.initialize_state("test objective")

        failed_result = TaskResult(
            task_id="failed_task",
            status=TaskStatus.FAILED.value,
            output="Task failed",
            error_message="Connection timeout",
        )

        update_result = supervisor.update_plan_from_results([failed_result])

        assert update_result["plan_modified"] is True
        assert len(update_result["updates_made"]) > 0
        assert "retry" in update_result["updates_made"][0].lower()

        # Check that retry task was added
        retry_tasks = [
            task
            for task in supervisor.current_state["todo_list"]
            if "retry" in task["id"]
        ]
        assert len(retry_tasks) == 1

    def test_update_plan_from_results_bugs_found(self, supervisor):
        """Test plan updates when bugs are found."""
        supervisor.initialize_state("test objective")

        bug_result = TaskResult(
            task_id="test_task",
            status=TaskStatus.COMPLETED.value,
            output="Testing completed. Found 3 critical bugs in the login system.",
        )

        update_result = supervisor.update_plan_from_results([bug_result])

        assert update_result["plan_modified"] is True
        assert len(update_result["updates_made"]) > 0
        assert "follow-up" in update_result["updates_made"][0].lower()

        # Check that follow-up task was added
        followup_tasks = [
            task
            for task in supervisor.current_state["todo_list"]
            if "followup" in task["id"]
        ]
        assert len(followup_tasks) == 1

    def test_get_supervisor_tools(self, supervisor, mock_tool_registry):
        """Test getting supervisor tools."""
        mock_shared_tools = {
            "read_file": Mock(),
            "write_file": Mock(),
            "edit_file": Mock(),
        }
        mock_tool_registry.get_agent_tools.return_value = mock_shared_tools

        tools = supervisor.get_supervisor_tools()

        assert "update_todo" in tools
        assert "task_tool" in tools
        assert "read_file" in tools
        assert "write_file" in tools
        assert "edit_file" in tools

        # Verify supervisor-specific tools are callable
        assert callable(tools["update_todo"])
        assert callable(tools["task_tool"])

    def test_simulate_task_execution_different_types(self, supervisor):
        """Test task execution simulation for different task types."""
        # Test web_testing task
        web_task = Task(
            id="web_test",
            description="Test web application",
            task_type="web_testing",
            priority=1,
        )

        result = supervisor._simulate_task_execution(web_task, Mock())
        assert result["status"] == TaskStatus.COMPLETED.value
        assert "web_scrape" in result["tool_usage"]
        assert "execute_code" in result["tool_usage"]

        # Test research task
        research_task = Task(
            id="research_test",
            description="Research AI trends",
            task_type="research",
            priority=1,
        )

        result = supervisor._simulate_task_execution(research_task, Mock())
        assert result["status"] == TaskStatus.COMPLETED.value
        assert "search_internet" in result["tool_usage"]
        assert "web_scrape" in result["tool_usage"]

        # Test analysis task
        analysis_task = Task(
            id="analysis_test",
            description="Analyze code",
            task_type="analysis",
            priority=1,
        )

        result = supervisor._simulate_task_execution(analysis_task, Mock())
        assert result["status"] == TaskStatus.COMPLETED.value
        assert "execute_code" in result["tool_usage"]

    def test_state_management(self, supervisor):
        """Test state getter and setter methods."""
        # Initially no state
        assert supervisor.get_state() is None

        # Initialize state
        state = supervisor.initialize_state("test objective")
        assert supervisor.get_state() == state

        # Set new state
        new_state = {
            "user_objective": "new objective",
            "todo_list": [],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 5,
            "final_result": "completed",
        }

        supervisor.set_state(new_state)
        assert supervisor.get_state() == new_state
        assert supervisor.current_state == new_state


if __name__ == "__main__":
    pytest.main([__file__])

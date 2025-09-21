"""Unit tests for data models."""

import pytest
from datetime import datetime
from src.models.data_models import (
    Task,
    TaskResult,
    SubAgent,
    TaskStatus,
    TaskType,
    task_to_dict,
    dict_to_task,
    task_result_to_dict,
    dict_to_task_result,
)


class TestTask:
    """Test cases for Task dataclass."""

    def test_task_creation_with_required_fields(self):
        """Test creating a task with only required fields."""
        task = Task(
            id="task-1",
            description="Test task",
            task_type=TaskType.GENERAL.value,
            priority=1,
        )

        assert task.id == "task-1"
        assert task.description == "Test task"
        assert task.task_type == TaskType.GENERAL.value
        assert task.priority == 1
        assert task.dependencies == []
        assert task.required_tools == []
        assert task.success_criteria == ""
        assert task.context == {}
        assert task.status == TaskStatus.PENDING.value
        assert isinstance(task.created_at, datetime)

    def test_task_creation_with_all_fields(self):
        """Test creating a task with all fields specified."""
        created_time = datetime.now()
        task = Task(
            id="task-2",
            description="Complex task",
            task_type=TaskType.WEB_TESTING.value,
            priority=2,
            dependencies=["task-1"],
            required_tools=["web_scrape", "execute_code"],
            success_criteria="All tests pass",
            context={"url": "https://example.com"},
            status=TaskStatus.IN_PROGRESS.value,
            created_at=created_time,
        )

        assert task.id == "task-2"
        assert task.description == "Complex task"
        assert task.task_type == TaskType.WEB_TESTING.value
        assert task.priority == 2
        assert task.dependencies == ["task-1"]
        assert task.required_tools == ["web_scrape", "execute_code"]
        assert task.success_criteria == "All tests pass"
        assert task.context == {"url": "https://example.com"}
        assert task.status == TaskStatus.IN_PROGRESS.value
        assert task.created_at == created_time

    def test_task_validation_empty_id(self):
        """Test task validation fails with empty ID."""
        with pytest.raises(ValueError, match="Task ID cannot be empty"):
            Task(
                id="",
                description="Test task",
                task_type=TaskType.GENERAL.value,
                priority=1,
            )

    def test_task_validation_empty_description(self):
        """Test task validation fails with empty description."""
        with pytest.raises(ValueError, match="Task description cannot be empty"):
            Task(
                id="task-1",
                description="",
                task_type=TaskType.GENERAL.value,
                priority=1,
            )

    def test_task_validation_negative_priority(self):
        """Test task validation fails with negative priority."""
        with pytest.raises(ValueError, match="Task priority must be non-negative"):
            Task(
                id="task-1",
                description="Test task",
                task_type=TaskType.GENERAL.value,
                priority=-1,
            )

    def test_task_validation_invalid_status(self):
        """Test task validation fails with invalid status."""
        with pytest.raises(ValueError, match="Invalid task status"):
            Task(
                id="task-1",
                description="Test task",
                task_type=TaskType.GENERAL.value,
                priority=1,
                status="invalid_status",
            )


class TestTaskResult:
    """Test cases for TaskResult dataclass."""

    def test_task_result_creation_with_required_fields(self):
        """Test creating a task result with only required fields."""
        result = TaskResult(
            task_id="task-1",
            status=TaskStatus.COMPLETED.value,
            output="Task completed successfully",
        )

        assert result.task_id == "task-1"
        assert result.status == TaskStatus.COMPLETED.value
        assert result.output == "Task completed successfully"
        assert result.artifacts == []
        assert result.execution_time == 0.0
        assert result.tool_usage == {}
        assert result.error_message is None
        assert isinstance(result.completed_at, datetime)

    def test_task_result_creation_with_all_fields(self):
        """Test creating a task result with all fields specified."""
        completed_time = datetime.now()
        result = TaskResult(
            task_id="task-2",
            status=TaskStatus.FAILED.value,
            output="Task failed",
            artifacts=["report.md", "logs.txt"],
            execution_time=45.5,
            tool_usage={"web_scrape": 3, "execute_code": 1},
            error_message="Connection timeout",
            completed_at=completed_time,
        )

        assert result.task_id == "task-2"
        assert result.status == TaskStatus.FAILED.value
        assert result.output == "Task failed"
        assert result.artifacts == ["report.md", "logs.txt"]
        assert result.execution_time == 45.5
        assert result.tool_usage == {"web_scrape": 3, "execute_code": 1}
        assert result.error_message == "Connection timeout"
        assert result.completed_at == completed_time

    def test_task_result_validation_empty_task_id(self):
        """Test task result validation fails with empty task ID."""
        with pytest.raises(ValueError, match="Task ID cannot be empty"):
            TaskResult(
                task_id="", status=TaskStatus.COMPLETED.value, output="Test output"
            )

    def test_task_result_validation_invalid_status(self):
        """Test task result validation fails with invalid status."""
        with pytest.raises(ValueError, match="Invalid task result status"):
            TaskResult(task_id="task-1", status="invalid_status", output="Test output")

    def test_task_result_validation_negative_execution_time(self):
        """Test task result validation fails with negative execution time."""
        with pytest.raises(ValueError, match="Execution time must be non-negative"):
            TaskResult(
                task_id="task-1",
                status=TaskStatus.COMPLETED.value,
                output="Test output",
                execution_time=-1.0,
            )


class TestSubAgent:
    """Test cases for SubAgent dataclass."""

    def test_subagent_creation_with_required_fields(self):
        """Test creating a subagent with only required fields."""
        task = Task(
            id="task-1",
            description="Test task",
            task_type=TaskType.GENERAL.value,
            priority=1,
        )

        agent = SubAgent(id="agent-1", task=task, prompt="You are a helpful assistant")

        assert agent.id == "agent-1"
        assert agent.task == task
        assert agent.prompt == "You are a helpful assistant"
        assert agent.available_tools == []
        assert agent.llm_config == {}
        assert isinstance(agent.created_at, datetime)
        assert agent.status == "created"

    def test_subagent_creation_with_all_fields(self):
        """Test creating a subagent with all fields specified."""
        task = Task(
            id="task-2",
            description="Complex task",
            task_type=TaskType.RESEARCH.value,
            priority=2,
        )

        created_time = datetime.now()
        agent = SubAgent(
            id="agent-2",
            task=task,
            prompt="You are a research assistant",
            available_tools=["search_internet", "write_file"],
            llm_config={"model": "gpt-4", "temperature": 0.7},
            created_at=created_time,
            status="active",
        )

        assert agent.id == "agent-2"
        assert agent.task == task
        assert agent.prompt == "You are a research assistant"
        assert agent.available_tools == ["search_internet", "write_file"]
        assert agent.llm_config == {"model": "gpt-4", "temperature": 0.7}
        assert agent.created_at == created_time
        assert agent.status == "active"

    def test_subagent_validation_empty_id(self):
        """Test subagent validation fails with empty ID."""
        task = Task(
            id="task-1",
            description="Test task",
            task_type=TaskType.GENERAL.value,
            priority=1,
        )

        with pytest.raises(ValueError, match="SubAgent ID cannot be empty"):
            SubAgent(id="", task=task, prompt="Test prompt")

    def test_subagent_validation_invalid_task(self):
        """Test subagent validation fails with invalid task."""
        with pytest.raises(ValueError, match="SubAgent task must be a Task instance"):
            SubAgent(
                id="agent-1",
                task="not a task",  # type: ignore
                prompt="Test prompt",
            )

    def test_subagent_validation_empty_prompt(self):
        """Test subagent validation fails with empty prompt."""
        task = Task(
            id="task-1",
            description="Test task",
            task_type=TaskType.GENERAL.value,
            priority=1,
        )

        with pytest.raises(ValueError, match="SubAgent prompt cannot be empty"):
            SubAgent(id="agent-1", task=task, prompt="")


class TestSerializationHelpers:
    """Test cases for serialization helper functions."""

    def test_task_serialization_roundtrip(self):
        """Test task serialization and deserialization."""
        original_task = Task(
            id="task-1",
            description="Test task",
            task_type=TaskType.WEB_TESTING.value,
            priority=2,
            dependencies=["task-0"],
            required_tools=["web_scrape"],
            success_criteria="All tests pass",
            context={"url": "https://example.com"},
            status=TaskStatus.IN_PROGRESS.value,
        )

        # Serialize to dict
        task_dict = task_to_dict(original_task)

        # Verify dict structure
        assert task_dict["id"] == "task-1"
        assert task_dict["description"] == "Test task"
        assert task_dict["task_type"] == TaskType.WEB_TESTING.value
        assert task_dict["priority"] == 2
        assert task_dict["dependencies"] == ["task-0"]
        assert task_dict["required_tools"] == ["web_scrape"]
        assert task_dict["success_criteria"] == "All tests pass"
        assert task_dict["context"] == {"url": "https://example.com"}
        assert task_dict["status"] == TaskStatus.IN_PROGRESS.value
        assert "created_at" in task_dict

        # Deserialize back to Task
        restored_task = dict_to_task(task_dict)

        # Verify restoration
        assert restored_task.id == original_task.id
        assert restored_task.description == original_task.description
        assert restored_task.task_type == original_task.task_type
        assert restored_task.priority == original_task.priority
        assert restored_task.dependencies == original_task.dependencies
        assert restored_task.required_tools == original_task.required_tools
        assert restored_task.success_criteria == original_task.success_criteria
        assert restored_task.context == original_task.context
        assert restored_task.status == original_task.status
        # Note: datetime comparison might have slight differences due to serialization

    def test_task_result_serialization_roundtrip(self):
        """Test task result serialization and deserialization."""
        original_result = TaskResult(
            task_id="task-1",
            status=TaskStatus.COMPLETED.value,
            output="Task completed successfully",
            artifacts=["report.md", "data.json"],
            execution_time=30.5,
            tool_usage={"web_scrape": 2, "write_file": 1},
            error_message=None,
        )

        # Serialize to dict
        result_dict = task_result_to_dict(original_result)

        # Verify dict structure
        assert result_dict["task_id"] == "task-1"
        assert result_dict["status"] == TaskStatus.COMPLETED.value
        assert result_dict["output"] == "Task completed successfully"
        assert result_dict["artifacts"] == ["report.md", "data.json"]
        assert result_dict["execution_time"] == 30.5
        assert result_dict["tool_usage"] == {"web_scrape": 2, "write_file": 1}
        assert result_dict["error_message"] is None
        assert "completed_at" in result_dict

        # Deserialize back to TaskResult
        restored_result = dict_to_task_result(result_dict)

        # Verify restoration
        assert restored_result.task_id == original_result.task_id
        assert restored_result.status == original_result.status
        assert restored_result.output == original_result.output
        assert restored_result.artifacts == original_result.artifacts
        assert restored_result.execution_time == original_result.execution_time
        assert restored_result.tool_usage == original_result.tool_usage
        assert restored_result.error_message == original_result.error_message

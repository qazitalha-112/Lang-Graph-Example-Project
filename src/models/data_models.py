"""Data models for the LangGraph Supervisor Agent system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
from enum import Enum


class TaskStatus(Enum):
    """Enumeration for task status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(Enum):
    """Enumeration for task types."""

    WEB_TESTING = "web_testing"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATION = "file_operation"
    GENERAL = "general"


@dataclass
class Task:
    """Represents a task to be executed by a subagent."""

    id: str
    description: str
    task_type: str
    priority: int
    dependencies: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    success_criteria: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = TaskStatus.PENDING.value
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate task fields after initialization."""
        if not self.id:
            raise ValueError("Task ID cannot be empty")
        if not self.description:
            raise ValueError("Task description cannot be empty")
        if self.priority < 0:
            raise ValueError("Task priority must be non-negative")
        if self.status not in [status.value for status in TaskStatus]:
            raise ValueError(f"Invalid task status: {self.status}")


@dataclass
class TaskResult:
    """Represents the result of a task execution."""

    task_id: str
    status: str
    output: str
    artifacts: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    tool_usage: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate task result fields after initialization."""
        if not self.task_id:
            raise ValueError("Task ID cannot be empty")
        if self.status not in [status.value for status in TaskStatus]:
            raise ValueError(f"Invalid task result status: {self.status}")
        if self.execution_time < 0:
            raise ValueError("Execution time must be non-negative")


@dataclass
class SubAgent:
    """Represents metadata for a dynamically created subagent."""

    id: str
    task: Task
    prompt: str
    available_tools: List[str] = field(default_factory=list)
    llm_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "created"

    def __post_init__(self):
        """Validate subagent fields after initialization."""
        if not self.id:
            raise ValueError("SubAgent ID cannot be empty")
        if not isinstance(self.task, Task):
            raise ValueError("SubAgent task must be a Task instance")
        if not self.prompt:
            raise ValueError("SubAgent prompt cannot be empty")


class AgentState(TypedDict):
    """LangGraph state schema for the supervisor agent system."""

    user_objective: str
    todo_list: List[Dict[str, Any]]  # Serialized Task objects
    completed_tasks: List[Dict[str, Any]]  # Serialized TaskResult objects
    current_task: Optional[Dict[str, Any]]  # Serialized Task object
    artifacts: Dict[str, Any]
    subagent_logs: List[Dict[str, Any]]
    file_system: Dict[str, str]  # Virtual file system state
    iteration_count: int
    final_result: Optional[str]


# Helper functions for serialization/deserialization
def task_to_dict(task: Task) -> Dict[str, Any]:
    """Convert Task dataclass to dictionary for state storage."""
    return {
        "id": task.id,
        "description": task.description,
        "task_type": task.task_type,
        "priority": task.priority,
        "dependencies": task.dependencies,
        "required_tools": task.required_tools,
        "success_criteria": task.success_criteria,
        "context": task.context,
        "status": task.status,
        "created_at": task.created_at.isoformat(),
    }


def dict_to_task(data: Dict[str, Any]) -> Task:
    """Convert dictionary to Task dataclass from state storage."""
    # Convert ISO format string back to datetime
    created_at = (
        datetime.fromisoformat(data["created_at"])
        if "created_at" in data
        else datetime.now()
    )

    return Task(
        id=data["id"],
        description=data["description"],
        task_type=data["task_type"],
        priority=data["priority"],
        dependencies=data.get("dependencies", []),
        required_tools=data.get("required_tools", []),
        success_criteria=data.get("success_criteria", ""),
        context=data.get("context", {}),
        status=data.get("status", TaskStatus.PENDING.value),
        created_at=created_at,
    )


def task_result_to_dict(result: TaskResult) -> Dict[str, Any]:
    """Convert TaskResult dataclass to dictionary for state storage."""
    return {
        "task_id": result.task_id,
        "status": result.status,
        "output": result.output,
        "artifacts": result.artifacts,
        "execution_time": result.execution_time,
        "tool_usage": result.tool_usage,
        "error_message": result.error_message,
        "completed_at": result.completed_at.isoformat(),
    }


def dict_to_task_result(data: Dict[str, Any]) -> TaskResult:
    """Convert dictionary to TaskResult dataclass from state storage."""
    # Convert ISO format string back to datetime
    completed_at = (
        datetime.fromisoformat(data["completed_at"])
        if "completed_at" in data
        else datetime.now()
    )

    return TaskResult(
        task_id=data["task_id"],
        status=data["status"],
        output=data["output"],
        artifacts=data.get("artifacts", []),
        execution_time=data.get("execution_time", 0.0),
        tool_usage=data.get("tool_usage", {}),
        error_message=data.get("error_message"),
        completed_at=completed_at,
    )

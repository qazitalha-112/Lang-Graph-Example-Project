"""Metrics collection for tool usage and performance tracking."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import json

from ..config import AgentConfig


@dataclass
class ToolMetrics:
    """Metrics for a specific tool."""

    tool_name: str
    usage_count: int = 0
    total_execution_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    execution_times: List[float] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        total = self.usage_count
        return (self.success_count / total * 100) if total > 0 else 0.0

    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        return (
            self.total_execution_time / self.usage_count
            if self.usage_count > 0
            else 0.0
        )

    @property
    def median_execution_time(self) -> float:
        """Calculate median execution time."""
        return statistics.median(self.execution_times) if self.execution_times else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "usage_count": self.usage_count,
            "total_execution_time": self.total_execution_time,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "average_execution_time": self.average_execution_time,
            "median_execution_time": self.median_execution_time,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "recent_errors": self.error_messages[-5:],  # Last 5 errors
        }


@dataclass
class AgentMetrics:
    """Metrics for a specific agent."""

    agent_id: str
    agent_type: str  # "supervisor" or "subagent"
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    tool_usage: Dict[str, int] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate task success rate as a percentage."""
        total = self.tasks_completed + self.tasks_failed
        return (self.tasks_completed / total * 100) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.success_rate,
            "total_execution_time": self.total_execution_time,
            "tool_usage": self.tool_usage,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
        }


@dataclass
class SessionMetrics:
    """Metrics for a complete supervisor session."""

    session_id: str
    objective: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_subagents: int = 0
    total_execution_time: float = 0.0
    artifacts_created: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Calculate session duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def completion_rate(self) -> float:
        """Calculate task completion rate as a percentage."""
        return (
            (self.completed_tasks / self.total_tasks * 100)
            if self.total_tasks > 0
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "objective": self.objective,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "completion_rate": self.completion_rate,
            "total_subagents": self.total_subagents,
            "total_execution_time": self.total_execution_time,
            "artifacts_created": self.artifacts_created,
        }


class MetricsCollector:
    """
    Collects and manages metrics for tool usage, agent performance, and session tracking.

    This class provides:
    - Tool usage metrics collection
    - Agent performance tracking
    - Session-level metrics
    - Aggregated statistics and reporting
    - Integration with LangSmith for metrics export
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the metrics collector.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.tool_metrics: Dict[str, ToolMetrics] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.session_metrics: Optional[SessionMetrics] = None
        self.current_session_id: Optional[str] = None

    def start_session(self, session_id: str, objective: str) -> None:
        """
        Start tracking a new supervisor session.

        Args:
            session_id: Unique identifier for the session
            objective: The user objective for this session
        """
        self.current_session_id = session_id
        self.session_metrics = SessionMetrics(
            session_id=session_id, objective=objective, start_time=datetime.now()
        )

    def end_session(self) -> Optional[SessionMetrics]:
        """
        End the current session and return final metrics.

        Returns:
            SessionMetrics for the completed session, or None if no active session
        """
        if self.session_metrics:
            self.session_metrics.end_time = datetime.now()
            return self.session_metrics
        return None

    def record_tool_usage(
        self,
        tool_name: str,
        execution_time: float,
        success: bool,
        agent_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Record tool usage metrics.

        Args:
            tool_name: Name of the tool used
            execution_time: Time taken to execute the tool
            success: Whether the tool execution was successful
            agent_id: ID of the agent that used the tool
            error_message: Error message if the tool failed
        """
        # Update tool metrics
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolMetrics(tool_name=tool_name)

        tool_metric = self.tool_metrics[tool_name]
        tool_metric.usage_count += 1
        tool_metric.total_execution_time += execution_time
        tool_metric.execution_times.append(execution_time)
        tool_metric.last_used = datetime.now()

        if success:
            tool_metric.success_count += 1
        else:
            tool_metric.error_count += 1
            if error_message:
                tool_metric.error_messages.append(error_message)

        # Update agent metrics if agent_id provided
        if agent_id:
            self._update_agent_tool_usage(agent_id, tool_name)

    def record_task_completion(
        self,
        agent_id: str,
        agent_type: str,
        success: bool,
        execution_time: float,
        task_id: Optional[str] = None,
    ) -> None:
        """
        Record task completion metrics.

        Args:
            agent_id: ID of the agent that completed the task
            agent_type: Type of agent ("supervisor" or "subagent")
            success: Whether the task was completed successfully
            execution_time: Time taken to complete the task
            task_id: ID of the completed task
        """
        # Update agent metrics
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id, agent_type=agent_type, created_at=datetime.now()
            )

        agent_metric = self.agent_metrics[agent_id]
        agent_metric.total_execution_time += execution_time
        agent_metric.last_active = datetime.now()

        if success:
            agent_metric.tasks_completed += 1
        else:
            agent_metric.tasks_failed += 1

        # Update session metrics
        if self.session_metrics:
            if success:
                self.session_metrics.completed_tasks += 1
            else:
                self.session_metrics.failed_tasks += 1

            self.session_metrics.total_execution_time += execution_time

            if agent_type == "subagent":
                self.session_metrics.total_subagents += 1

    def record_artifact_creation(self, artifact_path: str) -> None:
        """
        Record artifact creation.

        Args:
            artifact_path: Path of the created artifact
        """
        if self.session_metrics:
            self.session_metrics.artifacts_created.append(artifact_path)

    def _update_agent_tool_usage(self, agent_id: str, tool_name: str) -> None:
        """Update tool usage count for a specific agent."""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id, agent_type="unknown", created_at=datetime.now()
            )

        agent_metric = self.agent_metrics[agent_id]
        agent_metric.tool_usage[tool_name] = (
            agent_metric.tool_usage.get(tool_name, 0) + 1
        )

    def get_tool_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get tool usage metrics.

        Args:
            tool_name: Specific tool to get metrics for, or None for all tools

        Returns:
            Dictionary containing tool metrics
        """
        if tool_name:
            if tool_name in self.tool_metrics:
                return self.tool_metrics[tool_name].to_dict()
            return {}

        return {tool: metrics.to_dict() for tool, metrics in self.tool_metrics.items()}

    def get_agent_metrics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get agent performance metrics.

        Args:
            agent_id: Specific agent to get metrics for, or None for all agents

        Returns:
            Dictionary containing agent metrics
        """
        if agent_id:
            if agent_id in self.agent_metrics:
                return self.agent_metrics[agent_id].to_dict()
            return {}

        return {
            agent: metrics.to_dict() for agent, metrics in self.agent_metrics.items()
        }

    def get_session_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get current session metrics.

        Returns:
            Dictionary containing session metrics, or None if no active session
        """
        if self.session_metrics:
            return self.session_metrics.to_dict()
        return None

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all metrics.

        Returns:
            Dictionary containing summary statistics
        """
        # Tool summary
        tool_summary = {
            "total_tools_used": len(self.tool_metrics),
            "most_used_tool": self._get_most_used_tool(),
            "average_tool_success_rate": self._calculate_average_tool_success_rate(),
            "total_tool_executions": sum(
                m.usage_count for m in self.tool_metrics.values()
            ),
        }

        # Agent summary
        agent_summary = {
            "total_agents": len(self.agent_metrics),
            "supervisor_agents": len(
                [a for a in self.agent_metrics.values() if a.agent_type == "supervisor"]
            ),
            "subagents": len(
                [a for a in self.agent_metrics.values() if a.agent_type == "subagent"]
            ),
            "average_agent_success_rate": self._calculate_average_agent_success_rate(),
        }

        # Session summary
        session_summary = {}
        if self.session_metrics:
            session_summary = {
                "session_active": self.session_metrics.end_time is None,
                "session_duration": self.session_metrics.duration,
                "completion_rate": self.session_metrics.completion_rate,
                "total_artifacts": len(self.session_metrics.artifacts_created),
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "session": session_summary,
            "tools": tool_summary,
            "agents": agent_summary,
            "detailed_tool_metrics": self.get_tool_metrics(),
            "detailed_agent_metrics": self.get_agent_metrics(),
        }

    def _get_most_used_tool(self) -> Optional[str]:
        """Get the name of the most frequently used tool."""
        if not self.tool_metrics:
            return None

        return max(
            self.tool_metrics.keys(), key=lambda k: self.tool_metrics[k].usage_count
        )

    def _calculate_average_tool_success_rate(self) -> float:
        """Calculate average success rate across all tools."""
        if not self.tool_metrics:
            return 0.0

        success_rates = [m.success_rate for m in self.tool_metrics.values()]
        return statistics.mean(success_rates) if success_rates else 0.0

    def _calculate_average_agent_success_rate(self) -> float:
        """Calculate average success rate across all agents."""
        if not self.agent_metrics:
            return 0.0

        success_rates = [m.success_rate for m in self.agent_metrics.values()]
        return statistics.mean(success_rates) if success_rates else 0.0

    def export_metrics(self, format_type: str = "json") -> str:
        """
        Export all metrics in the specified format.

        Args:
            format_type: Export format ("json" or "csv")

        Returns:
            Formatted metrics string
        """
        if format_type.lower() == "json":
            return json.dumps(self.get_summary_report(), indent=2)
        elif format_type.lower() == "csv":
            # Simple CSV export for tool metrics
            lines = ["tool_name,usage_count,success_rate,avg_execution_time"]
            for tool_name, metrics in self.tool_metrics.items():
                lines.append(
                    f"{tool_name},{metrics.usage_count},{metrics.success_rate:.2f},{metrics.average_execution_time:.3f}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.tool_metrics.clear()
        self.agent_metrics.clear()
        self.session_metrics = None
        self.current_session_id = None

    def get_performance_insights(self) -> Dict[str, Any]:
        """
        Generate performance insights and recommendations.

        Returns:
            Dictionary containing insights and recommendations
        """
        insights = {"recommendations": [], "warnings": [], "performance_highlights": []}

        # Analyze tool performance
        for tool_name, metrics in self.tool_metrics.items():
            if metrics.success_rate < 80:
                insights["warnings"].append(
                    f"Tool '{tool_name}' has low success rate: {metrics.success_rate:.1f}%"
                )

            if metrics.average_execution_time > 30:
                insights["warnings"].append(
                    f"Tool '{tool_name}' has high average execution time: {metrics.average_execution_time:.1f}s"
                )

            if metrics.usage_count > 10 and metrics.success_rate > 95:
                insights["performance_highlights"].append(
                    f"Tool '{tool_name}' is highly reliable with {metrics.success_rate:.1f}% success rate"
                )

        # Analyze agent performance
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.success_rate < 70:
                insights["recommendations"].append(
                    f"Agent '{agent_id}' may need prompt optimization (success rate: {metrics.success_rate:.1f}%)"
                )

        # Session-level insights
        if self.session_metrics:
            if self.session_metrics.completion_rate < 80:
                insights["recommendations"].append(
                    "Consider breaking down complex objectives into smaller tasks"
                )

            if len(self.session_metrics.artifacts_created) == 0:
                insights["warnings"].append(
                    "No artifacts were created during this session"
                )

        return insights

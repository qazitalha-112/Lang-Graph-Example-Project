"""Simplified LangGraph workflow orchestration for testing and demonstration."""

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass

from .models.data_models import AgentState, TaskStatus
from .agents.supervisor import SupervisorAgent
from .agents.subagent_factory import SubAgentFactory
from .tools.tool_registry import ToolRegistry
from .models.virtual_file_system import VirtualFileSystem
from .config import AgentConfig


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    objective: str
    final_result: str
    iterations: int
    completed_tasks: int
    artifacts: Dict[str, Any]
    file_system: Dict[str, str]
    success: bool
    error_message: Optional[str] = None


class SimpleWorkflow:
    """
    Simplified workflow orchestration that implements the supervisor pattern
    without requiring full LangGraph integration for testing and demonstration.

    This class provides the same functionality as the full LangGraph workflow
    but uses a simpler execution model for easier testing and debugging.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the simple workflow.

        Args:
            config: Agent configuration
        """
        self.config = config

        # Initialize core components
        self.vfs = VirtualFileSystem()
        self.tool_registry = ToolRegistry(config, self.vfs)
        self.subagent_factory = SubAgentFactory(config, self.tool_registry, self.vfs)
        self.supervisor = SupervisorAgent(
            config, self.tool_registry, self.vfs, self.subagent_factory
        )

    def run(self, objective: str) -> WorkflowResult:
        """
        Run the complete workflow for a given objective.

        Args:
            objective: The user objective to accomplish

        Returns:
            WorkflowResult with execution details
        """
        try:
            # Initialize state
            state = self.supervisor.initialize_state(objective)

            # Create initial plan
            plan_result = self.supervisor.update_todo_tool(
                "create", objective=objective
            )
            if "error" in plan_result:
                return WorkflowResult(
                    objective=objective,
                    final_result=f"Failed to create plan: {plan_result['error']}",
                    iterations=0,
                    completed_tasks=0,
                    artifacts={},
                    file_system={},
                    success=False,
                    error_message=plan_result["error"],
                )

            # Execute workflow loop
            iteration = 0
            while iteration < self.config.max_iterations:
                iteration += 1
                state["iteration_count"] = iteration

                # Check if objective is complete
                if self.supervisor.is_objective_complete():
                    break

                # Get next task to execute
                next_task = self.supervisor.get_next_task()
                if not next_task:
                    # No tasks ready, try to collect results and update plan
                    self._collect_and_update_plan()
                    continue

                # Execute the task
                task_result = self.supervisor.task_tool(next_task["id"])

                # Handle task execution result
                if "error" in task_result:
                    print(f"Task {next_task['id']} failed: {task_result['error']}")
                    # Continue with other tasks
                    continue

                print(
                    f"Task {next_task['id']} completed: {task_result.get('status', 'unknown')}"
                )

            # Collect final results
            final_results = self.supervisor.collect_results()

            # Generate final result summary
            summary = self._generate_final_summary(final_results, iteration)

            return WorkflowResult(
                objective=objective,
                final_result=summary,
                iterations=iteration,
                completed_tasks=final_results.get("completed_tasks", 0),
                artifacts=final_results,
                file_system=dict(self.vfs.files),
                success=True,
            )

        except Exception as e:
            return WorkflowResult(
                objective=objective,
                final_result=f"Workflow execution failed: {str(e)}",
                iterations=0,
                completed_tasks=0,
                artifacts={},
                file_system={},
                success=False,
                error_message=str(e),
            )

    def _collect_and_update_plan(self) -> None:
        """Collect results and update plan if needed."""
        try:
            # Collect current results
            results = self.supervisor.collect_results()

            # Get completed task results for analysis
            completed_results = [
                result
                for result in self.supervisor.current_state["completed_tasks"]
                if result["status"] == TaskStatus.COMPLETED.value
            ]

            if completed_results:
                # Convert to TaskResult objects for analysis
                from .models.data_models import dict_to_task_result

                task_results = [
                    dict_to_task_result(result) for result in completed_results
                ]

                # Update plan based on results
                self.supervisor.update_plan_from_results(task_results)

        except Exception as e:
            print(f"Error in collect and update plan: {e}")

    def _generate_final_summary(self, results: Dict[str, Any], iterations: int) -> str:
        """
        Generate a final summary of the workflow execution.

        Args:
            results: Collected results from supervisor
            iterations: Number of iterations executed

        Returns:
            Formatted summary string
        """
        summary_parts = [
            f"Objective: {self.supervisor.current_state['user_objective']}",
            f"Tasks Completed: {results.get('completed_tasks', 0)}",
            f"Total Execution Time: {results.get('total_execution_time', 0):.1f}s",
            f"Artifacts Created: {len(results.get('artifacts_created', []))}",
            f"Iterations: {iterations}",
            "",
            "Summary:",
            results.get("consolidated_summary", "No summary available"),
        ]

        if results.get("artifacts_created"):
            summary_parts.extend(
                [
                    "",
                    "Artifacts Created:",
                    *[f"- {artifact}" for artifact in results["artifacts_created"]],
                ]
            )

        return "\n".join(summary_parts)

    def get_workflow_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the workflow execution.

        Returns:
            Dictionary with workflow statistics
        """
        current_state = self.supervisor.get_state()
        if not current_state:
            return {"error": "No workflow state available"}

        return {
            "objective": current_state.get("user_objective"),
            "total_tasks": len(current_state.get("todo_list", [])),
            "completed_tasks": len(
                [
                    task
                    for task in current_state.get("todo_list", [])
                    if task.get("status") == TaskStatus.COMPLETED.value
                ]
            ),
            "pending_tasks": len(
                [
                    task
                    for task in current_state.get("todo_list", [])
                    if task.get("status") == TaskStatus.PENDING.value
                ]
            ),
            "failed_tasks": len(
                [
                    task
                    for task in current_state.get("todo_list", [])
                    if task.get("status") == TaskStatus.FAILED.value
                ]
            ),
            "iterations": current_state.get("iteration_count", 0),
            "files_created": len(self.vfs.files),
            "subagents_created": len(current_state.get("subagent_logs", [])),
        }

    def get_task_details(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all tasks.

        Returns:
            List of task details
        """
        current_state = self.supervisor.get_state()
        if not current_state:
            return []

        return current_state.get("todo_list", [])

    def get_file_contents(self, file_path: str) -> Optional[str]:
        """
        Get the contents of a file from the virtual file system.

        Args:
            file_path: Path to the file

        Returns:
            File contents or None if not found
        """
        try:
            return self.vfs.read_file(file_path)
        except Exception:
            return None

    def list_files(self) -> List[str]:
        """
        List all files in the virtual file system.

        Returns:
            List of file paths
        """
        return self.vfs.list_files()

    def reset(self) -> None:
        """Reset the workflow to initial state."""
        self.vfs = VirtualFileSystem()
        self.tool_registry = ToolRegistry(self.config, self.vfs)
        self.subagent_factory = SubAgentFactory(
            self.config, self.tool_registry, self.vfs
        )
        self.supervisor = SupervisorAgent(
            self.config, self.tool_registry, self.vfs, self.subagent_factory
        )

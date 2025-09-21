"""Supervisor agent for orchestrating task decomposition and subagent coordination."""

import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re

from ..models.data_models import (
    Task,
    TaskResult,
    TaskStatus,
    AgentState,
    task_to_dict,
    dict_to_task,
    task_result_to_dict,
    dict_to_task_result,
)
from ..agents.subagent_factory import SubAgentFactory
from ..tools.tool_registry import ToolRegistry
from ..models.virtual_file_system import VirtualFileSystem
from ..config import AgentConfig
from ..tracing.langsmith_tracer import LangSmithTracer
from ..tracing.metrics_collector import MetricsCollector
from ..tracing.evaluation import EvaluationManager


class SupervisorAgent:
    """
    Central coordinator that manages the workflow by decomposing objectives,
    creating subagents, and collecting results.

    The Supervisor follows a plan-execute-update cycle where it:
    1. Decomposes user objectives into actionable tasks
    2. Creates specialized subagents to execute tasks
    3. Collects results and updates the plan
    4. Iterates until the objective is completed
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        vfs: VirtualFileSystem,
        subagent_factory: SubAgentFactory,
    ):
        """
        Initialize the Supervisor agent.

        Args:
            config: Agent configuration
            tool_registry: Registry for tool management
            vfs: Virtual file system
            subagent_factory: Factory for creating subagents
        """
        self.config = config
        self.tool_registry = tool_registry
        self.vfs = vfs
        self.subagent_factory = subagent_factory
        self.current_state: Optional[AgentState] = None

        # Initialize tracing and evaluation components
        self.tracer = LangSmithTracer(config)
        self.metrics_collector = MetricsCollector(config)
        self.evaluation_manager = EvaluationManager(config)

        # Set tracer on tool registry for tool usage tracking
        self.tool_registry.set_tracer(self.tracer)

    @property
    def _tracer_decorator(self):
        """Get tracer decorator for supervisor operations."""
        return self.tracer.supervisor_trace

    def decompose_objective(self, objective: str) -> List[Task]:
        """
        Decompose a user objective into actionable tasks.

        Args:
            objective: The high-level objective to decompose

        Returns:
            List of Task objects representing the decomposed plan

        This method analyzes the objective and creates a structured TODO list
        with appropriate task types, priorities, and dependencies.
        """

        # Apply tracing decorator
        @self.tracer.supervisor_trace("decompose_objective", {"objective": objective})
        def _decompose_with_tracing():
            return self._decompose_objective_impl(objective)

        return _decompose_with_tracing()

    def _decompose_objective_impl(self, objective: str) -> List[Task]:
        # Analyze objective to determine task types and complexity
        tasks = []
        task_counter = 1

        # Simple heuristic-based decomposition
        # In a real implementation, this would use LLM-based planning

        if "test" in objective.lower() and (
            "web" in objective.lower() or "app" in objective.lower()
        ):
            # Web application testing scenario
            tasks.extend(
                [
                    Task(
                        id=f"task_{task_counter}",
                        description="Analyze the application structure and identify key features to test",
                        task_type="analysis",
                        priority=1,
                        required_tools=["web_scrape", "execute_code"],
                        success_criteria="Create a comprehensive list of application features and test areas",
                        context={"objective": objective},
                    ),
                    Task(
                        id=f"task_{task_counter + 1}",
                        description="Test user authentication and login functionality",
                        task_type="web_testing",
                        priority=2,
                        dependencies=[f"task_{task_counter}"],
                        required_tools=["web_scrape", "execute_code"],
                        success_criteria="Verify login works correctly and document any issues found",
                        context={"objective": objective},
                    ),
                    Task(
                        id=f"task_{task_counter + 2}",
                        description="Test core application features and user workflows",
                        task_type="web_testing",
                        priority=3,
                        dependencies=[f"task_{task_counter + 1}"],
                        required_tools=["web_scrape", "execute_code"],
                        success_criteria="Complete testing of main features and document findings",
                        context={"objective": objective},
                    ),
                    Task(
                        id=f"task_{task_counter + 3}",
                        description="Compile test results and create bug report",
                        task_type="analysis",
                        priority=4,
                        dependencies=[f"task_{task_counter + 2}"],
                        required_tools=["execute_code"],
                        success_criteria="Generate comprehensive test report with findings and recommendations",
                        context={"objective": objective},
                    ),
                ]
            )

        elif "research" in objective.lower():
            # Research scenario
            tasks.extend(
                [
                    Task(
                        id=f"task_{task_counter}",
                        description=f"Conduct internet research on: {objective}",
                        task_type="research",
                        priority=1,
                        required_tools=["search_internet", "web_scrape"],
                        success_criteria="Gather comprehensive information from multiple reliable sources",
                        context={"objective": objective},
                    ),
                    Task(
                        id=f"task_{task_counter + 1}",
                        description="Analyze and synthesize research findings",
                        task_type="analysis",
                        priority=2,
                        dependencies=[f"task_{task_counter}"],
                        required_tools=["execute_code"],
                        success_criteria="Create structured analysis of research findings",
                        context={"objective": objective},
                    ),
                    Task(
                        id=f"task_{task_counter + 2}",
                        description="Create comprehensive research report",
                        task_type="analysis",
                        priority=3,
                        dependencies=[f"task_{task_counter + 1}"],
                        required_tools=["execute_code"],
                        success_criteria="Generate well-structured research report with citations",
                        context={"objective": objective},
                    ),
                ]
            )

        elif "analyze" in objective.lower() or "code" in objective.lower():
            # Code analysis scenario
            tasks.extend(
                [
                    Task(
                        id=f"task_{task_counter}",
                        description="Examine codebase structure and identify analysis areas",
                        task_type="analysis",
                        priority=1,
                        required_tools=["execute_code"],
                        success_criteria="Map out codebase structure and identify key areas for analysis",
                        context={"objective": objective},
                    ),
                    Task(
                        id=f"task_{task_counter + 1}",
                        description="Perform detailed code analysis and identify issues",
                        task_type="code_execution",
                        priority=2,
                        dependencies=[f"task_{task_counter}"],
                        required_tools=["execute_code"],
                        success_criteria="Complete analysis with identified issues and improvement opportunities",
                        context={"objective": objective},
                    ),
                    Task(
                        id=f"task_{task_counter + 2}",
                        description="Generate analysis report with recommendations",
                        task_type="analysis",
                        priority=3,
                        dependencies=[f"task_{task_counter + 1}"],
                        required_tools=["execute_code"],
                        success_criteria="Create comprehensive analysis report with actionable recommendations",
                        context={"objective": objective},
                    ),
                ]
            )

        else:
            # General objective - create a flexible plan
            tasks.append(
                Task(
                    id=f"task_{task_counter}",
                    description=f"Execute objective: {objective}",
                    task_type="general",
                    priority=1,
                    required_tools=["execute_code"],
                    success_criteria="Successfully complete the requested objective",
                    context={"objective": objective},
                )
            )

        return tasks

    def update_todo_tool(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Tool for plan creation and modification.

        Args:
            action: Action to perform ("create", "update", "add_task", "remove_task", "reorder")
            **kwargs: Additional parameters based on action

        Returns:
            Dictionary with operation results

        This tool allows the supervisor to create initial plans and modify them
        based on task results and changing requirements.
        """

        # Apply tracing decorator
        @self.tracer.supervisor_trace(
            "update_todo", {"action": action, "kwargs_keys": list(kwargs.keys())}
        )
        def _update_todo_with_tracing():
            return self._update_todo_tool_impl(action, **kwargs)

        return _update_todo_with_tracing()

    def _update_todo_tool_impl(self, action: str, **kwargs) -> Dict[str, Any]:
        if not self.current_state:
            return {"error": "No active state to update"}

        try:
            if action == "create":
                objective = kwargs.get("objective")
                if not objective:
                    return {"error": "Objective is required for plan creation"}

                tasks = self.decompose_objective(objective)
                self.current_state["todo_list"] = [task_to_dict(task) for task in tasks]
                self.current_state["user_objective"] = objective

                return {
                    "action": "create",
                    "tasks_created": len(tasks),
                    "todo_list": self.current_state["todo_list"],
                    "message": f"Created plan with {len(tasks)} tasks for objective: {objective}",
                }

            elif action == "add_task":
                task_data = kwargs.get("task")
                if not task_data:
                    return {"error": "Task data is required"}

                # Create task from provided data
                task = Task(
                    id=task_data.get("id", f"task_{uuid.uuid4().hex[:8]}"),
                    description=task_data["description"],
                    task_type=task_data.get("task_type", "general"),
                    priority=task_data.get("priority", 1),
                    dependencies=task_data.get("dependencies", []),
                    required_tools=task_data.get("required_tools", []),
                    success_criteria=task_data.get("success_criteria", ""),
                    context=task_data.get("context", {}),
                )

                self.current_state["todo_list"].append(task_to_dict(task))

                return {
                    "action": "add_task",
                    "task_added": task.id,
                    "message": f"Added task: {task.description}",
                }

            elif action == "update_task":
                task_id = kwargs.get("task_id")
                updates = kwargs.get("updates", {})

                if not task_id:
                    return {"error": "Task ID is required for update"}

                # Find and update task
                for i, task_dict in enumerate(self.current_state["todo_list"]):
                    if task_dict["id"] == task_id:
                        task_dict.update(updates)
                        return {
                            "action": "update_task",
                            "task_updated": task_id,
                            "message": f"Updated task {task_id}",
                        }

                return {"error": f"Task {task_id} not found"}

            elif action == "remove_task":
                task_id = kwargs.get("task_id")
                if not task_id:
                    return {"error": "Task ID is required for removal"}

                # Remove task and update dependencies
                original_count = len(self.current_state["todo_list"])
                self.current_state["todo_list"] = [
                    task
                    for task in self.current_state["todo_list"]
                    if task["id"] != task_id
                ]

                # Remove from dependencies of other tasks
                for task_dict in self.current_state["todo_list"]:
                    if task_id in task_dict.get("dependencies", []):
                        task_dict["dependencies"].remove(task_id)

                removed_count = original_count - len(self.current_state["todo_list"])

                return {
                    "action": "remove_task",
                    "task_removed": task_id if removed_count > 0 else None,
                    "message": f"Removed task {task_id}"
                    if removed_count > 0
                    else f"Task {task_id} not found",
                }

            elif action == "reorder":
                new_order = kwargs.get("task_order", [])
                if not new_order:
                    return {"error": "Task order list is required"}

                # Reorder tasks based on provided order
                ordered_tasks = []
                task_dict_map = {
                    task["id"]: task for task in self.current_state["todo_list"]
                }

                for task_id in new_order:
                    if task_id in task_dict_map:
                        ordered_tasks.append(task_dict_map[task_id])

                # Add any tasks not in the new order at the end
                for task in self.current_state["todo_list"]:
                    if task["id"] not in new_order:
                        ordered_tasks.append(task)

                self.current_state["todo_list"] = ordered_tasks

                return {
                    "action": "reorder",
                    "tasks_reordered": len(ordered_tasks),
                    "message": "Tasks reordered successfully",
                }

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Failed to update todo: {str(e)}"}

    def task_tool(
        self, task_id: str, custom_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Tool for subagent creation and execution.

        Args:
            task_id: ID of the task to execute
            custom_tools: Optional custom list of tools to assign

        Returns:
            Dictionary with task execution results

        This tool creates a subagent for the specified task, executes it,
        and returns the results for integration into the workflow.
        """

        # Apply tracing decorator
        @self.tracer.supervisor_trace(
            "task_execution", {"task_id": task_id, "custom_tools": custom_tools}
        )
        def _task_tool_with_tracing():
            return self._task_tool_impl(task_id, custom_tools)

        return _task_tool_with_tracing()

    def _task_tool_impl(
        self, task_id: str, custom_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not self.current_state:
            return {"error": "No active state available"}

        try:
            # Find the task in the todo list
            task_dict = None
            for t in self.current_state["todo_list"]:
                if t["id"] == task_id:
                    task_dict = t
                    break

            if not task_dict:
                return {"error": f"Task {task_id} not found in todo list"}

            # Convert to Task object
            task = dict_to_task(task_dict)

            # Check if task is ready to execute (dependencies completed)
            if not self._are_dependencies_completed(task):
                return {
                    "error": f"Task {task_id} cannot be executed - dependencies not completed",
                    "dependencies": task.dependencies,
                }

            # Update task status to in_progress
            task_dict["status"] = TaskStatus.IN_PROGRESS.value
            self.current_state["current_task"] = task_dict

            # Get previous results for context
            previous_results = [
                dict_to_task_result(result)
                for result in self.current_state["completed_tasks"]
            ]

            # Create subagent
            subagent = self.subagent_factory.create_agent(
                task=task, previous_results=previous_results, custom_tools=custom_tools
            )

            # Log subagent creation
            subagent_log = {
                "agent_id": subagent.id,
                "task_id": task.id,
                "created_at": subagent.created_at.isoformat(),
                "tools_assigned": subagent.available_tools,
                "status": "created",
            }
            self.current_state["subagent_logs"].append(subagent_log)

            # Simulate task execution (in real implementation, this would invoke the LLM)
            # For now, we'll create a mock result
            start_time = datetime.now()
            execution_result = self._simulate_task_execution(task, subagent)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Create task result
            task_result = TaskResult(
                task_id=task.id,
                status=execution_result["status"],
                output=execution_result["output"],
                artifacts=execution_result.get("artifacts", []),
                execution_time=execution_result.get("execution_time", execution_time),
                tool_usage=execution_result.get("tool_usage", {}),
                error_message=execution_result.get("error_message"),
            )

            # Collect metrics for task completion
            self.metrics_collector.record_task_completion(
                agent_id=subagent.id,
                agent_type="subagent",
                success=task_result.status == TaskStatus.COMPLETED.value,
                execution_time=task_result.execution_time,
                task_id=task.id,
            )

            # Record tool usage metrics
            for tool_name, usage_count in task_result.tool_usage.items():
                for _ in range(usage_count):
                    self.metrics_collector.record_tool_usage(
                        tool_name=tool_name,
                        execution_time=task_result.execution_time
                        / usage_count,  # Approximate per-tool time
                        success=task_result.status == TaskStatus.COMPLETED.value,
                        agent_id=subagent.id,
                        error_message=task_result.error_message
                        if task_result.status != TaskStatus.COMPLETED.value
                        else None,
                    )

            # Record artifact creation
            for artifact in task_result.artifacts:
                self.metrics_collector.record_artifact_creation(artifact)

            # Perform task evaluation
            task_evaluation = self.evaluation_manager.evaluate_task_comprehensive(
                task, task_result
            )

            # Update state
            task_dict["status"] = task_result.status
            self.current_state["completed_tasks"].append(
                task_result_to_dict(task_result)
            )
            self.current_state["current_task"] = None

            # Update subagent log
            for log in self.current_state["subagent_logs"]:
                if log["agent_id"] == subagent.id:
                    log["status"] = "completed"
                    log["completed_at"] = datetime.now().isoformat()
                    log["evaluation_score"] = task_evaluation.overall_score
                    break

            # Clean up subagent resources
            self.subagent_factory.cleanup_agent(subagent.id)

            return {
                "task_id": task.id,
                "agent_id": subagent.id,
                "status": task_result.status,
                "output": task_result.output,
                "artifacts": task_result.artifacts,
                "execution_time": task_result.execution_time,
                "message": f"Task {task_id} completed successfully",
                "success": task_result.status == TaskStatus.COMPLETED.value,
            }

        except Exception as e:
            # Update task status to failed
            if task_dict:
                task_dict["status"] = TaskStatus.FAILED.value

            # Record failed task metrics
            if "subagent" in locals():
                self.metrics_collector.record_task_completion(
                    agent_id=subagent.id,
                    agent_type="subagent",
                    success=False,
                    execution_time=0.0,
                    task_id=task_id,
                )

            return {"error": f"Failed to execute task {task_id}: {str(e)}"}

    def _are_dependencies_completed(self, task: Task) -> bool:
        """
        Check if all dependencies for a task are completed.

        Args:
            task: Task to check dependencies for

        Returns:
            True if all dependencies are completed, False otherwise
        """
        if not task.dependencies:
            return True

        completed_task_ids = {
            result["task_id"]
            for result in self.current_state["completed_tasks"]
            if result["status"] == TaskStatus.COMPLETED.value
        }

        return all(dep_id in completed_task_ids for dep_id in task.dependencies)

    def _simulate_task_execution(self, task: Task, subagent) -> Dict[str, Any]:
        """
        Simulate task execution for testing purposes.

        In a real implementation, this would invoke the LLM with the subagent's prompt
        and tools to actually execute the task.

        Args:
            task: Task being executed
            subagent: SubAgent executing the task

        Returns:
            Dictionary with execution results
        """
        # Create mock execution based on task type
        if task.task_type == "web_testing":
            return {
                "status": TaskStatus.COMPLETED.value,
                "output": f"Completed web testing for: {task.description}. Found 2 minor UI issues and 1 functionality bug.",
                "artifacts": [f"test_report_{task.id}.md"],
                "execution_time": 45.2,
                "tool_usage": {"web_scrape": 3, "execute_code": 2},
            }
        elif task.task_type == "research":
            return {
                "status": TaskStatus.COMPLETED.value,
                "output": f"Research completed for: {task.description}. Gathered information from 5 sources.",
                "artifacts": [
                    f"research_report_{task.id}.md",
                    f"sources_{task.id}.json",
                ],
                "execution_time": 32.1,
                "tool_usage": {"search_internet": 4, "web_scrape": 5},
            }
        elif task.task_type == "analysis":
            return {
                "status": TaskStatus.COMPLETED.value,
                "output": f"Analysis completed for: {task.description}. Identified key patterns and recommendations.",
                "artifacts": [f"analysis_{task.id}.md"],
                "execution_time": 28.7,
                "tool_usage": {"execute_code": 3},
            }
        else:
            return {
                "status": TaskStatus.COMPLETED.value,
                "output": f"Task completed: {task.description}",
                "artifacts": [f"output_{task.id}.txt"],
                "execution_time": 15.0,
                "tool_usage": {"execute_code": 1},
            }

    def collect_results(self) -> Dict[str, Any]:
        """
        Collect and consolidate results from completed tasks.

        Returns:
            Dictionary with consolidated results and artifacts

        This method analyzes completed tasks and creates a comprehensive
        summary of the work done and artifacts produced.
        """
        if not self.current_state:
            return {"error": "No active state available"}

        completed_tasks = [
            dict_to_task_result(result)
            for result in self.current_state["completed_tasks"]
        ]

        if not completed_tasks:
            return {
                "message": "No completed tasks to collect results from",
                "completed_count": 0,
                "artifacts": [],
            }

        # Consolidate results
        total_execution_time = sum(task.execution_time for task in completed_tasks)
        all_artifacts = []
        tool_usage_summary = {}

        for task_result in completed_tasks:
            all_artifacts.extend(task_result.artifacts)
            for tool, count in task_result.tool_usage.items():
                tool_usage_summary[tool] = tool_usage_summary.get(tool, 0) + count

        # Create summary
        summary_parts = []
        for task_result in completed_tasks:
            summary_parts.append(f"- {task_result.task_id}: {task_result.output}")

        consolidated_summary = "\n".join(summary_parts)

        # End metrics collection session
        session_metrics = self.metrics_collector.end_session()

        # Get evaluation summary
        evaluation_summary = self.evaluation_manager.get_evaluation_summary()

        final_result = {
            "objective": self.current_state["user_objective"],
            "completed_tasks": len(completed_tasks),
            "total_execution_time": total_execution_time,
            "artifacts_created": all_artifacts,
            "tool_usage_summary": tool_usage_summary,
            "consolidated_summary": consolidated_summary,
            "iteration_count": self.current_state["iteration_count"],
            "session_metrics": session_metrics.to_dict() if session_metrics else None,
            "evaluation_summary": evaluation_summary,
            "langsmith_run_url": self.tracer.get_run_url(),
        }

        # End tracing session
        self.tracer.end_supervisor_run(final_result)

        return final_result

    def update_plan_from_results(self, results: List[TaskResult]) -> Dict[str, Any]:
        """
        Update the plan based on task results and current progress.

        Args:
            results: List of completed task results

        Returns:
            Dictionary with plan update information

        This method analyzes task results to determine if the plan needs
        modification, such as adding new tasks or adjusting priorities.
        """
        if not self.current_state or not results:
            return {"message": "No updates needed"}

        updates_made = []

        # Analyze results for plan adjustments
        for result in results:
            if result.status == TaskStatus.FAILED.value:
                # Create retry task for failed tasks
                retry_task = {
                    "id": f"retry_{result.task_id}_{uuid.uuid4().hex[:4]}",
                    "description": f"Retry failed task: {result.task_id}",
                    "task_type": "general",
                    "priority": 1,
                    "dependencies": [],
                    "required_tools": ["execute_code"],
                    "success_criteria": "Successfully complete the previously failed task",
                    "context": {
                        "retry_of": result.task_id,
                        "error": result.error_message,
                    },
                }

                self.current_state["todo_list"].append(retry_task)
                updates_made.append(
                    f"Added retry task for failed task {result.task_id}"
                )

            elif "bug" in result.output.lower() or "issue" in result.output.lower():
                # If bugs were found, consider adding follow-up tasks
                if not any(
                    "follow-up" in task["description"]
                    for task in self.current_state["todo_list"]
                ):
                    followup_task = {
                        "id": f"followup_{result.task_id}_{uuid.uuid4().hex[:4]}",
                        "description": f"Follow-up investigation for issues found in {result.task_id}",
                        "task_type": "analysis",
                        "priority": 2,
                        "dependencies": [result.task_id],
                        "required_tools": ["execute_code"],
                        "success_criteria": "Provide detailed analysis of identified issues",
                        "context": {"followup_to": result.task_id},
                    }

                    self.current_state["todo_list"].append(followup_task)
                    updates_made.append(
                        f"Added follow-up task for issues found in {result.task_id}"
                    )

        return {
            "updates_made": updates_made,
            "plan_modified": len(updates_made) > 0,
            "current_todo_count": len(self.current_state["todo_list"]),
        }

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next task that is ready to be executed.

        Returns:
            Dictionary with next task information or None if no tasks are ready

        This method finds the highest priority task that has all its
        dependencies completed and is not already in progress.
        """
        if not self.current_state:
            return None

        ready_tasks = []

        for task_dict in self.current_state["todo_list"]:
            if task_dict["status"] == TaskStatus.PENDING.value:
                task = dict_to_task(task_dict)
                if self._are_dependencies_completed(task):
                    ready_tasks.append(task_dict)

        if not ready_tasks:
            return None

        # Sort by priority (lower number = higher priority)
        ready_tasks.sort(key=lambda t: t["priority"])

        return ready_tasks[0]

    def is_objective_complete(self) -> bool:
        """
        Check if the objective has been completed.

        Returns:
            True if all tasks are completed, False otherwise
        """
        if not self.current_state:
            return False

        for task_dict in self.current_state["todo_list"]:
            if task_dict["status"] not in [
                TaskStatus.COMPLETED.value,
                TaskStatus.FAILED.value,
            ]:
                return False

        return True

    def get_supervisor_tools(self) -> Dict[str, Any]:
        """
        Get the tools available to the supervisor.

        Returns:
            Dictionary of supervisor tools
        """
        # Get shared file tools
        shared_tools = self.tool_registry.get_agent_tools("supervisor")

        # Add supervisor-specific tools
        supervisor_tools = {
            "update_todo": self.update_todo_tool,
            "task_tool": self.task_tool,
        }

        # Combine with shared tools
        supervisor_tools.update(shared_tools)

        return supervisor_tools

    def initialize_state(self, objective: str) -> AgentState:
        """
        Initialize a new agent state for the given objective.

        Args:
            objective: The user objective to work on

        Returns:
            Initialized AgentState
        """
        # Start tracing session
        session_id = self.tracer.start_supervisor_run(objective)

        # Start metrics collection session
        self.metrics_collector.start_session(session_id, objective)

        # Create tool whitelist for supervisor
        self.tool_registry.create_tool_whitelist("supervisor", [])

        # Initialize state
        self.current_state = {
            "user_objective": objective,
            "todo_list": [],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": self.vfs.files.copy(),
            "iteration_count": 0,
            "final_result": None,
        }

        return self.current_state

    def get_state(self) -> Optional[AgentState]:
        """Get the current agent state."""
        return self.current_state

    def set_state(self, state: AgentState) -> None:
        """Set the agent state."""
        self.current_state = state

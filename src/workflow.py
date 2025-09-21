"""LangGraph workflow orchestration for the Supervisor Agent system."""

from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from .models.data_models import AgentState, TaskStatus, dict_to_task
from .agents.supervisor import SupervisorAgent
from .agents.subagent_factory import SubAgentFactory
from .tools.tool_registry import ToolRegistry
from .models.virtual_file_system import VirtualFileSystem
from .config import AgentConfig


class SupervisorWorkflow:
    """
    LangGraph workflow orchestration for the Supervisor Agent system.

    This class implements the complete workflow using LangGraph nodes and edges,
    managing state transitions, tool integration, and workflow routing logic.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the supervisor workflow.

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

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.llm_model,
            api_key=config.openai_api_key,
            temperature=0.1,
        )

        # Initialize studio debugging if available
        self.debug_mode = getattr(config, "debug_mode", False)
        self.studio_debugger = None

        if self.debug_mode:
            try:
                from .studio_config import StudioDebugger

                self.studio_debugger = StudioDebugger(self)
            except ImportError:
                print("Studio debugging not available")

        # Build the workflow graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph workflow with nodes and edges.

        Returns:
            Configured LangGraph workflow
        """
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("execute_task", self._execute_task_node)
        workflow.add_node("collect_results", self._collect_results_node)
        workflow.add_node("update_plan", self._update_plan_node)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add conditional edges
        workflow.add_conditional_edges(
            "supervisor",
            self._supervisor_router,
            {
                "create_plan": "supervisor",
                "execute_task": "execute_task",
                "collect_results": "collect_results",
                "finalize": "finalize",
            },
        )

        workflow.add_conditional_edges(
            "execute_task",
            self._task_execution_router,
            {
                "continue": "supervisor",
                "error": "update_plan",
                "complete": "collect_results",
            },
        )

        workflow.add_conditional_edges(
            "collect_results",
            self._results_router,
            {
                "continue": "update_plan",
                "complete": "finalize",
            },
        )

        workflow.add_conditional_edges(
            "update_plan",
            self._plan_update_router,
            {
                "continue": "supervisor",
                "complete": "finalize",
            },
        )

        # Finalize node leads to END
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _supervisor_node(self, state: AgentState) -> AgentState:
        """
        Supervisor node that manages plan creation and task coordination.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        # Debug logging for studio
        if self.studio_debugger:
            self.studio_debugger.log_execution(
                "supervisor",
                state,
                {
                    "action": "supervisor_coordination",
                    "has_plan": bool(state["todo_list"]),
                    "iteration": state["iteration_count"],
                },
            )
            self.studio_debugger.capture_state_snapshot(state, "supervisor_entry")

        # Set supervisor state
        self.supervisor.set_state(state)

        # If no plan exists, create one
        if not state["todo_list"]:
            if self.debug_mode:
                print(
                    f"🎯 Creating initial plan for objective: {state['user_objective'][:100]}..."
                )

            result = self.supervisor.update_todo_tool(
                "create", objective=state["user_objective"]
            )
            if "error" in result:
                state["final_result"] = f"Error creating plan: {result['error']}"
                return state

            if self.debug_mode:
                print(f"📋 Created plan with {len(state['todo_list'])} tasks")

        # Update iteration count
        state["iteration_count"] += 1

        # Update file system state
        state["file_system"] = dict(self.vfs.files)

        # Debug logging for studio
        if self.studio_debugger:
            self.studio_debugger.capture_state_snapshot(state, "supervisor_exit")

        return state

    def _execute_task_node(self, state: AgentState) -> AgentState:
        """
        Task execution node that runs subagents for specific tasks.

        Args:
            state: Current workflow state

        Returns:
            Updated state with task execution results
        """
        # Set supervisor state
        self.supervisor.set_state(state)

        # Get next task to execute
        next_task = self.supervisor.get_next_task()
        if not next_task:
            return state

        # Execute the task
        result = self.supervisor.task_tool(next_task["id"])

        # Update state with execution results
        if "error" in result:
            # Mark task as failed and log error
            for task_dict in state["todo_list"]:
                if task_dict["id"] == next_task["id"]:
                    task_dict["status"] = TaskStatus.FAILED.value
                    break

            # Log execution error in artifacts
            state["artifacts"]["execution_errors"] = state["artifacts"].get(
                "execution_errors", []
            )
            state["artifacts"]["execution_errors"].append(
                {
                    "task_id": next_task["id"],
                    "error": result["error"],
                    "iteration": state["iteration_count"],
                }
            )
        else:
            # Log successful execution
            state["artifacts"]["successful_executions"] = state["artifacts"].get(
                "successful_executions", []
            )
            state["artifacts"]["successful_executions"].append(
                {
                    "task_id": next_task["id"],
                    "output": result.get("output", ""),
                    "artifacts": result.get("artifacts", []),
                    "execution_time": result.get("execution_time", 0),
                    "iteration": state["iteration_count"],
                }
            )

        # Update iteration count
        state["iteration_count"] += 1

        # Update file system state
        state["file_system"] = dict(self.vfs.files)

        return state

    def _collect_results_node(self, state: AgentState) -> AgentState:
        """
        Results collection node that consolidates completed work.

        Args:
            state: Current workflow state

        Returns:
            Updated state with collected results
        """
        # Set supervisor state
        self.supervisor.set_state(state)

        # Collect results from completed tasks
        results = self.supervisor.collect_results()

        # Update artifacts in state
        state["artifacts"].update(
            {
                "consolidated_results": results,
                "collection_timestamp": results.get("timestamp"),
                "collection_iteration": state["iteration_count"],
            }
        )

        # Update file system state
        state["file_system"] = dict(self.vfs.files)

        return state

    def _update_plan_node(self, state: AgentState) -> AgentState:
        """
        Plan update node that modifies the plan based on results.

        Args:
            state: Current workflow state

        Returns:
            Updated state with plan modifications
        """
        # Set supervisor state
        self.supervisor.set_state(state)

        # Get completed task results for analysis
        completed_results = [
            result
            for result in state["completed_tasks"]
            if result["status"] == TaskStatus.COMPLETED.value
        ]

        if completed_results:
            # Convert to TaskResult objects for analysis
            from .models.data_models import dict_to_task_result

            task_results = [dict_to_task_result(result) for result in completed_results]

            # Update plan based on results
            update_result = self.supervisor.update_plan_from_results(task_results)

            # Log plan updates in artifacts
            if update_result.get("plan_modified"):
                state["artifacts"]["plan_updates"] = state["artifacts"].get(
                    "plan_updates", []
                )
                state["artifacts"]["plan_updates"].append(update_result)

        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        """
        Finalization node that prepares final results.

        Args:
            state: Current workflow state

        Returns:
            Final state with consolidated results
        """
        # Set supervisor state
        self.supervisor.set_state(state)

        # Generate final consolidated results
        final_results = self.supervisor.collect_results()

        # Determine completion reason
        completion_reason = "objective_completed"
        if state["iteration_count"] >= self.config.max_iterations:
            completion_reason = "max_iterations_reached"
        elif any(
            task.get("status") == TaskStatus.FAILED.value for task in state["todo_list"]
        ):
            failed_count = sum(
                1
                for task in state["todo_list"]
                if task.get("status") == TaskStatus.FAILED.value
            )
            total_count = len(state["todo_list"])
            if failed_count == total_count:
                completion_reason = "all_tasks_failed"
            elif failed_count > total_count / 2:
                completion_reason = "majority_tasks_failed"

        # Create comprehensive final result summary
        summary_parts = [
            f"=== SUPERVISOR AGENT EXECUTION SUMMARY ===",
            f"Objective: {state['user_objective']}",
            f"Completion Reason: {completion_reason.replace('_', ' ').title()}",
            f"Total Iterations: {state['iteration_count']}",
            f"Tasks Completed: {final_results.get('completed_tasks', 0)}",
            f"Total Execution Time: {final_results.get('total_execution_time', 0):.1f}s",
            f"Artifacts Created: {len(final_results.get('artifacts_created', []))}",
            "",
        ]

        # Add task breakdown
        completed_tasks = [
            t
            for t in state["todo_list"]
            if t.get("status") == TaskStatus.COMPLETED.value
        ]
        failed_tasks = [
            t for t in state["todo_list"] if t.get("status") == TaskStatus.FAILED.value
        ]
        pending_tasks = [
            t for t in state["todo_list"] if t.get("status") == TaskStatus.PENDING.value
        ]

        summary_parts.extend(
            [
                f"Task Breakdown:",
                f"- Completed: {len(completed_tasks)}",
                f"- Failed: {len(failed_tasks)}",
                f"- Pending: {len(pending_tasks)}",
                "",
            ]
        )

        # Add execution summary
        if final_results.get("consolidated_summary"):
            summary_parts.extend(
                [
                    "Execution Summary:",
                    final_results["consolidated_summary"],
                    "",
                ]
            )

        # Add artifacts list
        if final_results.get("artifacts_created"):
            summary_parts.extend(
                [
                    "Artifacts Created:",
                    *[
                        f"- {artifact}"
                        for artifact in final_results["artifacts_created"]
                    ],
                    "",
                ]
            )

        # Add error summary if any
        if state["artifacts"].get("execution_errors"):
            summary_parts.extend(
                [
                    "Execution Errors:",
                    *[
                        f"- Task {err['task_id']}: {err['error']}"
                        for err in state["artifacts"]["execution_errors"]
                    ],
                    "",
                ]
            )

        # Add performance metrics if available
        if final_results.get("session_metrics"):
            metrics = final_results["session_metrics"]
            summary_parts.extend(
                [
                    "Performance Metrics:",
                    f"- Average Task Execution Time: {metrics.get('avg_task_execution_time', 0):.1f}s",
                    f"- Tool Usage Count: {metrics.get('total_tool_usage', 0)}",
                    f"- Success Rate: {metrics.get('success_rate', 0):.1%}",
                    "",
                ]
            )

        # Add LangSmith tracing info if available
        if final_results.get("langsmith_run_url"):
            summary_parts.extend(
                [
                    f"LangSmith Trace: {final_results['langsmith_run_url']}",
                    "",
                ]
            )

        # Final status
        if completion_reason == "objective_completed":
            summary_parts.append("✅ Objective completed successfully!")
        elif completion_reason == "max_iterations_reached":
            summary_parts.append("⚠️ Workflow stopped due to maximum iterations limit.")
        else:
            summary_parts.append("❌ Workflow completed with errors.")

        state["final_result"] = "\n".join(summary_parts)

        # Update final file system state
        state["file_system"] = dict(self.vfs.files)

        # Add completion metadata to artifacts
        state["artifacts"]["completion_metadata"] = {
            "completion_reason": completion_reason,
            "final_iteration": state["iteration_count"],
            "completion_timestamp": final_results.get("timestamp"),
            "total_tasks": len(state["todo_list"]),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "pending_tasks": len(pending_tasks),
        }

        return state

    def _supervisor_router(
        self, state: AgentState
    ) -> Literal["create_plan", "execute_task", "collect_results", "finalize"]:
        """
        Route supervisor decisions based on current state.

        Args:
            state: Current workflow state

        Returns:
            Next node to execute
        """
        # Check if we've exceeded max iterations
        if state["iteration_count"] >= self.config.max_iterations:
            return "finalize"

        # If no plan exists, create one (but only on first iteration)
        if not state["todo_list"] and state["iteration_count"] == 0:
            return "create_plan"

        # Set supervisor state for decision making
        self.supervisor.set_state(state)

        # Check if objective is complete
        if self.supervisor.is_objective_complete():
            return "collect_results"

        # Check if there are tasks ready to execute
        next_task = self.supervisor.get_next_task()
        if next_task:
            return "execute_task"

        # Check if all remaining tasks are blocked or failed
        pending_tasks = [
            t for t in state["todo_list"] if t.get("status") == TaskStatus.PENDING.value
        ]
        if not pending_tasks:
            # No pending tasks, collect results and finalize
            return "collect_results"

        # If we have pending tasks but none are ready, collect results for analysis
        return "collect_results"

    def _task_execution_router(
        self, state: AgentState
    ) -> Literal["continue", "error", "complete"]:
        """
        Route task execution results.

        Args:
            state: Current workflow state

        Returns:
            Next action based on execution results
        """
        # Check if we've exceeded max iterations
        if state["iteration_count"] >= self.config.max_iterations:
            return "complete"

        # Check if there are failed tasks that need attention
        failed_tasks = [
            task
            for task in state["todo_list"]
            if task["status"] == TaskStatus.FAILED.value
        ]

        if failed_tasks:
            return "error"

        # Check if objective is complete
        self.supervisor.set_state(state)
        if self.supervisor.is_objective_complete():
            return "complete"

        return "continue"

    def _results_router(self, state: AgentState) -> Literal["continue", "complete"]:
        """
        Route results collection decisions.

        Args:
            state: Current workflow state

        Returns:
            Next action based on results analysis
        """
        # Check if objective is complete
        self.supervisor.set_state(state)
        if self.supervisor.is_objective_complete():
            return "complete"

        # Check if we've exceeded max iterations
        if state["iteration_count"] >= self.config.max_iterations:
            return "complete"

        return "continue"

    def _plan_update_router(self, state: AgentState) -> Literal["continue", "complete"]:
        """
        Route plan update decisions.

        Args:
            state: Current workflow state

        Returns:
            Next action based on plan updates
        """
        # Check if objective is complete
        self.supervisor.set_state(state)
        if self.supervisor.is_objective_complete():
            return "complete"

        # Check if we've exceeded max iterations
        if state["iteration_count"] >= self.config.max_iterations:
            return "complete"

        return "continue"

    def run(self, objective: str) -> Dict[str, Any]:
        """
        Run the complete workflow for a given objective.

        Args:
            objective: The user objective to accomplish

        Returns:
            Final workflow results
        """
        # Initialize state
        initial_state = self.supervisor.initialize_state(objective)

        # Run the workflow
        final_state = self.graph.invoke(initial_state)

        return {
            "objective": objective,
            "final_result": final_state.get("final_result"),
            "iterations": final_state.get("iteration_count", 0),
            "completed_tasks": len(final_state.get("completed_tasks", [])),
            "artifacts": final_state.get("artifacts", {}),
            "file_system": final_state.get("file_system", {}),
        }

    async def arun(self, objective: str) -> Dict[str, Any]:
        """
        Run the workflow asynchronously.

        Args:
            objective: The user objective to accomplish

        Returns:
            Final workflow results
        """
        # Initialize state
        initial_state = self.supervisor.initialize_state(objective)

        # Run the workflow asynchronously
        final_state = await self.graph.ainvoke(initial_state)

        return {
            "objective": objective,
            "final_result": final_state.get("final_result"),
            "iterations": final_state.get("iteration_count", 0),
            "completed_tasks": len(final_state.get("completed_tasks", [])),
            "artifacts": final_state.get("artifacts", {}),
            "file_system": final_state.get("file_system", {}),
        }

    def get_graph_config(self) -> Dict[str, Any]:
        """
        Get configuration for LangGraph Studio.

        Returns:
            Graph configuration dictionary with enhanced visualization and debugging
        """
        return {
            "graph": self.graph,
            "config": {
                "configurable": {
                    "model": self.config.llm_model,
                    "max_iterations": self.config.max_iterations,
                    "max_subagents": self.config.max_subagents,
                    "enable_tracing": self.config.enable_tracing,
                    "debug_mode": getattr(self.config, "debug_mode", False),
                    "tool_timeout": self.config.tool_timeout,
                }
            },
            "metadata": {
                "name": "Supervisor Agent Workflow",
                "description": "LangGraph workflow for supervisor-subagent coordination",
                "version": "1.0.0",
                "nodes": {
                    "supervisor": {
                        "label": "🎯 Supervisor",
                        "description": "Central coordinator managing plan creation and task delegation",
                        "color": "#4CAF50",
                        "icon": "supervisor_account",
                    },
                    "execute_task": {
                        "label": "⚡ Execute Task",
                        "description": "Creates and runs subagents for specific tasks",
                        "color": "#2196F3",
                        "icon": "play_arrow",
                    },
                    "collect_results": {
                        "label": "📊 Collect Results",
                        "description": "Consolidates completed work and analyzes progress",
                        "color": "#FF9800",
                        "icon": "assessment",
                    },
                    "update_plan": {
                        "label": "📝 Update Plan",
                        "description": "Modifies the plan based on execution results",
                        "color": "#9C27B0",
                        "icon": "edit",
                    },
                    "finalize": {
                        "label": "✅ Finalize",
                        "description": "Prepares final results and completion summary",
                        "color": "#4CAF50",
                        "icon": "check_circle",
                    },
                },
                "state_schema": {
                    "user_objective": {
                        "type": "string",
                        "description": "The original user objective",
                        "display_priority": 1,
                    },
                    "todo_list": {
                        "type": "array",
                        "description": "Current list of tasks to be executed",
                        "display_priority": 2,
                    },
                    "completed_tasks": {
                        "type": "array",
                        "description": "List of completed task results",
                        "display_priority": 3,
                    },
                    "current_task": {
                        "type": "object",
                        "description": "Currently executing task",
                        "display_priority": 4,
                    },
                    "artifacts": {
                        "type": "object",
                        "description": "Generated artifacts and execution metadata",
                        "display_priority": 5,
                    },
                    "file_system": {
                        "type": "object",
                        "description": "Virtual file system state",
                        "display_priority": 6,
                    },
                    "iteration_count": {
                        "type": "integer",
                        "description": "Current workflow iteration number",
                        "display_priority": 7,
                    },
                    "final_result": {
                        "type": "string",
                        "description": "Final consolidated results",
                        "display_priority": 8,
                    },
                },
            },
        }

    def visualize_graph(self) -> str:
        """
        Generate a visual representation of the workflow graph.

        Returns:
            Mermaid diagram string
        """
        return """
        graph TD
            A[Start] --> B[Supervisor]
            B --> C{Has Plan?}
            C -->|No| D[Create Plan]
            D --> B
            C -->|Yes| E{Tasks Ready?}
            E -->|Yes| F[Execute Task]
            E -->|No| G[Collect Results]
            F --> H{Task Success?}
            H -->|Yes| I{More Tasks?}
            H -->|No| J[Update Plan]
            I -->|Yes| B
            I -->|No| G
            J --> B
            G --> K{Complete?}
            K -->|Yes| L[Finalize]
            K -->|No| J
            L --> M[End]
        """

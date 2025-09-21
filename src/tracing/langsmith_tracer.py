"""LangSmith tracing integration for supervisor and subagent operations."""

import os
import functools
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import uuid
import json

from langsmith import Client
from langsmith.run_helpers import traceable

from ..config import AgentConfig


class LangSmithTracer:
    """
    Handles LangSmith tracing configuration and decorators for supervisor and subagent operations.

    This class provides:
    - LangSmith client configuration
    - Tracing decorators for supervisor operations
    - Tracing decorators for subagent operations
    - Run context management
    - Metadata collection and tagging
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the LangSmith tracer.

        Args:
            config: Agent configuration containing LangSmith settings
        """
        self.config = config
        self.client: Optional[Client] = None
        self.current_run_id: Optional[str] = None
        self.supervisor_run_id: Optional[str] = None

        if config.enable_tracing and config.langsmith_api_key:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the LangSmith client with proper configuration."""
        try:
            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.config.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.config.langsmith_project

            # Initialize client
            self.client = Client(
                api_key=self.config.langsmith_api_key,
                api_url=os.getenv(
                    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
                ),
            )

            # Verify connection
            self.client.read_project(project_name=self.config.langsmith_project)

        except Exception as e:
            print(f"Warning: Failed to initialize LangSmith client: {e}")
            self.client = None

    def is_enabled(self) -> bool:
        """Check if tracing is enabled and properly configured."""
        return self.config.enable_tracing and self.client is not None

    def supervisor_trace(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Decorator for tracing supervisor operations.

        Args:
            operation_name: Name of the supervisor operation
            metadata: Additional metadata to include in the trace

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            if not self.is_enabled():
                return func

            @traceable(
                name=f"supervisor_{operation_name}",
                project_name=self.config.langsmith_project,
                tags=["supervisor", operation_name],
                metadata=metadata or {},
            )
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Set supervisor run context
                if not self.supervisor_run_id:
                    self.supervisor_run_id = str(uuid.uuid4())

                # Add supervisor context to metadata
                trace_metadata = {
                    "supervisor_run_id": self.supervisor_run_id,
                    "operation": operation_name,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                }

                try:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Log successful completion
                    if self.client:
                        self._log_supervisor_event(
                            operation_name, "completed", trace_metadata, result
                        )

                    return result

                except Exception as e:
                    # Log error
                    if self.client:
                        self._log_supervisor_event(
                            operation_name, "error", trace_metadata, {"error": str(e)}
                        )
                    raise

            return wrapper

        return decorator

    def subagent_trace(
        self, agent_id: str, task_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Decorator for tracing subagent operations.

        Args:
            agent_id: ID of the subagent
            task_id: ID of the task being executed
            metadata: Additional metadata to include in the trace

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            if not self.is_enabled():
                return func

            @traceable(
                name=f"subagent_execution",
                project_name=self.config.langsmith_project,
                tags=["subagent", "task_execution", task_id],
                metadata=metadata or {},
                parent_run_id=self.supervisor_run_id,
            )
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Add subagent context to metadata
                trace_metadata = {
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "supervisor_run_id": self.supervisor_run_id,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                }

                try:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Log successful completion
                    if self.client:
                        self._log_subagent_event(
                            agent_id, task_id, "completed", trace_metadata, result
                        )

                    return result

                except Exception as e:
                    # Log error
                    if self.client:
                        self._log_subagent_event(
                            agent_id,
                            task_id,
                            "error",
                            trace_metadata,
                            {"error": str(e)},
                        )
                    raise

            return wrapper

        return decorator

    def tool_trace(
        self,
        tool_name: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator for tracing tool usage.

        Args:
            tool_name: Name of the tool being used
            agent_id: ID of the agent using the tool (if applicable)
            metadata: Additional metadata to include in the trace

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            if not self.is_enabled():
                return func

            @traceable(
                name=f"tool_{tool_name}",
                project_name=self.config.langsmith_project,
                tags=["tool", tool_name, agent_id or "supervisor"],
                metadata=metadata or {},
                parent_run_id=self.current_run_id or self.supervisor_run_id,
            )
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Add tool context to metadata
                trace_metadata = {
                    "tool_name": tool_name,
                    "agent_id": agent_id or "supervisor",
                    "supervisor_run_id": self.supervisor_run_id,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                }

                start_time = datetime.now()

                try:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Calculate execution time
                    execution_time = (datetime.now() - start_time).total_seconds()

                    # Log successful completion
                    if self.client:
                        self._log_tool_event(
                            tool_name,
                            agent_id,
                            "completed",
                            trace_metadata,
                            result,
                            execution_time,
                        )

                    return result

                except Exception as e:
                    # Calculate execution time
                    execution_time = (datetime.now() - start_time).total_seconds()

                    # Log error
                    if self.client:
                        self._log_tool_event(
                            tool_name,
                            agent_id,
                            "error",
                            trace_metadata,
                            {"error": str(e)},
                            execution_time,
                        )
                    raise

            return wrapper

        return decorator

    def start_supervisor_run(self, objective: str) -> str:
        """
        Start a new supervisor run for the given objective.

        Args:
            objective: The user objective being worked on

        Returns:
            Run ID for the supervisor run
        """
        self.supervisor_run_id = str(uuid.uuid4())

        if self.client:
            try:
                # Create a run for the entire supervisor session
                run = self.client.create_run(
                    name="supervisor_session",
                    project_name=self.config.langsmith_project,
                    run_type="chain",
                    inputs={"objective": objective},
                    tags=["supervisor", "session"],
                    extra={
                        "supervisor_run_id": self.supervisor_run_id,
                        "objective": objective,
                        "start_time": datetime.now().isoformat(),
                    },
                )
                self.current_run_id = str(run.id)
            except Exception as e:
                print(f"Warning: Failed to create supervisor run: {e}")

        return self.supervisor_run_id

    def end_supervisor_run(self, final_result: Dict[str, Any]) -> None:
        """
        End the current supervisor run.

        Args:
            final_result: Final result of the supervisor session
        """
        if self.client and self.current_run_id:
            try:
                self.client.update_run(
                    run_id=self.current_run_id,
                    outputs=final_result,
                    end_time=datetime.now(),
                    extra={
                        "end_time": datetime.now().isoformat(),
                        "final_result": final_result,
                    },
                )
            except Exception as e:
                print(f"Warning: Failed to end supervisor run: {e}")

        # Reset run context
        self.supervisor_run_id = None
        self.current_run_id = None

    def _log_supervisor_event(
        self, operation: str, status: str, metadata: Dict[str, Any], result: Any
    ) -> None:
        """Log a supervisor event to LangSmith."""
        try:
            event_data = {
                "operation": operation,
                "status": status,
                "metadata": metadata,
                "result": self._serialize_result(result),
                "timestamp": datetime.now().isoformat(),
            }

            # In a real implementation, this would use LangSmith's event logging
            # For now, we'll use the feedback API as a proxy
            if hasattr(self.client, "create_feedback"):
                self.client.create_feedback(
                    run_id=self.current_run_id,
                    key=f"supervisor_{operation}",
                    score=1.0 if status == "completed" else 0.0,
                    value=json.dumps(event_data),
                )
        except Exception as e:
            print(f"Warning: Failed to log supervisor event: {e}")

    def _log_subagent_event(
        self,
        agent_id: str,
        task_id: str,
        status: str,
        metadata: Dict[str, Any],
        result: Any,
    ) -> None:
        """Log a subagent event to LangSmith."""
        try:
            event_data = {
                "agent_id": agent_id,
                "task_id": task_id,
                "status": status,
                "metadata": metadata,
                "result": self._serialize_result(result),
                "timestamp": datetime.now().isoformat(),
            }

            # In a real implementation, this would use LangSmith's event logging
            if hasattr(self.client, "create_feedback"):
                self.client.create_feedback(
                    run_id=self.current_run_id,
                    key=f"subagent_{agent_id}",
                    score=1.0 if status == "completed" else 0.0,
                    value=json.dumps(event_data),
                )
        except Exception as e:
            print(f"Warning: Failed to log subagent event: {e}")

    def _log_tool_event(
        self,
        tool_name: str,
        agent_id: Optional[str],
        status: str,
        metadata: Dict[str, Any],
        result: Any,
        execution_time: float,
    ) -> None:
        """Log a tool usage event to LangSmith."""
        try:
            event_data = {
                "tool_name": tool_name,
                "agent_id": agent_id or "supervisor",
                "status": status,
                "execution_time": execution_time,
                "metadata": metadata,
                "result": self._serialize_result(result),
                "timestamp": datetime.now().isoformat(),
            }

            # In a real implementation, this would use LangSmith's event logging
            if hasattr(self.client, "create_feedback"):
                self.client.create_feedback(
                    run_id=self.current_run_id,
                    key=f"tool_{tool_name}",
                    score=1.0 if status == "completed" else 0.0,
                    value=json.dumps(event_data),
                )
        except Exception as e:
            print(f"Warning: Failed to log tool event: {e}")

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """Serialize result for logging, handling complex objects."""
        try:
            if isinstance(result, dict):
                return result
            elif isinstance(result, (str, int, float, bool)):
                return {"value": result}
            elif hasattr(result, "__dict__"):
                return {"object": str(result), "type": type(result).__name__}
            else:
                return {"value": str(result), "type": type(result).__name__}
        except Exception:
            return {"value": "Unable to serialize result", "type": "unknown"}

    def get_run_url(self) -> Optional[str]:
        """Get the LangSmith URL for the current run."""
        if not self.current_run_id or not self.config.langsmith_project:
            return None

        base_url = os.getenv("LANGCHAIN_ENDPOINT", "https://smith.langchain.com")
        return f"{base_url}/o/default/projects/p/{self.config.langsmith_project}/r/{self.current_run_id}"

    def add_tags(self, tags: List[str]) -> None:
        """Add tags to the current run."""
        if self.client and self.current_run_id:
            try:
                # Get current run
                run = self.client.read_run(self.current_run_id)
                existing_tags = run.tags or []

                # Add new tags
                updated_tags = list(set(existing_tags + tags))

                # Update run
                self.client.update_run(run_id=self.current_run_id, tags=updated_tags)
            except Exception as e:
                print(f"Warning: Failed to add tags: {e}")

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to the current run."""
        if self.client and self.current_run_id:
            try:
                # Get current run
                run = self.client.read_run(self.current_run_id)
                existing_extra = run.extra or {}

                # Merge metadata
                updated_extra = {**existing_extra, **metadata}

                # Update run
                self.client.update_run(run_id=self.current_run_id, extra=updated_extra)
            except Exception as e:
                print(f"Warning: Failed to add metadata: {e}")

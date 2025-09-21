"""LangGraph Studio configuration and debugging utilities."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .workflow import SupervisorWorkflow
from .config import AgentConfig


class StudioDebugger:
    """Enhanced debugging utilities for LangGraph Studio."""

    def __init__(self, workflow: SupervisorWorkflow):
        """
        Initialize the studio debugger.

        Args:
            workflow: The supervisor workflow instance
        """
        self.workflow = workflow
        self.execution_log: List[Dict[str, Any]] = []
        self.breakpoints: Dict[str, bool] = {}
        self.state_snapshots: List[Dict[str, Any]] = []

    def add_breakpoint(self, node_name: str) -> None:
        """
        Add a breakpoint to a specific node.

        Args:
            node_name: Name of the node to add breakpoint to
        """
        self.breakpoints[node_name] = True

    def remove_breakpoint(self, node_name: str) -> None:
        """
        Remove a breakpoint from a specific node.

        Args:
            node_name: Name of the node to remove breakpoint from
        """
        self.breakpoints.pop(node_name, None)

    def log_execution(
        self,
        node_name: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log execution details for debugging.

        Args:
            node_name: Name of the executing node
            state: Current state
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "iteration": state.get("iteration_count", 0),
            "state_summary": self._create_state_summary(state),
            "metadata": metadata or {},
        }
        self.execution_log.append(log_entry)

    def capture_state_snapshot(self, state: Dict[str, Any], label: str = "") -> None:
        """
        Capture a snapshot of the current state.

        Args:
            state: Current state to snapshot
            label: Optional label for the snapshot
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "iteration": state.get("iteration_count", 0),
            "state": self._sanitize_state_for_json(state),
        }
        self.state_snapshots.append(snapshot)

        # Keep only the last 20 snapshots to prevent memory issues
        if len(self.state_snapshots) > 20:
            self.state_snapshots = self.state_snapshots[-20:]

    def _create_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a concise summary of the state for logging.

        Args:
            state: Current state

        Returns:
            Summarized state information
        """
        return {
            "objective": state.get("user_objective", "")[:100] + "..."
            if len(state.get("user_objective", "")) > 100
            else state.get("user_objective", ""),
            "todo_count": len(state.get("todo_list", [])),
            "completed_count": len(state.get("completed_tasks", [])),
            "current_task_id": state.get("current_task", {}).get("id")
            if state.get("current_task")
            else None,
            "file_count": len(state.get("file_system", {})),
            "iteration": state.get("iteration_count", 0),
        }

    def _sanitize_state_for_json(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize state for JSON serialization.

        Args:
            state: State to sanitize

        Returns:
            JSON-serializable state
        """
        sanitized = {}
        for key, value in state.items():
            try:
                json.dumps(value)  # Test if serializable
                sanitized[key] = value
            except (TypeError, ValueError):
                sanitized[key] = str(value)  # Convert to string if not serializable
        return sanitized

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the execution for debugging.

        Returns:
            Execution summary
        """
        return {
            "total_executions": len(self.execution_log),
            "nodes_executed": list(set(entry["node"] for entry in self.execution_log)),
            "execution_timeline": self.execution_log[-10:],  # Last 10 entries
            "active_breakpoints": [
                node for node, active in self.breakpoints.items() if active
            ],
            "snapshot_count": len(self.state_snapshots),
        }


class StudioVisualizer:
    """Enhanced visualization utilities for LangGraph Studio."""

    @staticmethod
    def generate_mermaid_diagram(workflow: SupervisorWorkflow) -> str:
        """
        Generate an enhanced Mermaid diagram for the workflow.

        Args:
            workflow: The supervisor workflow instance

        Returns:
            Mermaid diagram string with enhanced styling
        """
        return """
        graph TD
            %% Node Definitions with Styling
            A[🎯 Supervisor<br/>Plan & Coordinate] --> B{📋 Has Plan?}
            B -->|No| C[📝 Create Plan<br/>Decompose Objective]
            C --> A
            B -->|Yes| D{⚡ Tasks Ready?}
            D -->|Yes| E[🤖 Execute Task<br/>Create Subagent]
            D -->|No| F[📊 Collect Results<br/>Analyze Progress]
            E --> G{✅ Task Success?}
            G -->|Yes| H{🔄 More Tasks?}
            G -->|No| I[📝 Update Plan<br/>Handle Failures]
            H -->|Yes| A
            H -->|No| F
            I --> A
            F --> J{🎯 Complete?}
            J -->|Yes| K[✅ Finalize<br/>Generate Summary]
            J -->|No| I
            K --> L[🏁 End]

            %% Styling
            classDef supervisorNode fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
            classDef executionNode fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
            classDef analysisNode fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff
            classDef planningNode fill:#9C27B0,stroke:#6A1B9A,stroke-width:2px,color:#fff
            classDef finalNode fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
            classDef decisionNode fill:#FFC107,stroke:#F57F17,stroke-width:2px,color:#000

            class A supervisorNode
            class E executionNode
            class F analysisNode
            class C,I planningNode
            class K finalNode
            class B,D,G,H,J decisionNode
        """

    @staticmethod
    def generate_state_inspector_config() -> Dict[str, Any]:
        """
        Generate configuration for the state inspector.

        Returns:
            State inspector configuration
        """
        return {
            "inspector_panels": [
                {
                    "name": "Objective & Progress",
                    "fields": ["user_objective", "iteration_count"],
                    "display_mode": "summary",
                    "priority": 1,
                },
                {
                    "name": "Task Management",
                    "fields": ["todo_list", "current_task", "completed_tasks"],
                    "display_mode": "detailed",
                    "priority": 2,
                },
                {
                    "name": "File System",
                    "fields": ["file_system"],
                    "display_mode": "tree",
                    "priority": 3,
                },
                {
                    "name": "Artifacts & Logs",
                    "fields": ["artifacts", "subagent_logs"],
                    "display_mode": "expandable",
                    "priority": 4,
                },
                {
                    "name": "Final Results",
                    "fields": ["final_result"],
                    "display_mode": "formatted",
                    "priority": 5,
                },
            ],
            "field_formatters": {
                "todo_list": {
                    "type": "task_list",
                    "show_status": True,
                    "show_dependencies": True,
                },
                "file_system": {
                    "type": "file_tree",
                    "show_content_preview": True,
                    "max_preview_length": 200,
                },
                "artifacts": {"type": "key_value", "expandable": True},
                "final_result": {"type": "markdown", "syntax_highlighting": True},
            },
        }

    @staticmethod
    def generate_execution_timeline_config() -> Dict[str, Any]:
        """
        Generate configuration for the execution timeline.

        Returns:
            Execution timeline configuration
        """
        return {
            "timeline_settings": {
                "show_duration": True,
                "show_state_changes": True,
                "show_tool_calls": True,
                "group_by_iteration": True,
                "highlight_errors": True,
            },
            "event_types": {
                "node_execution": {
                    "color": "#2196F3",
                    "icon": "play_arrow",
                    "show_details": True,
                },
                "tool_call": {
                    "color": "#FF9800",
                    "icon": "build",
                    "show_details": True,
                },
                "state_update": {
                    "color": "#4CAF50",
                    "icon": "update",
                    "show_details": False,
                },
                "error": {
                    "color": "#F44336",
                    "icon": "error",
                    "show_details": True,
                    "highlight": True,
                },
            },
        }


def create_studio_config(config: AgentConfig) -> Dict[str, Any]:
    """
    Create comprehensive LangGraph Studio configuration.

    Args:
        config: Agent configuration

    Returns:
        Complete studio configuration
    """
    workflow = SupervisorWorkflow(config)
    visualizer = StudioVisualizer()

    return {
        "graph_config": workflow.get_graph_config(),
        "visualization": {
            "mermaid_diagram": visualizer.generate_mermaid_diagram(workflow),
            "state_inspector": visualizer.generate_state_inspector_config(),
            "execution_timeline": visualizer.generate_execution_timeline_config(),
        },
        "debugging": {
            "enable_breakpoints": True,
            "enable_step_through": True,
            "enable_state_snapshots": True,
            "log_level": "INFO",
            "capture_intermediate_states": True,
        },
        "environments": {
            "development": {
                "description": "Development environment with full debugging",
                "config": {
                    "debug_mode": True,
                    "max_iterations": 5,
                    "enable_tracing": True,
                    "tool_timeout": 60,
                },
            },
            "testing": {
                "description": "Testing environment with controlled execution",
                "config": {
                    "debug_mode": False,
                    "max_iterations": 3,
                    "max_subagents": 2,
                    "enable_tracing": False,
                    "tool_timeout": 30,
                },
            },
            "production": {
                "description": "Production environment optimized for performance",
                "config": {
                    "debug_mode": False,
                    "max_iterations": 20,
                    "max_subagents": 10,
                    "enable_tracing": True,
                    "tool_timeout": 120,
                },
            },
        },
    }


def export_studio_config(
    config: AgentConfig, output_path: str = "studio_config.json"
) -> None:
    """
    Export studio configuration to a JSON file.

    Args:
        config: Agent configuration
        output_path: Path to save the configuration file
    """
    studio_config = create_studio_config(config)

    with open(output_path, "w") as f:
        json.dump(studio_config, f, indent=2, default=str)

    print(f"Studio configuration exported to {output_path}")

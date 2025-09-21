"""State corruption detection and recovery mechanisms."""

import copy
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass

from ..models.data_models import AgentState, TaskStatus
from .exceptions import StateCorruptionError, RecoveryError


@dataclass
class StateSnapshot:
    """Snapshot of agent state for recovery purposes."""

    timestamp: datetime
    state: Dict[str, Any]
    checksum: str
    iteration_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "state": self.state,
            "checksum": self.checksum,
            "iteration_count": self.iteration_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateSnapshot":
        """Create snapshot from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state=data["state"],
            checksum=data["checksum"],
            iteration_count=data["iteration_count"],
        )


class StateValidator:
    """Validates agent state integrity."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize state validator.

        Args:
            logger: Optional logger for validation events
        """
        self.logger = logger or logging.getLogger(__name__)

    def validate_state(self, state: AgentState) -> List[str]:
        """
        Validate agent state and return list of issues found.

        Args:
            state: Agent state to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        try:
            # Check required fields
            required_fields = [
                "user_objective",
                "todo_list",
                "completed_tasks",
                "artifacts",
                "subagent_logs",
                "file_system",
                "iteration_count",
            ]

            for field in required_fields:
                if field not in state:
                    issues.append(f"Missing required field: {field}")
                elif state[field] is None:
                    issues.append(f"Field {field} is None")

            # Validate field types
            if "todo_list" in state and not isinstance(state["todo_list"], list):
                issues.append("todo_list must be a list")

            if "completed_tasks" in state and not isinstance(
                state["completed_tasks"], list
            ):
                issues.append("completed_tasks must be a list")

            if "artifacts" in state and not isinstance(state["artifacts"], dict):
                issues.append("artifacts must be a dictionary")

            if "subagent_logs" in state and not isinstance(
                state["subagent_logs"], list
            ):
                issues.append("subagent_logs must be a list")

            if "file_system" in state and not isinstance(state["file_system"], dict):
                issues.append("file_system must be a dictionary")

            if "iteration_count" in state and not isinstance(
                state["iteration_count"], int
            ):
                issues.append("iteration_count must be an integer")

            # Validate task consistency
            if "todo_list" in state and "completed_tasks" in state:
                issues.extend(self._validate_task_consistency(state))

            # Validate task statuses
            if "todo_list" in state:
                issues.extend(self._validate_task_statuses(state["todo_list"]))

            # Validate dependencies
            if "todo_list" in state:
                issues.extend(self._validate_task_dependencies(state["todo_list"]))

        except Exception as e:
            issues.append(f"Validation error: {str(e)}")

        return issues

    def _validate_task_consistency(self, state: AgentState) -> List[str]:
        """Validate consistency between todo_list and completed_tasks."""
        issues = []

        try:
            todo_task_ids = {
                task.get("id") for task in state["todo_list"] if isinstance(task, dict)
            }
            completed_task_ids = {
                task.get("task_id")
                for task in state["completed_tasks"]
                if isinstance(task, dict)
            }

            # Check for duplicate task IDs in todo_list
            todo_ids_list = [
                task.get("id") for task in state["todo_list"] if isinstance(task, dict)
            ]
            if len(todo_ids_list) != len(set(todo_ids_list)):
                issues.append("Duplicate task IDs found in todo_list")

            # Check for orphaned completed tasks
            orphaned = completed_task_ids - todo_task_ids
            if orphaned:
                issues.append(f"Orphaned completed tasks: {orphaned}")

        except Exception as e:
            issues.append(f"Task consistency validation error: {str(e)}")

        return issues

    def _validate_task_statuses(self, todo_list: List[Dict[str, Any]]) -> List[str]:
        """Validate task status values."""
        issues = []

        valid_statuses = {status.value for status in TaskStatus}

        for i, task in enumerate(todo_list):
            if not isinstance(task, dict):
                issues.append(f"Task {i} is not a dictionary")
                continue

            status = task.get("status")
            if status and status not in valid_statuses:
                issues.append(f"Invalid status '{status}' for task {task.get('id', i)}")

        return issues

    def _validate_task_dependencies(self, todo_list: List[Dict[str, Any]]) -> List[str]:
        """Validate task dependencies."""
        issues = []

        try:
            task_ids = {task.get("id") for task in todo_list if isinstance(task, dict)}

            for task in todo_list:
                if not isinstance(task, dict):
                    continue

                dependencies = task.get("dependencies", [])
                if not isinstance(dependencies, list):
                    issues.append(
                        f"Dependencies for task {task.get('id')} must be a list"
                    )
                    continue

                # Check for invalid dependency references
                for dep_id in dependencies:
                    if dep_id not in task_ids:
                        issues.append(
                            f"Task {task.get('id')} has invalid dependency: {dep_id}"
                        )

                # Check for self-dependency
                if task.get("id") in dependencies:
                    issues.append(f"Task {task.get('id')} has self-dependency")

        except Exception as e:
            issues.append(f"Dependency validation error: {str(e)}")

        return issues


class StateRecoveryManager:
    """Manages state snapshots and recovery operations."""

    def __init__(
        self, max_snapshots: int = 10, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize state recovery manager.

        Args:
            max_snapshots: Maximum number of snapshots to keep
            logger: Optional logger for recovery events
        """
        self.max_snapshots = max_snapshots
        self.logger = logger or logging.getLogger(__name__)
        self.snapshots: List[StateSnapshot] = []
        self.validator = StateValidator(logger)

    def create_snapshot(self, state: AgentState) -> StateSnapshot:
        """
        Create a snapshot of the current state.

        Args:
            state: Current agent state

        Returns:
            StateSnapshot object
        """
        try:
            # Create deep copy of state
            state_copy = copy.deepcopy(state)

            # Generate checksum for integrity verification
            state_json = json.dumps(state_copy, sort_keys=True, default=str)
            checksum = str(hash(state_json))

            snapshot = StateSnapshot(
                timestamp=datetime.now(),
                state=state_copy,
                checksum=checksum,
                iteration_count=state.get("iteration_count", 0),
            )

            # Add to snapshots list
            self.snapshots.append(snapshot)

            # Maintain max snapshots limit
            if len(self.snapshots) > self.max_snapshots:
                removed = self.snapshots.pop(0)
                self.logger.debug(f"Removed old snapshot from {removed.timestamp}")

            self.logger.debug(
                f"Created state snapshot at iteration {snapshot.iteration_count}"
            )
            return snapshot

        except Exception as e:
            self.logger.error(f"Failed to create state snapshot: {e}")
            raise RecoveryError(
                "snapshot_creation", f"Failed to create snapshot: {e}", e
            )

    def validate_and_recover(self, state: AgentState) -> AgentState:
        """
        Validate state and recover if corruption is detected.

        Args:
            state: Current agent state

        Returns:
            Validated or recovered state

        Raises:
            StateCorruptionError: If state is corrupted and cannot be recovered
        """
        # Validate current state
        issues = self.validator.validate_state(state)

        if not issues:
            self.logger.debug("State validation passed")
            return state

        self.logger.warning(f"State corruption detected: {issues}")

        # Attempt recovery
        try:
            recovered_state = self._attempt_recovery(state, issues)

            # Validate recovered state
            recovery_issues = self.validator.validate_state(recovered_state)
            if recovery_issues:
                raise StateCorruptionError(
                    "State recovery failed - recovered state still has issues",
                    corrupted_fields=[issue.split(":")[0] for issue in recovery_issues],
                    context={
                        "original_issues": issues,
                        "recovery_issues": recovery_issues,
                    },
                )

            self.logger.info("State successfully recovered")
            return recovered_state

        except Exception as e:
            raise StateCorruptionError(
                "Failed to recover corrupted state",
                corrupted_fields=[issue.split(":")[0] for issue in issues],
                context={"validation_issues": issues, "recovery_error": str(e)},
            )

    def _attempt_recovery(
        self, corrupted_state: AgentState, issues: List[str]
    ) -> AgentState:
        """
        Attempt to recover corrupted state.

        Args:
            corrupted_state: The corrupted state
            issues: List of validation issues

        Returns:
            Recovered state

        Raises:
            RecoveryError: If recovery fails
        """
        # Try to recover from most recent valid snapshot
        if self.snapshots:
            latest_snapshot = self.snapshots[-1]

            # Verify snapshot integrity
            if self._verify_snapshot_integrity(latest_snapshot):
                self.logger.info(
                    f"Recovering from snapshot at {latest_snapshot.timestamp}"
                )

                # Merge any valid data from corrupted state
                recovered_state = self._merge_states(
                    latest_snapshot.state, corrupted_state, issues
                )
                return recovered_state
            else:
                self.logger.warning("Latest snapshot failed integrity check")

        # Try older snapshots
        for snapshot in reversed(self.snapshots[:-1]):
            if self._verify_snapshot_integrity(snapshot):
                self.logger.info(
                    f"Recovering from older snapshot at {snapshot.timestamp}"
                )
                recovered_state = self._merge_states(
                    snapshot.state, corrupted_state, issues
                )
                return recovered_state

        # If no valid snapshots, attempt minimal recovery
        self.logger.warning("No valid snapshots found, attempting minimal recovery")
        return self._minimal_recovery(corrupted_state, issues)

    def _verify_snapshot_integrity(self, snapshot: StateSnapshot) -> bool:
        """
        Verify snapshot integrity using checksum.

        Args:
            snapshot: Snapshot to verify

        Returns:
            True if snapshot is valid
        """
        try:
            state_json = json.dumps(snapshot.state, sort_keys=True, default=str)
            current_checksum = str(hash(state_json))
            return current_checksum == snapshot.checksum
        except Exception as e:
            self.logger.error(f"Snapshot integrity check failed: {e}")
            return False

    def _merge_states(
        self, base_state: AgentState, corrupted_state: AgentState, issues: List[str]
    ) -> AgentState:
        """
        Merge valid data from corrupted state into base state.

        Args:
            base_state: Base state from snapshot
            corrupted_state: Corrupted current state
            issues: List of validation issues

        Returns:
            Merged state
        """
        merged_state = copy.deepcopy(base_state)

        # Identify fields that are not corrupted
        corrupted_fields = {issue.split(":")[0].strip() for issue in issues}

        # Merge non-corrupted fields that might have newer data
        safe_fields = ["iteration_count", "artifacts", "file_system"]

        for field in safe_fields:
            if field not in corrupted_fields and field in corrupted_state:
                # Only update if the corrupted state has a newer value
                if field == "iteration_count":
                    if corrupted_state[field] > merged_state.get(field, 0):
                        merged_state[field] = corrupted_state[field]
                else:
                    merged_state[field] = corrupted_state[field]

        return merged_state

    def _minimal_recovery(
        self, corrupted_state: AgentState, issues: List[str]
    ) -> AgentState:
        """
        Perform minimal recovery when no snapshots are available.

        Args:
            corrupted_state: Corrupted state
            issues: List of validation issues

        Returns:
            Minimally recovered state
        """
        # Create a minimal valid state
        minimal_state = {
            "user_objective": corrupted_state.get("user_objective", ""),
            "todo_list": [],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 0,
            "final_result": None,
        }

        # Try to preserve any valid data
        if "user_objective" not in [issue.split(":")[0].strip() for issue in issues]:
            minimal_state["user_objective"] = corrupted_state.get("user_objective", "")

        if "iteration_count" not in [issue.split(":")[0].strip() for issue in issues]:
            minimal_state["iteration_count"] = corrupted_state.get("iteration_count", 0)

        self.logger.warning("Performed minimal state recovery - some data may be lost")
        return minimal_state

    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about recovery operations.

        Returns:
            Dictionary with recovery statistics
        """
        return {
            "snapshots_count": len(self.snapshots),
            "oldest_snapshot": self.snapshots[0].timestamp.isoformat()
            if self.snapshots
            else None,
            "newest_snapshot": self.snapshots[-1].timestamp.isoformat()
            if self.snapshots
            else None,
            "max_snapshots": self.max_snapshots,
        }

    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self.snapshots.clear()
        self.logger.info("Cleared all state snapshots")

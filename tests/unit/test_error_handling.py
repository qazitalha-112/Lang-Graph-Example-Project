"""Unit tests for error handling and recovery mechanisms."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.error_handling import (
    ErrorHandler,
    ErrorRecoveryManager,
    RetryManager,
    RetryConfig,
    StateRecoveryManager,
    StateValidator,
)
from src.error_handling.exceptions import (
    SupervisorError,
    ToolExecutionError,
    StateCorruptionError,
    RetryExhaustedError,
    RecoveryError,
    NetworkError,
    TimeoutError,
)
from src.config import AgentConfig
from src.models.data_models import AgentState, TaskStatus


class TestRetryManager:
    """Test retry manager functionality."""

    def test_retry_config_creation(self):
        """Test retry configuration creation."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False,
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_retry_manager_success_on_first_attempt(self):
        """Test successful operation on first attempt."""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_manager = RetryManager(config)

        def successful_operation():
            return "success"

        result = retry_manager.execute_with_retry(successful_operation, "test_op")
        assert result == "success"

    def test_retry_manager_success_after_retries(self):
        """Test successful operation after retries."""
        config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        retry_manager = RetryManager(config)

        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = retry_manager.execute_with_retry(flaky_operation, "test_op")
        assert result == "success"
        assert call_count == 3

    def test_retry_manager_exhausted_retries(self):
        """Test retry exhaustion."""
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        retry_manager = RetryManager(config)

        def failing_operation():
            raise ConnectionError("Persistent error")

        with pytest.raises(RetryExhaustedError) as exc_info:
            retry_manager.execute_with_retry(failing_operation, "test_op")

        assert exc_info.value.operation == "test_op"
        assert exc_info.value.attempts == 2

    def test_retry_manager_non_retryable_error(self):
        """Test non-retryable error handling."""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_manager = RetryManager(config)

        def operation_with_value_error():
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            retry_manager.execute_with_retry(operation_with_value_error, "test_op")

    def test_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0, exponential_base=2.0, max_delay=10.0, jitter=False
        )
        retry_manager = RetryManager(config)

        # Test exponential backoff
        assert retry_manager._calculate_delay(1) == 1.0
        assert retry_manager._calculate_delay(2) == 2.0
        assert retry_manager._calculate_delay(3) == 4.0

        # Test max delay cap
        assert retry_manager._calculate_delay(10) == 10.0

    def test_retry_decorator(self):
        """Test retry decorator functionality."""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_manager = RetryManager(config)

        call_count = 0

        @retry_manager.retry("decorated_op")
        def decorated_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return "decorated_success"

        result = decorated_operation()
        assert result == "decorated_success"
        assert call_count == 2


class TestStateValidator:
    """Test state validation functionality."""

    def test_valid_state_validation(self):
        """Test validation of a valid state."""
        validator = StateValidator()

        valid_state = {
            "user_objective": "Test objective",
            "todo_list": [],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 0,
            "final_result": None,
        }

        issues = validator.validate_state(valid_state)
        assert len(issues) == 0

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        validator = StateValidator()

        invalid_state = {
            "user_objective": "Test objective",
            # Missing other required fields
        }

        issues = validator.validate_state(invalid_state)
        assert len(issues) > 0
        assert any("Missing required field" in issue for issue in issues)

    def test_invalid_field_types(self):
        """Test validation with invalid field types."""
        validator = StateValidator()

        invalid_state = {
            "user_objective": "Test objective",
            "todo_list": "not_a_list",  # Should be list
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": "not_an_int",  # Should be int
            "final_result": None,
        }

        issues = validator.validate_state(invalid_state)
        assert len(issues) >= 2
        assert any("todo_list must be a list" in issue for issue in issues)
        assert any("iteration_count must be an integer" in issue for issue in issues)

    def test_task_dependency_validation(self):
        """Test task dependency validation."""
        validator = StateValidator()

        state_with_invalid_deps = {
            "user_objective": "Test objective",
            "todo_list": [{"id": "task1", "dependencies": ["nonexistent_task"]}],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 0,
            "final_result": None,
        }

        issues = validator.validate_state(state_with_invalid_deps)
        assert any("invalid dependency" in issue for issue in issues)

    def test_self_dependency_validation(self):
        """Test self-dependency validation."""
        validator = StateValidator()

        state_with_self_dep = {
            "user_objective": "Test objective",
            "todo_list": [
                {
                    "id": "task1",
                    "dependencies": ["task1"],  # Self-dependency
                }
            ],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 0,
            "final_result": None,
        }

        issues = validator.validate_state(state_with_self_dep)
        assert any("self-dependency" in issue for issue in issues)


class TestStateRecoveryManager:
    """Test state recovery manager functionality."""

    def test_create_snapshot(self):
        """Test state snapshot creation."""
        recovery_manager = StateRecoveryManager(max_snapshots=5)

        test_state = {
            "user_objective": "Test objective",
            "todo_list": [],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 1,
            "final_result": None,
        }

        snapshot = recovery_manager.create_snapshot(test_state)

        assert snapshot.state == test_state
        assert snapshot.iteration_count == 1
        assert len(recovery_manager.snapshots) == 1

    def test_snapshot_limit_enforcement(self):
        """Test snapshot limit enforcement."""
        recovery_manager = StateRecoveryManager(max_snapshots=2)

        # Create more snapshots than the limit
        for i in range(3):
            test_state = {"iteration_count": i}
            recovery_manager.create_snapshot(test_state)

        # Should only keep the last 2 snapshots
        assert len(recovery_manager.snapshots) == 2
        assert recovery_manager.snapshots[0].iteration_count == 1
        assert recovery_manager.snapshots[1].iteration_count == 2

    def test_validate_and_recover_valid_state(self):
        """Test validation and recovery with valid state."""
        recovery_manager = StateRecoveryManager()

        valid_state = {
            "user_objective": "Test objective",
            "todo_list": [],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 0,
            "final_result": None,
        }

        result = recovery_manager.validate_and_recover(valid_state)
        assert result == valid_state

    def test_validate_and_recover_corrupted_state(self):
        """Test validation and recovery with corrupted state."""
        recovery_manager = StateRecoveryManager()

        # Create a valid snapshot first
        valid_state = {
            "user_objective": "Test objective",
            "todo_list": [],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 0,
            "final_result": None,
        }
        recovery_manager.create_snapshot(valid_state)

        # Create corrupted state
        corrupted_state = {
            "user_objective": "Test objective",
            "todo_list": "not_a_list",  # Invalid type
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 1,
            "final_result": None,
        }

        recovered_state = recovery_manager.validate_and_recover(corrupted_state)

        # Should recover from snapshot
        assert isinstance(recovered_state["todo_list"], list)
        assert (
            recovered_state["iteration_count"] == 1
        )  # Should preserve newer iteration count

    def test_recovery_without_snapshots(self):
        """Test recovery when no snapshots are available."""
        recovery_manager = StateRecoveryManager()

        corrupted_state = {
            "user_objective": "Test objective",
            "todo_list": "not_a_list",  # Invalid type
        }

        recovered_state = recovery_manager.validate_and_recover(corrupted_state)

        # Should perform minimal recovery
        assert isinstance(recovered_state["todo_list"], list)
        assert recovered_state["user_objective"] == "Test objective"


class TestErrorHandler:
    """Test error handler functionality."""

    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        config = AgentConfig()
        error_handler = ErrorHandler(config)

        assert error_handler.config == config
        assert error_handler.retry_manager is not None
        assert len(error_handler.circuit_breakers) > 0

    def test_handle_tool_error_with_fallback(self):
        """Test tool error handling with fallback."""
        config = AgentConfig()
        error_handler = ErrorHandler(config)

        def fallback_func():
            return {"status": "fallback", "result": "fallback_result"}

        test_error = ConnectionError("Network error")

        with patch.object(error_handler, "_is_retryable_error", return_value=False):
            result = error_handler.handle_tool_error(
                "test_tool", test_error, fallback_func=fallback_func
            )

        assert result["recovery_status"] == "fallback_successful"
        assert result["result"]["status"] == "fallback"

    def test_error_statistics_recording(self):
        """Test error statistics recording."""
        config = AgentConfig()
        error_handler = ErrorHandler(config)

        # Record some errors
        error_handler._record_error("tool1", ValueError("Error 1"))
        error_handler._record_error("tool1", ConnectionError("Error 2"))
        error_handler._record_error("tool2", ValueError("Error 3"))

        stats = error_handler.get_error_statistics()

        assert stats["total_errors"] == 3
        assert stats["errors_by_tool"]["tool1"] == 2
        assert stats["errors_by_tool"]["tool2"] == 1
        assert stats["errors_by_type"]["ValueError"] == 2
        assert stats["errors_by_type"]["ConnectionError"] == 1

    def test_create_error_safe_wrapper(self):
        """Test error-safe wrapper creation."""
        config = AgentConfig()
        error_handler = ErrorHandler(config)

        def test_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2

        def fallback_function():
            return -1

        wrapped_func = error_handler.create_error_safe_wrapper(
            test_function, "test_tool", fallback_func=fallback_function
        )

        # Test successful execution
        result = wrapped_func(5)
        assert result == 10

        # Test error handling with fallback
        result = wrapped_func(-1)
        assert result["recovery_status"] == "fallback_successful"


class TestErrorRecoveryManager:
    """Test error recovery manager functionality."""

    def test_error_recovery_manager_initialization(self):
        """Test error recovery manager initialization."""
        config = AgentConfig()
        recovery_manager = ErrorRecoveryManager(config)

        assert recovery_manager.config == config
        assert recovery_manager.error_handler is not None
        assert recovery_manager.state_recovery_manager is not None
        assert len(recovery_manager.recovery_strategies) > 0

    def test_register_custom_recovery_strategy(self):
        """Test registering custom recovery strategy."""
        config = AgentConfig()
        recovery_manager = ErrorRecoveryManager(config)

        def custom_strategy(error, context):
            return {"recovery_status": "custom_success"}

        recovery_manager.register_recovery_strategy("custom_error", custom_strategy)

        assert "custom_error" in recovery_manager.recovery_strategies

    def test_handle_tool_execution_error(self):
        """Test handling tool execution error."""
        config = AgentConfig()
        recovery_manager = ErrorRecoveryManager(config)

        tool_error = ToolExecutionError("test_tool", "Tool failed")

        with patch.object(
            recovery_manager, "_handle_tool_failure_recovery"
        ) as mock_handler:
            mock_handler.return_value = {"recovery_status": "successful"}

            result = recovery_manager.handle_error(tool_error)

            assert result["recovery_status"] == "successful"
            mock_handler.assert_called_once()

    def test_handle_state_corruption_error(self):
        """Test handling state corruption error."""
        config = AgentConfig()
        recovery_manager = ErrorRecoveryManager(config)

        state_error = StateCorruptionError("State corrupted")
        context = {"state": {"user_objective": "test"}}

        with patch.object(
            recovery_manager, "_handle_state_corruption_recovery"
        ) as mock_handler:
            mock_handler.return_value = {"recovery_status": "successful"}

            result = recovery_manager.handle_error(state_error, context)

            assert result["recovery_status"] == "successful"
            mock_handler.assert_called_once()

    def test_create_state_snapshot(self):
        """Test state snapshot creation."""
        config = AgentConfig()
        recovery_manager = ErrorRecoveryManager(config)

        test_state = {
            "user_objective": "Test objective",
            "todo_list": [],
            "completed_tasks": [],
            "current_task": None,
            "artifacts": {},
            "subagent_logs": [],
            "file_system": {},
            "iteration_count": 0,
            "final_result": None,
        }

        # Should not raise an exception
        recovery_manager.create_state_snapshot(test_state)

        # Verify snapshot was created
        assert len(recovery_manager.state_recovery_manager.snapshots) == 1

    def test_get_recovery_statistics(self):
        """Test getting recovery statistics."""
        config = AgentConfig()
        recovery_manager = ErrorRecoveryManager(config)

        stats = recovery_manager.get_recovery_statistics()

        assert "error_handler_stats" in stats
        assert "state_recovery_stats" in stats
        assert isinstance(stats["error_handler_stats"], dict)
        assert isinstance(stats["state_recovery_stats"], dict)


class TestExceptions:
    """Test custom exception classes."""

    def test_supervisor_error(self):
        """Test SupervisorError exception."""
        error = SupervisorError(
            "Test error", error_code="TEST_ERROR", context={"key": "value"}
        )

        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.context == {"key": "value"}

    def test_tool_execution_error(self):
        """Test ToolExecutionError exception."""
        original_error = ValueError("Original error")
        error = ToolExecutionError(
            "test_tool", "Tool failed", original_error=original_error
        )

        assert error.tool_name == "test_tool"
        assert error.original_error == original_error
        assert error.error_code == "TOOL_EXECUTION_FAILED"

    def test_state_corruption_error(self):
        """Test StateCorruptionError exception."""
        error = StateCorruptionError(
            "State corrupted", corrupted_fields=["field1", "field2"]
        )

        assert error.corrupted_fields == ["field1", "field2"]
        assert error.error_code == "STATE_CORRUPTION"

    def test_retry_exhausted_error(self):
        """Test RetryExhaustedError exception."""
        last_error = ConnectionError("Network error")
        error = RetryExhaustedError("test_operation", 3, last_error=last_error)

        assert error.operation == "test_operation"
        assert error.attempts == 3
        assert error.last_error == last_error
        assert error.error_code == "RETRY_EXHAUSTED"

    def test_network_error(self):
        """Test NetworkError exception."""
        error = NetworkError("web_scrape", "Connection failed", status_code=500)

        assert error.tool_name == "web_scrape"
        assert error.status_code == 500
        assert error.error_code == "NETWORK_ERROR"

    def test_timeout_error(self):
        """Test TimeoutError exception."""
        error = TimeoutError("execute_code", 30.0)

        assert error.tool_name == "execute_code"
        assert error.timeout_duration == 30.0
        assert error.error_code == "TIMEOUT_ERROR"


if __name__ == "__main__":
    pytest.main([__file__])

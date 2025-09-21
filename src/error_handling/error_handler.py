"""Main error handler and recovery manager for the supervisor agent system."""

import logging
import subprocess
import requests
from typing import Dict, Any, Optional, Callable, Type, List
from functools import wraps
from datetime import datetime

from ..config import AgentConfig
from .exceptions import (
    SupervisorError,
    ToolExecutionError,
    StateCorruptionError,
    RetryExhaustedError,
    RecoveryError,
    NetworkError,
    TimeoutError,
)
from .retry_manager import RetryManager, RetryConfig, CircuitBreaker
from .state_recovery import StateRecoveryManager, StateValidator
from ..models.data_models import AgentState


class ErrorHandler:
    """Central error handler for the supervisor agent system."""

    def __init__(self, config: AgentConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler.

        Args:
            config: Agent configuration
            logger: Optional logger for error events
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize retry manager with configuration
        retry_config = RetryConfig(
            max_attempts=getattr(config, "max_retry_attempts", 3),
            base_delay=getattr(config, "retry_base_delay", 1.0),
            max_delay=getattr(config, "retry_max_delay", 60.0),
            exponential_base=getattr(config, "retry_exponential_base", 2.0),
            jitter=getattr(config, "retry_jitter", True),
        )
        self.retry_manager = RetryManager(retry_config, logger)

        # Initialize circuit breakers for external services
        self.circuit_breakers = {
            "tavily": CircuitBreaker(
                failure_threshold=5, recovery_timeout=60.0, logger=logger
            ),
            "firecrawl": CircuitBreaker(
                failure_threshold=5, recovery_timeout=60.0, logger=logger
            ),
            "openai": CircuitBreaker(
                failure_threshold=3, recovery_timeout=30.0, logger=logger
            ),
        }

        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_tool": {},
            "recovery_attempts": 0,
            "successful_recoveries": 0,
        }

    def handle_tool_error(
        self,
        tool_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        fallback_func: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Handle tool execution errors with fallback mechanisms.

        Args:
            tool_name: Name of the tool that failed
            error: The exception that occurred
            context: Optional context information
            fallback_func: Optional fallback function to execute

        Returns:
            Dictionary with error handling results
        """
        self._record_error(tool_name, error)

        error_info = {
            "tool_name": tool_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
        }

        self.logger.error(f"Tool {tool_name} failed: {error}")

        # Determine if error is retryable
        if self._is_retryable_error(error):
            try:
                # Attempt retry with exponential backoff
                result = self._attempt_tool_retry(tool_name, error, context)
                error_info["recovery_status"] = "retry_successful"
                error_info["result"] = result
                return error_info

            except RetryExhaustedError as retry_error:
                self.logger.error(
                    f"Retry exhausted for tool {tool_name}: {retry_error}"
                )
                error_info["recovery_status"] = "retry_exhausted"

        # Try fallback mechanism if available
        if fallback_func:
            try:
                self.logger.info(f"Attempting fallback for tool {tool_name}")
                fallback_result = fallback_func()
                error_info["recovery_status"] = "fallback_successful"
                error_info["result"] = fallback_result
                self.error_stats["successful_recoveries"] += 1
                return error_info

            except Exception as fallback_error:
                self.logger.error(
                    f"Fallback failed for tool {tool_name}: {fallback_error}"
                )
                error_info["fallback_error"] = str(fallback_error)

        # No recovery possible
        error_info["recovery_status"] = "failed"

        # Wrap in appropriate exception type
        if isinstance(error, (ConnectionError, requests.exceptions.RequestException)):
            raise NetworkError(
                tool_name, str(error), original_error=error, context=context
            )
        elif isinstance(error, subprocess.TimeoutExpired):
            raise TimeoutError(
                tool_name,
                self.config.tool_timeout,
                original_error=error,
                context=context,
            )
        else:
            raise ToolExecutionError(
                tool_name, str(error), original_error=error, context=context
            )

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: The exception to check

        Returns:
            True if the error is retryable
        """
        retryable_types = (
            ConnectionError,
            TimeoutError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        )

        # Check for specific HTTP status codes that are retryable
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            retryable_status_codes = {429, 500, 502, 503, 504}
            return error.response.status_code in retryable_status_codes

        return isinstance(error, retryable_types)

    def _attempt_tool_retry(
        self,
        tool_name: str,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Attempt to retry a failed tool operation.

        Args:
            tool_name: Name of the tool
            original_error: The original error
            context: Optional context information

        Returns:
            Result of successful retry

        Raises:
            RetryExhaustedError: If all retry attempts fail
        """
        # This is a placeholder - in real implementation, this would
        # re-execute the original tool function with the same parameters
        # For now, we'll simulate a retry attempt

        def retry_operation():
            # In real implementation, this would call the actual tool function
            # with the original parameters stored in context
            raise original_error  # Simulate failure for now

        return self.retry_manager.execute_with_retry(
            retry_operation, f"retry_{tool_name}"
        )

    def handle_state_corruption(
        self, state: AgentState, recovery_manager: StateRecoveryManager
    ) -> AgentState:
        """
        Handle state corruption with recovery mechanisms.

        Args:
            state: Potentially corrupted state
            recovery_manager: State recovery manager

        Returns:
            Recovered state

        Raises:
            StateCorruptionError: If state cannot be recovered
        """
        self.error_stats["recovery_attempts"] += 1

        try:
            self.logger.info("Attempting state corruption recovery")
            recovered_state = recovery_manager.validate_and_recover(state)

            self.error_stats["successful_recoveries"] += 1
            self.logger.info("State corruption recovery successful")

            return recovered_state

        except StateCorruptionError as e:
            self.logger.error(f"State corruption recovery failed: {e}")
            self._record_error("state_recovery", e)
            raise e

    def create_error_safe_wrapper(
        self,
        func: Callable,
        tool_name: str,
        fallback_func: Optional[Callable] = None,
        circuit_breaker_key: Optional[str] = None,
    ) -> Callable:
        """
        Create an error-safe wrapper for a function.

        Args:
            func: Function to wrap
            tool_name: Name of the tool/operation
            fallback_func: Optional fallback function
            circuit_breaker_key: Optional circuit breaker key

        Returns:
            Wrapped function with error handling
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Use circuit breaker if specified
                if circuit_breaker_key and circuit_breaker_key in self.circuit_breakers:
                    circuit_breaker = self.circuit_breakers[circuit_breaker_key]
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                return self.handle_tool_error(
                    tool_name=tool_name,
                    error=e,
                    context={"args": str(args)[:100], "kwargs": str(kwargs)[:100]},
                    fallback_func=fallback_func,
                )

        return wrapper

    def _record_error(self, tool_name: str, error: Exception):
        """Record error statistics."""
        self.error_stats["total_errors"] += 1

        error_type = type(error).__name__
        self.error_stats["errors_by_type"][error_type] = (
            self.error_stats["errors_by_type"].get(error_type, 0) + 1
        )

        self.error_stats["errors_by_tool"][tool_name] = (
            self.error_stats["errors_by_tool"].get(tool_name, 0) + 1
        )

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error handling statistics.

        Returns:
            Dictionary with error statistics
        """
        return {
            **self.error_stats,
            "circuit_breaker_states": {
                key: breaker.state for key, breaker in self.circuit_breakers.items()
            },
            "recovery_success_rate": (
                self.error_stats["successful_recoveries"]
                / max(self.error_stats["recovery_attempts"], 1)
            )
            * 100,
        }

    def reset_statistics(self):
        """Reset error statistics."""
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_tool": {},
            "recovery_attempts": 0,
            "successful_recoveries": 0,
        }


class ErrorRecoveryManager:
    """High-level error recovery manager that coordinates all error handling components."""

    def __init__(self, config: AgentConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize error recovery manager.

        Args:
            config: Agent configuration
            logger: Optional logger
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.error_handler = ErrorHandler(config, logger)
        self.state_recovery_manager = StateRecoveryManager(
            max_snapshots=getattr(config, "max_state_snapshots", 10), logger=logger
        )

        # Recovery strategies
        self.recovery_strategies = {
            "tool_failure": self._handle_tool_failure_recovery,
            "state_corruption": self._handle_state_corruption_recovery,
            "network_error": self._handle_network_error_recovery,
            "timeout_error": self._handle_timeout_error_recovery,
        }

    def register_recovery_strategy(self, error_type: str, strategy_func: Callable):
        """
        Register a custom recovery strategy.

        Args:
            error_type: Type of error to handle
            strategy_func: Function to handle the error
        """
        self.recovery_strategies[error_type] = strategy_func
        self.logger.info(f"Registered recovery strategy for {error_type}")

    def handle_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an error using appropriate recovery strategy.

        Args:
            error: The exception to handle
            context: Optional context information

        Returns:
            Recovery result dictionary
        """
        error_type = type(error).__name__.lower()

        # Map exception types to recovery strategies
        strategy_mapping = {
            "toolexecutionerror": "tool_failure",
            "statecorruptionerror": "state_corruption",
            "networkerror": "network_error",
            "timeouterror": "timeout_error",
            "connectionerror": "network_error",
            "httperror": "network_error",
        }

        strategy_key = strategy_mapping.get(error_type, "tool_failure")

        if strategy_key in self.recovery_strategies:
            try:
                return self.recovery_strategies[strategy_key](error, context)
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery strategy {strategy_key} failed: {recovery_error}"
                )
                return {
                    "recovery_status": "failed",
                    "error": str(error),
                    "recovery_error": str(recovery_error),
                }
        else:
            self.logger.warning(
                f"No recovery strategy found for error type: {error_type}"
            )
            return {"recovery_status": "no_strategy", "error": str(error)}

    def _handle_tool_failure_recovery(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle tool failure recovery."""
        tool_name = getattr(error, "tool_name", "unknown")
        return self.error_handler.handle_tool_error(tool_name, error, context)

    def _handle_state_corruption_recovery(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle state corruption recovery."""
        if context and "state" in context:
            try:
                recovered_state = self.error_handler.handle_state_corruption(
                    context["state"], self.state_recovery_manager
                )
                return {
                    "recovery_status": "successful",
                    "recovered_state": recovered_state,
                }
            except Exception as e:
                return {"recovery_status": "failed", "error": str(e)}
        else:
            return {
                "recovery_status": "failed",
                "error": "No state provided in context for recovery",
            }

    def _handle_network_error_recovery(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle network error recovery."""
        # Implement network-specific recovery logic
        return {
            "recovery_status": "retry_recommended",
            "error": str(error),
            "recommendation": "Check network connectivity and retry operation",
        }

    def _handle_timeout_error_recovery(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle timeout error recovery."""
        # Implement timeout-specific recovery logic
        return {
            "recovery_status": "retry_with_longer_timeout",
            "error": str(error),
            "recommendation": "Increase timeout duration and retry operation",
        }

    def create_state_snapshot(self, state: AgentState) -> None:
        """Create a state snapshot for recovery purposes."""
        try:
            self.state_recovery_manager.create_snapshot(state)
        except Exception as e:
            self.logger.error(f"Failed to create state snapshot: {e}")

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        return {
            "error_handler_stats": self.error_handler.get_error_statistics(),
            "state_recovery_stats": self.state_recovery_manager.get_recovery_stats(),
        }

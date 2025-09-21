"""Custom exceptions for error handling and recovery."""

from typing import Optional, Dict, Any


class SupervisorError(Exception):
    """Base exception for supervisor agent errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize supervisor error.

        Args:
            message: Error message
            error_code: Optional error code for categorization
            context: Optional context information
        """
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}


class ToolExecutionError(SupervisorError):
    """Exception raised when tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize tool execution error.

        Args:
            tool_name: Name of the tool that failed
            message: Error message
            original_error: Original exception that caused the failure
            context: Optional context information
        """
        super().__init__(message, error_code="TOOL_EXECUTION_FAILED", context=context)
        self.tool_name = tool_name
        self.original_error = original_error


class StateCorruptionError(SupervisorError):
    """Exception raised when agent state becomes corrupted."""

    def __init__(
        self,
        message: str,
        corrupted_fields: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize state corruption error.

        Args:
            message: Error message
            corrupted_fields: List of corrupted state fields
            context: Optional context information
        """
        super().__init__(message, error_code="STATE_CORRUPTION", context=context)
        self.corrupted_fields = corrupted_fields or []


class RetryExhaustedError(SupervisorError):
    """Exception raised when retry attempts are exhausted."""

    def __init__(
        self,
        operation: str,
        attempts: int,
        last_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize retry exhausted error.

        Args:
            operation: Name of the operation that failed
            attempts: Number of retry attempts made
            last_error: Last error encountered
            context: Optional context information
        """
        super().__init__(
            f"Retry exhausted for {operation} after {attempts} attempts",
            error_code="RETRY_EXHAUSTED",
            context=context,
        )
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error


class RecoveryError(SupervisorError):
    """Exception raised when recovery operations fail."""

    def __init__(
        self,
        recovery_type: str,
        message: str,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize recovery error.

        Args:
            recovery_type: Type of recovery that failed
            message: Error message
            original_error: Original error that triggered recovery
            context: Optional context information
        """
        super().__init__(message, error_code="RECOVERY_FAILED", context=context)
        self.recovery_type = recovery_type
        self.original_error = original_error


class NetworkError(ToolExecutionError):
    """Exception raised for network-related tool failures."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize network error.

        Args:
            tool_name: Name of the tool that failed
            message: Error message
            status_code: HTTP status code if applicable
            original_error: Original exception that caused the failure
            context: Optional context information
        """
        super().__init__(tool_name, message, original_error, context)
        self.status_code = status_code
        self.error_code = "NETWORK_ERROR"


class TimeoutError(ToolExecutionError):
    """Exception raised when tool execution times out."""

    def __init__(
        self,
        tool_name: str,
        timeout_duration: float,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize timeout error.

        Args:
            tool_name: Name of the tool that timed out
            timeout_duration: Duration of the timeout
            original_error: Original exception that caused the timeout
            context: Optional context information
        """
        message = f"Tool {tool_name} timed out after {timeout_duration} seconds"
        super().__init__(tool_name, message, original_error, context)
        self.timeout_duration = timeout_duration
        self.error_code = "TIMEOUT_ERROR"


class ValidationError(SupervisorError):
    """Exception raised for validation failures."""

    def __init__(
        self,
        field: str,
        value: Any,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize validation error.

        Args:
            field: Field that failed validation
            value: Value that failed validation
            message: Error message
            context: Optional context information
        """
        super().__init__(message, error_code="VALIDATION_ERROR", context=context)
        self.field = field
        self.value = value

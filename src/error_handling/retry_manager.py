"""Retry management with exponential backoff for external API calls."""

import time
import random
import logging
from typing import Callable, Any, Optional, Dict, Type, Union
from dataclasses import dataclass, field
from functools import wraps

from .exceptions import RetryExhaustedError, NetworkError, TimeoutError


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd
    retryable_exceptions: tuple = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            NetworkError,
            Exception,  # Catch-all for unexpected errors
        )
    )
    non_retryable_exceptions: tuple = field(
        default_factory=lambda: (ValueError, TypeError, KeyError, AttributeError)
    )


class RetryManager:
    """Manages retry logic with exponential backoff for operations."""

    def __init__(self, config: RetryConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize retry manager.

        Args:
            config: Retry configuration
            logger: Optional logger for retry events
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def retry(self, operation_name: str = "operation"):
        """
        Decorator for adding retry logic to functions.

        Args:
            operation_name: Name of the operation for logging

        Returns:
            Decorated function with retry logic
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                return self.execute_with_retry(func, operation_name, *args, **kwargs)

            return wrapper

        return decorator

    def execute_with_retry(
        self, func: Callable, operation_name: str, *args, **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            operation_name: Name of the operation for logging
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function execution

        Raises:
            RetryExhaustedError: When all retry attempts are exhausted
        """
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.logger.debug(
                    f"Attempting {operation_name} (attempt {attempt}/{self.config.max_attempts})"
                )
                result = func(*args, **kwargs)

                if attempt > 1:
                    self.logger.info(f"{operation_name} succeeded on attempt {attempt}")

                return result

            except self.config.non_retryable_exceptions as e:
                self.logger.error(
                    f"{operation_name} failed with non-retryable error: {e}"
                )
                raise e

            except self.config.retryable_exceptions as e:
                last_exception = e
                self.logger.warning(
                    f"{operation_name} failed on attempt {attempt}: {e}"
                )

                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    self.logger.info(
                        f"Retrying {operation_name} in {delay:.2f} seconds"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"{operation_name} failed after {self.config.max_attempts} attempts"
                    )

        # All attempts exhausted
        raise RetryExhaustedError(
            operation=operation_name,
            attempts=self.config.max_attempts,
            last_error=last_exception,
            context={
                "function": func.__name__,
                "args": str(args)[:100],  # Truncate for logging
                "kwargs": str(kwargs)[:100],
            },
        )

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for exponential backoff.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds
        """
        # Calculate exponential delay
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))

        # Cap at maximum delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            jitter = random.uniform(-jitter_range, jitter_range)
            delay += jitter

        return max(0, delay)  # Ensure non-negative delay

    def create_retryable_wrapper(self, func: Callable, operation_name: str) -> Callable:
        """
        Create a wrapper function with retry logic.

        Args:
            func: Function to wrap
            operation_name: Name of the operation

        Returns:
            Wrapped function with retry logic
        """

        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, operation_name, *args, **kwargs)

        wrapper.__name__ = f"retryable_{func.__name__}"
        wrapper.__doc__ = f"Retryable version of {func.__name__}"

        return wrapper


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            logger: Optional logger
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.logger = logger or logging.getLogger(__name__)

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - operation blocked")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful operation."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("Circuit breaker reset to CLOSED state")

        self.failure_count = 0

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


def create_retry_manager(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> RetryManager:
    """
    Create a retry manager with specified configuration.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential backoff multiplier
        jitter: Whether to add random jitter

    Returns:
        Configured RetryManager instance
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
    )

    return RetryManager(config)

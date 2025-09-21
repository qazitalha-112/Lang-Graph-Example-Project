"""Configuration management for the LangGraph Supervisor Agent."""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Environment(Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass
class AgentConfig:
    """Configuration settings for the supervisor agent system."""

    # Environment Configuration
    environment: Environment = field(
        default_factory=lambda: Environment(os.getenv("ENVIRONMENT", "development"))
    )

    # LLM Configuration
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4"))
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_organization: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_ORGANIZATION")
    )
    openai_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )

    # Agent Behavior
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10"))
    )
    max_subagents: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUBAGENTS", "5"))
    )
    tool_timeout: int = field(
        default_factory=lambda: int(os.getenv("TOOL_TIMEOUT", "30"))
    )
    subagent_timeout: int = field(
        default_factory=lambda: int(os.getenv("SUBAGENT_TIMEOUT", "300"))
    )

    # LangSmith Configuration
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "supervisor-agent")
    )
    langsmith_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY")
    )
    langsmith_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )
    )
    enable_tracing: bool = field(
        default_factory=lambda: os.getenv("ENABLE_TRACING", "false").lower() == "true"
    )

    # External API Keys
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )
    firecrawl_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("FIRECRAWL_API_KEY")
    )

    # File System Configuration
    virtual_fs_root: str = field(
        default_factory=lambda: os.getenv("VIRTUAL_FS_ROOT", "/virtual")
    )
    max_file_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE", "10485760"))
    )  # 10MB
    max_files: int = field(default_factory=lambda: int(os.getenv("MAX_FILES", "1000")))

    # Error Handling Configuration
    max_retry_attempts: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    )
    retry_base_delay: float = field(
        default_factory=lambda: float(os.getenv("RETRY_BASE_DELAY", "1.0"))
    )
    retry_max_delay: float = field(
        default_factory=lambda: float(os.getenv("RETRY_MAX_DELAY", "60.0"))
    )
    retry_exponential_base: float = field(
        default_factory=lambda: float(os.getenv("RETRY_EXPONENTIAL_BASE", "2.0"))
    )
    retry_jitter: bool = field(
        default_factory=lambda: os.getenv("RETRY_JITTER", "true").lower() == "true"
    )
    max_state_snapshots: int = field(
        default_factory=lambda: int(os.getenv("MAX_STATE_SNAPSHOTS", "10"))
    )
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    )
    circuit_breaker_recovery_timeout: float = field(
        default_factory=lambda: float(
            os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60.0")
        )
    )

    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(
        default_factory=lambda: os.getenv(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    enable_file_logging: bool = field(
        default_factory=lambda: os.getenv("ENABLE_FILE_LOGGING", "false").lower()
        == "true"
    )
    log_file_path: str = field(
        default_factory=lambda: os.getenv("LOG_FILE_PATH", "logs/supervisor_agent.log")
    )

    # Security Configuration
    api_key_validation: bool = field(
        default_factory=lambda: os.getenv("API_KEY_VALIDATION", "true").lower()
        == "true"
    )
    secure_mode: bool = field(
        default_factory=lambda: os.getenv("SECURE_MODE", "false").lower() == "true"
    )
    allowed_file_extensions: List[str] = field(
        default_factory=lambda: os.getenv(
            "ALLOWED_FILE_EXTENSIONS", ".txt,.md,.py,.json,.yaml,.yml"
        ).split(",")
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_required_keys()
        self._validate_numeric_values()
        self._validate_environment_specific()
        self._validate_security_settings()

    def _validate_required_keys(self):
        """Validate that required API keys are present."""
        if not self.openai_api_key:
            raise ConfigValidationError("OPENAI_API_KEY is required")

        if self.enable_tracing and not self.langsmith_api_key:
            raise ConfigValidationError(
                "LANGSMITH_API_KEY is required when tracing is enabled"
            )

        # Environment-specific validations
        if self.environment == Environment.PRODUCTION:
            if not self.langsmith_api_key:
                raise ConfigValidationError(
                    "LANGSMITH_API_KEY is required in production"
                )
            if not self.secure_mode:
                raise ConfigValidationError("SECURE_MODE must be enabled in production")

    def _validate_numeric_values(self):
        """Validate that numeric configuration values are within acceptable ranges."""
        validations = [
            (self.max_iterations, "max_iterations", 1, 100),
            (self.max_subagents, "max_subagents", 1, 20),
            (self.tool_timeout, "tool_timeout", 1, 600),
            (self.subagent_timeout, "subagent_timeout", 1, 3600),
            (self.max_file_size, "max_file_size", 1, 100 * 1024 * 1024),  # 100MB max
            (self.max_files, "max_files", 1, 10000),
            (self.max_retry_attempts, "max_retry_attempts", 1, 10),
            (self.retry_base_delay, "retry_base_delay", 0.1, 60.0),
            (self.retry_max_delay, "retry_max_delay", 1.0, 3600.0),
            (self.max_state_snapshots, "max_state_snapshots", 1, 100),
            (
                self.circuit_breaker_failure_threshold,
                "circuit_breaker_failure_threshold",
                1,
                50,
            ),
            (
                self.circuit_breaker_recovery_timeout,
                "circuit_breaker_recovery_timeout",
                1.0,
                3600.0,
            ),
        ]

        for value, name, min_val, max_val in validations:
            if not (min_val <= value <= max_val):
                raise ConfigValidationError(
                    f"{name} must be between {min_val} and {max_val}, got {value}"
                )

        if self.retry_exponential_base <= 1:
            raise ConfigValidationError("retry_exponential_base must be greater than 1")

        if self.retry_max_delay <= self.retry_base_delay:
            raise ConfigValidationError(
                "retry_max_delay must be greater than retry_base_delay"
            )

    def _validate_environment_specific(self):
        """Validate environment-specific configuration."""
        if self.environment == Environment.PRODUCTION:
            # Production-specific validations
            if self.max_iterations > 50:
                raise ConfigValidationError(
                    "max_iterations should not exceed 50 in production"
                )
            if self.max_subagents > 10:
                raise ConfigValidationError(
                    "max_subagents should not exceed 10 in production"
                )

        elif self.environment == Environment.TESTING:
            # Testing-specific validations
            if self.enable_tracing:
                raise ConfigValidationError(
                    "Tracing should be disabled in testing environment"
                )

    def _validate_security_settings(self):
        """Validate security-related configuration."""
        if self.secure_mode:
            # Ensure secure defaults
            if self.tool_timeout > 60:
                raise ConfigValidationError(
                    "tool_timeout should not exceed 60 seconds in secure mode"
                )
            if self.subagent_timeout > 600:
                raise ConfigValidationError(
                    "subagent_timeout should not exceed 600 seconds in secure mode"
                )

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigValidationError(f"log_level must be one of {valid_log_levels}")

        # Validate file extensions
        for ext in self.allowed_file_extensions:
            if not ext.startswith("."):
                raise ConfigValidationError(
                    f"File extension '{ext}' must start with a dot"
                )

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Args:
            include_sensitive: Whether to include sensitive information like API keys
        """
        config_dict = {
            "environment": self.environment.value,
            "llm_model": self.llm_model,
            "max_iterations": self.max_iterations,
            "max_subagents": self.max_subagents,
            "tool_timeout": self.tool_timeout,
            "subagent_timeout": self.subagent_timeout,
            "langsmith_project": self.langsmith_project,
            "langsmith_endpoint": self.langsmith_endpoint,
            "enable_tracing": self.enable_tracing,
            "virtual_fs_root": self.virtual_fs_root,
            "max_file_size": self.max_file_size,
            "max_files": self.max_files,
            "max_retry_attempts": self.max_retry_attempts,
            "retry_base_delay": self.retry_base_delay,
            "retry_max_delay": self.retry_max_delay,
            "retry_exponential_base": self.retry_exponential_base,
            "retry_jitter": self.retry_jitter,
            "max_state_snapshots": self.max_state_snapshots,
            "circuit_breaker_failure_threshold": self.circuit_breaker_failure_threshold,
            "circuit_breaker_recovery_timeout": self.circuit_breaker_recovery_timeout,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "enable_file_logging": self.enable_file_logging,
            "log_file_path": self.log_file_path,
            "api_key_validation": self.api_key_validation,
            "secure_mode": self.secure_mode,
            "allowed_file_extensions": self.allowed_file_extensions,
        }

        if include_sensitive:
            config_dict.update(
                {
                    "openai_api_key": self.openai_api_key,
                    "openai_organization": self.openai_organization,
                    "openai_base_url": self.openai_base_url,
                    "langsmith_api_key": self.langsmith_api_key,
                    "tavily_api_key": self.tavily_api_key,
                    "firecrawl_api_key": self.firecrawl_api_key,
                }
            )

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create configuration from dictionary."""
        # Handle environment enum conversion
        if "environment" in config_dict and isinstance(config_dict["environment"], str):
            config_dict["environment"] = Environment(config_dict["environment"])

        return cls(**config_dict)

    def get_masked_dict(self) -> Dict[str, Any]:
        """Get configuration dictionary with sensitive values masked."""
        config_dict = self.to_dict(include_sensitive=True)

        # Mask sensitive keys
        sensitive_keys = [
            "openai_api_key",
            "langsmith_api_key",
            "tavily_api_key",
            "firecrawl_api_key",
        ]

        for key in sensitive_keys:
            if config_dict.get(key):
                config_dict[key] = "***MASKED***"

        return config_dict


class ConfigManager:
    """Manages configuration loading, validation, and environment-specific settings."""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[AgentConfig] = None

    def __new__(cls) -> "ConfigManager":
        """Singleton pattern for configuration manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(
        self,
        config_file: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None,
        reload: bool = False,
    ) -> AgentConfig:
        """Load configuration from environment variables and optional config file.

        Args:
            config_file: Optional path to JSON configuration file
            environment: Override environment setting
            reload: Force reload of configuration

        Returns:
            Loaded and validated configuration
        """
        if self._config is not None and not reload:
            return self._config

        # Load environment-specific .env file if it exists
        if environment:
            os.environ["ENVIRONMENT"] = environment
            env_file = f".env.{environment}"
            if Path(env_file).exists():
                load_dotenv(env_file, override=True)

        # Load configuration from file if provided
        config_dict = {}
        if config_file:
            config_dict = self._load_config_file(config_file)

        # Create configuration with environment variables taking precedence
        self._config = AgentConfig(**config_dict)

        return self._config

    def _load_config_file(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_path = Path(config_file)

        if not config_path.exists():
            raise ConfigValidationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Error reading configuration file: {e}")

    def save_config(
        self,
        config: AgentConfig,
        config_file: Union[str, Path],
        include_sensitive: bool = False,
    ) -> None:
        """Save configuration to JSON file.

        Args:
            config: Configuration to save
            config_file: Path to save configuration
            include_sensitive: Whether to include sensitive information
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.to_dict(include_sensitive=include_sensitive)

        try:
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
        except Exception as e:
            raise ConfigValidationError(f"Error saving configuration file: {e}")

    def validate_api_keys(self, config: AgentConfig) -> Dict[str, bool]:
        """Validate API keys by making test requests.

        Args:
            config: Configuration to validate

        Returns:
            Dictionary mapping API key names to validation status
        """
        if not config.api_key_validation:
            return {}

        validation_results = {}

        # Validate OpenAI API key
        if config.openai_api_key:
            validation_results["openai"] = self._validate_openai_key(
                config.openai_api_key
            )

        # Validate LangSmith API key
        if config.langsmith_api_key:
            validation_results["langsmith"] = self._validate_langsmith_key(
                config.langsmith_api_key, config.langsmith_endpoint
            )

        # Validate Tavily API key
        if config.tavily_api_key:
            validation_results["tavily"] = self._validate_tavily_key(
                config.tavily_api_key
            )

        # Validate Firecrawl API key
        if config.firecrawl_api_key:
            validation_results["firecrawl"] = self._validate_firecrawl_key(
                config.firecrawl_api_key
            )

        return validation_results

    def _validate_openai_key(self, api_key: str) -> bool:
        """Validate OpenAI API key."""
        try:
            import openai

            client = openai.OpenAI(api_key=api_key)
            # Make a minimal request to validate the key
            client.models.list()
            return True
        except Exception:
            return False

    def _validate_langsmith_key(self, api_key: str, endpoint: str) -> bool:
        """Validate LangSmith API key."""
        try:
            import requests

            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{endpoint}/info", headers=headers, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def _validate_tavily_key(self, api_key: str) -> bool:
        """Validate Tavily API key."""
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
            # Make a minimal search to validate the key
            client.search("test", max_results=1)
            return True
        except Exception:
            return False

    def _validate_firecrawl_key(self, api_key: str) -> bool:
        """Validate Firecrawl API key."""
        try:
            from firecrawl import FirecrawlApp

            app = FirecrawlApp(api_key=api_key)
            # Make a minimal request to validate the key
            app.scrape_url("https://example.com")
            return True
        except Exception:
            return False

    def get_environment_config(self, environment: Environment) -> Dict[str, Any]:
        """Get environment-specific configuration overrides."""
        configs = {
            Environment.DEVELOPMENT: {
                "log_level": "DEBUG",
                "enable_tracing": True,
                "secure_mode": False,
                "max_iterations": 20,
                "tool_timeout": 60,
            },
            Environment.TESTING: {
                "log_level": "WARNING",
                "enable_tracing": False,
                "secure_mode": False,
                "max_iterations": 5,
                "tool_timeout": 10,
            },
            Environment.STAGING: {
                "log_level": "INFO",
                "enable_tracing": True,
                "secure_mode": True,
                "max_iterations": 15,
                "tool_timeout": 30,
            },
            Environment.PRODUCTION: {
                "log_level": "WARNING",
                "enable_tracing": True,
                "secure_mode": True,
                "max_iterations": 10,
                "tool_timeout": 30,
            },
        }

        return configs.get(environment, {})


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config(
    config_file: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None,
    reload: bool = False,
) -> AgentConfig:
    """Get the current configuration instance.

    Args:
        config_file: Optional path to JSON configuration file
        environment: Override environment setting
        reload: Force reload of configuration

    Returns:
        Loaded and validated configuration
    """
    return _config_manager.load_config(config_file, environment, reload)


def validate_environment() -> bool:
    """Validate that the environment is properly configured."""
    try:
        config = get_config()

        # Validate API keys if enabled
        if config.api_key_validation:
            validation_results = _config_manager.validate_api_keys(config)
            failed_validations = [k for k, v in validation_results.items() if not v]

            if failed_validations:
                print(f"API key validation failed for: {', '.join(failed_validations)}")
                return False

        return True
    except ConfigValidationError as e:
        print(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during validation: {e}")
        return False


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    return _config_manager


def create_config_template(
    output_file: Union[str, Path] = "config.template.json",
    environment: Environment = Environment.DEVELOPMENT,
) -> None:
    """Create a configuration template file.

    Args:
        output_file: Path to save the template
        environment: Environment to create template for
    """
    # Create a default configuration
    config = AgentConfig()

    # Apply environment-specific overrides
    env_config = _config_manager.get_environment_config(environment)
    for key, value in env_config.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Save template without sensitive information
    _config_manager.save_config(config, output_file, include_sensitive=False)

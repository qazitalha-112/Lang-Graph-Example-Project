"""Tests for configuration management system."""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from tempfile import NamedTemporaryFile, TemporaryDirectory

from src.config import (
    AgentConfig,
    ConfigManager,
    Environment,
    ConfigValidationError,
    get_config,
    validate_environment,
    get_config_manager,
    create_config_template,
    _config_manager,
)


class TestAgentConfig:
    """Test AgentConfig class."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = AgentConfig()

            assert config.environment == Environment.DEVELOPMENT
            assert config.llm_model == "gpt-4"
            assert config.max_iterations == 10
            assert config.max_subagents == 5
            assert config.tool_timeout == 30
            assert config.enable_tracing is False  # Default is now False
            assert config.secure_mode is False

    def test_config_from_environment_variables(self):
        """Test config creation from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "LLM_MODEL": "gpt-3.5-turbo",
                "MAX_ITERATIONS": "15",
                "SECURE_MODE": "true",
                "OPENAI_API_KEY": "test-key",
                "LANGSMITH_API_KEY": "langsmith-key",  # Required for production
            },
        ):
            config = AgentConfig()

            assert config.environment == Environment.PRODUCTION
            assert config.llm_model == "gpt-3.5-turbo"
            assert config.max_iterations == 15
            assert config.secure_mode is True
            assert config.openai_api_key == "test-key"

    def test_config_validation_success(self):
        """Test successful config validation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = AgentConfig()
            # Should not raise any exception
            assert config.openai_api_key == "test-key"

    def test_config_validation_missing_openai_key(self):
        """Test validation failure when OpenAI key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ConfigValidationError, match="OPENAI_API_KEY is required"
            ):
                AgentConfig()

    def test_config_validation_tracing_without_langsmith_key(self):
        """Test validation failure when tracing is enabled without LangSmith key."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "ENABLE_TRACING": "true"}
        ):
            with pytest.raises(
                ConfigValidationError, match="LANGSMITH_API_KEY is required"
            ):
                AgentConfig()

    def test_config_validation_production_requirements(self):
        """Test production-specific validation requirements."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "OPENAI_API_KEY": "test-key",
                "LANGSMITH_API_KEY": "langsmith-key",
                "SECURE_MODE": "false",
            },
        ):
            with pytest.raises(
                ConfigValidationError, match="SECURE_MODE must be enabled"
            ):
                AgentConfig()

    def test_config_validation_numeric_ranges(self):
        """Test numeric value range validation."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "MAX_ITERATIONS": "0"}
        ):
            with pytest.raises(
                ConfigValidationError, match="max_iterations must be between"
            ):
                AgentConfig()

    def test_config_validation_retry_delays(self):
        """Test retry delay validation."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "RETRY_BASE_DELAY": "10.0",
                "RETRY_MAX_DELAY": "5.0",
            },
        ):
            with pytest.raises(
                ConfigValidationError, match="retry_max_delay must be greater"
            ):
                AgentConfig()

    def test_config_to_dict_without_sensitive(self):
        """Test converting config to dict without sensitive information."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-key"}):
            config = AgentConfig()
            config_dict = config.to_dict(include_sensitive=False)

            assert "openai_api_key" not in config_dict
            assert "langsmith_api_key" not in config_dict
            assert config_dict["llm_model"] == "gpt-4"
            assert config_dict["max_iterations"] == 10

    def test_config_to_dict_with_sensitive(self):
        """Test converting config to dict with sensitive information."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-key"}):
            config = AgentConfig()
            config_dict = config.to_dict(include_sensitive=True)

            assert config_dict["openai_api_key"] == "secret-key"
            assert config_dict["llm_model"] == "gpt-4"

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "environment": "testing",
            "llm_model": "gpt-3.5-turbo",
            "max_iterations": 5,
            "openai_api_key": "test-key",
            "enable_tracing": False,  # Disable tracing for testing
        }

        config = AgentConfig.from_dict(config_dict)

        assert config.environment == Environment.TESTING
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.max_iterations == 5
        assert config.openai_api_key == "test-key"

    def test_get_masked_dict(self):
        """Test getting masked configuration dictionary."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "secret-key", "LANGSMITH_API_KEY": "another-secret"},
        ):
            config = AgentConfig()
            masked_dict = config.get_masked_dict()

            assert masked_dict["openai_api_key"] == "***MASKED***"
            assert masked_dict["langsmith_api_key"] == "***MASKED***"
            assert masked_dict["llm_model"] == "gpt-4"


class TestConfigManager:
    """Test ConfigManager class."""

    def test_singleton_pattern(self):
        """Test that ConfigManager follows singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()

        assert manager1 is manager2

    def test_load_config_from_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "llm_model": "gpt-3.5-turbo",
            "max_iterations": 15,
            "openai_api_key": "file-key",
            "enable_tracing": False,
        }

        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            manager = ConfigManager()
            config = manager.load_config(config_file=config_file)

            assert config.llm_model == "gpt-3.5-turbo"
            assert config.max_iterations == 15
            assert config.openai_api_key == "file-key"
        finally:
            os.unlink(config_file)

    def test_load_config_file_not_found(self):
        """Test error handling when config file is not found."""
        manager = ConfigManager()
        manager._config = None  # Reset cached config

        with pytest.raises(ConfigValidationError, match="Configuration file not found"):
            manager.load_config(config_file="nonexistent.json")

    def test_load_config_invalid_json(self):
        """Test error handling for invalid JSON in config file."""
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            config_file = f.name

        try:
            manager = ConfigManager()
            manager._config = None  # Reset cached config

            with pytest.raises(ConfigValidationError, match="Invalid JSON"):
                manager.load_config(config_file=config_file)
        finally:
            os.unlink(config_file)

    def test_save_config(self):
        """Test saving configuration to file."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = AgentConfig()

            with TemporaryDirectory() as temp_dir:
                config_file = Path(temp_dir) / "test_config.json"

                manager = ConfigManager()
                manager.save_config(config, config_file, include_sensitive=False)

                assert config_file.exists()

                with open(config_file) as f:
                    saved_data = json.load(f)

                assert saved_data["llm_model"] == "gpt-4"
                assert "openai_api_key" not in saved_data

    def test_environment_specific_config(self):
        """Test environment-specific configuration loading."""
        manager = ConfigManager()

        # Test development environment
        dev_config = manager.get_environment_config(Environment.DEVELOPMENT)
        assert dev_config["log_level"] == "DEBUG"
        assert dev_config["secure_mode"] is False

        # Test production environment
        prod_config = manager.get_environment_config(Environment.PRODUCTION)
        assert prod_config["log_level"] == "WARNING"
        assert prod_config["secure_mode"] is True

    @patch("src.config.ConfigManager._validate_openai_key")
    def test_validate_api_keys(self, mock_validate_openai):
        """Test API key validation."""
        mock_validate_openai.return_value = True

        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "API_KEY_VALIDATION": "true"}
        ):
            config = AgentConfig()
            manager = ConfigManager()

            results = manager.validate_api_keys(config)

            assert results["openai"] is True
            mock_validate_openai.assert_called_once_with("test-key")

    @patch("openai.OpenAI")
    def test_validate_openai_key_success(self, mock_openai_client):
        """Test successful OpenAI key validation."""
        mock_client = MagicMock()
        mock_openai_client.return_value = mock_client
        mock_client.models.list.return_value = []

        manager = ConfigManager()
        result = manager._validate_openai_key("test-key")

        assert result is True
        mock_openai_client.assert_called_once_with(api_key="test-key")

    @patch("openai.OpenAI")
    def test_validate_openai_key_failure(self, mock_openai_client):
        """Test failed OpenAI key validation."""
        mock_openai_client.side_effect = Exception("Invalid API key")

        manager = ConfigManager()
        result = manager._validate_openai_key("invalid-key")

        assert result is False


class TestConfigurationFunctions:
    """Test module-level configuration functions."""

    def test_get_config(self):
        """Test get_config function."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Reset the global config manager
            _config_manager._config = None
            config = get_config(reload=True)

            assert isinstance(config, AgentConfig)
            assert config.openai_api_key == "test-key"

    def test_validate_environment_success(self):
        """Test successful environment validation."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "API_KEY_VALIDATION": "false"}
        ):
            # Reset the global config manager
            _config_manager._config = None
            result = validate_environment()

            assert result is True

    def test_validate_environment_failure(self):
        """Test failed environment validation."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment()

            assert result is False

    def test_get_config_manager(self):
        """Test get_config_manager function."""
        manager = get_config_manager()

        assert isinstance(manager, ConfigManager)

    def test_create_config_template(self):
        """Test creating configuration template."""
        with TemporaryDirectory() as temp_dir:
            template_file = Path(temp_dir) / "config.template.json"

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                create_config_template(template_file, Environment.DEVELOPMENT)

            assert template_file.exists()

            with open(template_file) as f:
                template_data = json.load(f)

            assert template_data["environment"] == "development"
            assert template_data["log_level"] == "DEBUG"
            assert "openai_api_key" not in template_data


class TestEnvironmentSpecificBehavior:
    """Test environment-specific configuration behavior."""

    def test_development_environment(self):
        """Test development environment configuration."""
        with patch.dict(
            os.environ, {"ENVIRONMENT": "development", "OPENAI_API_KEY": "test-key"}
        ):
            config = AgentConfig()

            assert config.environment == Environment.DEVELOPMENT
            # Development allows higher limits
            assert config.max_iterations <= 20

    def test_testing_environment(self):
        """Test testing environment configuration."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "testing",
                "OPENAI_API_KEY": "test-key",
                "ENABLE_TRACING": "true",
                "LANGSMITH_API_KEY": "langsmith-key",  # Provide key to get past first validation
            },
        ):
            with pytest.raises(
                ConfigValidationError, match="Tracing should be disabled"
            ):
                AgentConfig()

    def test_production_environment(self):
        """Test production environment configuration."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "OPENAI_API_KEY": "test-key",
                "LANGSMITH_API_KEY": "langsmith-key",
                "SECURE_MODE": "true",
                "MAX_ITERATIONS": "60",
            },
        ):
            with pytest.raises(
                ConfigValidationError, match="should not exceed 50 in production"
            ):
                AgentConfig()

    def test_secure_mode_restrictions(self):
        """Test secure mode configuration restrictions."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "SECURE_MODE": "true",
                "TOOL_TIMEOUT": "120",
            },
        ):
            with pytest.raises(
                ConfigValidationError, match="should not exceed 60 seconds"
            ):
                AgentConfig()

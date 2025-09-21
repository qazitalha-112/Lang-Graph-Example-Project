"""Unit tests for tool registry operations."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.tools.tool_registry import ToolRegistry, ToolType, ToolInfo, BaseTool
from src.config import AgentConfig
from src.models.virtual_file_system import VirtualFileSystem


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str, description: str = "Mock tool"):
        self.name = name
        self.description = description

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_parameters(self) -> dict:
        return {"param1": "string", "param2": "optional_int"}

    def execute(self, **kwargs):
        return f"Executed {self.name} with {kwargs}"


class TestToolInfo:
    """Test cases for ToolInfo dataclass."""

    def test_tool_info_creation(self):
        """Test ToolInfo creation with all fields."""

        def mock_func():
            pass

        tool_info = ToolInfo(
            name="test_tool",
            tool_type=ToolType.SHARED,
            function=mock_func,
            description="Test tool description",
            required_params=["param1"],
            optional_params=["param2"],
            requires_api_key=True,
            api_service="TestAPI",
        )

        assert tool_info.name == "test_tool"
        assert tool_info.tool_type == ToolType.SHARED
        assert tool_info.function == mock_func
        assert tool_info.description == "Test tool description"
        assert tool_info.required_params == ["param1"]
        assert tool_info.optional_params == ["param2"]
        assert tool_info.requires_api_key is True
        assert tool_info.api_service == "TestAPI"

    def test_tool_info_defaults(self):
        """Test ToolInfo creation with default values."""

        def mock_func():
            pass

        tool_info = ToolInfo(
            name="simple_tool",
            tool_type=ToolType.ASSIGNABLE,
            function=mock_func,
            description="Simple tool",
        )

        assert tool_info.required_params == []
        assert tool_info.optional_params == []
        assert tool_info.requires_api_key is False
        assert tool_info.api_service is None


class TestBaseTool:
    """Test cases for BaseTool interface."""

    def test_mock_tool_implementation(self):
        """Test that MockTool properly implements BaseTool."""
        tool = MockTool("test_tool", "Test description")

        assert tool.get_name() == "test_tool"
        assert tool.get_description() == "Test description"
        assert isinstance(tool.get_parameters(), dict)
        assert "Executed test_tool" in tool.execute(param1="value1")


class TestToolRegistry:
    """Test cases for ToolRegistry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=AgentConfig)
        self.config.tavily_api_key = "test_tavily_key"
        self.config.firecrawl_api_key = "test_firecrawl_key"
        self.config.tool_timeout = 30

        self.vfs = Mock(spec=VirtualFileSystem)

        # Mock the tool creation functions to avoid external dependencies
        with (
            patch("src.tools.tool_registry.create_file_tools") as mock_file_tools,
            patch(
                "src.tools.tool_registry.create_assignable_tools"
            ) as mock_assignable_tools,
        ):
            # Setup mock file tools
            mock_file_tools.return_value = {
                "read_file": lambda path: f"content of {path}",
                "write_file": lambda path, content: {
                    "path": path,
                    "bytes_written": len(content),
                },
                "edit_file": lambda path, edits: {"path": path, "diff": "mock diff"},
            }

            # Setup mock assignable tools
            mock_assignable_tools.return_value = {
                "execute_code": lambda code: {
                    "status": "success",
                    "output": "mock output",
                },
                "search_internet": lambda query: [
                    {"title": "Mock Result", "url": "http://example.com"}
                ],
                "web_scrape": lambda url: {
                    "status": "success",
                    "content": "mock content",
                },
            }

            self.registry = ToolRegistry(self.config, self.vfs)

    def test_initialization(self):
        """Test registry initialization."""
        assert self.registry.config == self.config
        assert self.registry.vfs == self.vfs
        assert isinstance(self.registry.tools, dict)
        assert isinstance(self.registry.whitelists, dict)

    def test_get_tool_existing(self):
        """Test getting an existing tool."""
        tool_info = self.registry.get_tool("read_file")
        assert tool_info is not None
        assert tool_info.name == "read_file"
        assert tool_info.tool_type == ToolType.SHARED

    def test_get_tool_nonexistent(self):
        """Test getting a non-existent tool."""
        tool_info = self.registry.get_tool("nonexistent_tool")
        assert tool_info is None

    def test_get_all_tools(self):
        """Test getting all tools."""
        all_tools = self.registry.get_all_tools()
        assert isinstance(all_tools, dict)
        assert len(all_tools) > 0
        assert "read_file" in all_tools
        assert "execute_code" in all_tools

    def test_get_tools_by_type_shared(self):
        """Test getting shared tools."""
        shared_tools = self.registry.get_tools_by_type(ToolType.SHARED)
        assert isinstance(shared_tools, dict)
        assert "read_file" in shared_tools
        assert "write_file" in shared_tools
        assert "edit_file" in shared_tools

    def test_get_tools_by_type_assignable(self):
        """Test getting assignable tools."""
        assignable_tools = self.registry.get_tools_by_type(ToolType.ASSIGNABLE)
        assert isinstance(assignable_tools, dict)
        assert "execute_code" in assignable_tools

    def test_get_shared_tools(self):
        """Test getting shared tools convenience method."""
        shared_tools = self.registry.get_shared_tools()
        assert isinstance(shared_tools, dict)
        for tool in shared_tools.values():
            assert tool.tool_type == ToolType.SHARED

    def test_get_assignable_tools(self):
        """Test getting assignable tools convenience method."""
        assignable_tools = self.registry.get_assignable_tools()
        assert isinstance(assignable_tools, dict)
        for tool in assignable_tools.values():
            assert tool.tool_type == ToolType.ASSIGNABLE

    def test_get_available_assignable_tools(self):
        """Test getting available assignable tools."""
        available_tools = self.registry.get_available_assignable_tools()
        assert isinstance(available_tools, dict)
        # Should include execute_code (no API key required)
        assert "execute_code" in available_tools

    def test_register_custom_tool_success(self):
        """Test successful custom tool registration."""

        def custom_tool(param1: str, param2: int = 10):
            """Custom tool for testing."""
            return f"Custom: {param1}, {param2}"

        self.registry.register_custom_tool(
            name="custom_tool",
            tool_type=ToolType.ASSIGNABLE,
            function=custom_tool,
            description="A custom test tool",
        )

        tool_info = self.registry.get_tool("custom_tool")
        assert tool_info is not None
        assert tool_info.name == "custom_tool"
        assert tool_info.tool_type == ToolType.ASSIGNABLE
        assert tool_info.description == "A custom test tool"
        assert "param1" in tool_info.required_params
        assert "param2" in tool_info.optional_params

    def test_register_custom_tool_duplicate_name(self):
        """Test registering tool with duplicate name."""

        def custom_tool():
            pass

        # First registration should succeed
        self.registry.register_custom_tool(
            name="duplicate_tool", tool_type=ToolType.SHARED, function=custom_tool
        )

        # Second registration should fail
        with pytest.raises(ValueError, match="Tool 'duplicate_tool' already exists"):
            self.registry.register_custom_tool(
                name="duplicate_tool",
                tool_type=ToolType.ASSIGNABLE,
                function=custom_tool,
            )

    def test_register_custom_tool_invalid_name(self):
        """Test registering tool with invalid name."""

        def custom_tool():
            pass

        with pytest.raises(ValueError, match="Tool name cannot be empty"):
            self.registry.register_custom_tool(
                name="", tool_type=ToolType.SHARED, function=custom_tool
            )

        with pytest.raises(ValueError, match="Tool name cannot be empty"):
            self.registry.register_custom_tool(
                name="   ", tool_type=ToolType.SHARED, function=custom_tool
            )

    def test_register_custom_tool_invalid_function(self):
        """Test registering tool with invalid function."""
        with pytest.raises(ValueError, match="Tool function must be callable"):
            self.registry.register_custom_tool(
                name="invalid_tool",
                tool_type=ToolType.SHARED,
                function="not_a_function",
            )

    def test_validate_tool_assignment_valid(self):
        """Test validating valid tool assignment."""
        results = self.registry.validate_tool_assignment(["execute_code"])
        assert results["execute_code"] == "valid"

    def test_validate_tool_assignment_not_found(self):
        """Test validating non-existent tool."""
        results = self.registry.validate_tool_assignment(["nonexistent_tool"])
        assert results["nonexistent_tool"] == "not_found"

    def test_validate_tool_assignment_not_assignable(self):
        """Test validating shared tool (not assignable)."""
        results = self.registry.validate_tool_assignment(["read_file"])
        assert results["read_file"] == "not_assignable"

    def test_validate_tool_assignment_api_key_missing(self):
        """Test validating tool with missing API key."""
        # Create registry without API keys
        config_no_keys = Mock(spec=AgentConfig)
        config_no_keys.tavily_api_key = None
        config_no_keys.firecrawl_api_key = None
        config_no_keys.tool_timeout = 30

        with (
            patch("src.tools.tool_registry.create_file_tools") as mock_file_tools,
            patch(
                "src.tools.tool_registry.create_assignable_tools"
            ) as mock_assignable_tools,
        ):
            mock_file_tools.return_value = {}
            mock_assignable_tools.return_value = {"execute_code": lambda: None}

            registry_no_keys = ToolRegistry(config_no_keys, self.vfs)

            # Manually add a tool that requires API key for testing
            def mock_search(query):
                return []

            registry_no_keys._register_tool(
                name="search_internet",
                tool_type=ToolType.ASSIGNABLE,
                function=mock_search,
                description="Search tool",
                requires_api_key=True,
                api_service="Tavily",
            )

            results = registry_no_keys.validate_tool_assignment(["search_internet"])
            assert results["search_internet"] == "api_key_missing"

    def test_create_tool_whitelist_success(self):
        """Test successful tool whitelist creation."""
        result = self.registry.create_tool_whitelist("agent_1", ["execute_code"])

        assert result["agent_id"] == "agent_1"
        assert "execute_code" in result["whitelisted_tools"]
        assert "read_file" in result["whitelisted_tools"]  # Shared tools included
        assert result["validation_results"]["execute_code"] == "valid"

    def test_create_tool_whitelist_invalid_agent_id(self):
        """Test whitelist creation with invalid agent ID."""
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            self.registry.create_tool_whitelist("", ["execute_code"])

        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            self.registry.create_tool_whitelist("   ", ["execute_code"])

    def test_get_agent_whitelist_existing(self):
        """Test getting existing agent whitelist."""
        self.registry.create_tool_whitelist("agent_1", ["execute_code"])
        whitelist = self.registry.get_agent_whitelist("agent_1")

        assert whitelist is not None
        assert isinstance(whitelist, set)
        assert "execute_code" in whitelist

    def test_get_agent_whitelist_nonexistent(self):
        """Test getting non-existent agent whitelist."""
        whitelist = self.registry.get_agent_whitelist("nonexistent_agent")
        assert whitelist is None

    def test_get_agent_tools_success(self):
        """Test getting agent tools successfully."""
        self.registry.create_tool_whitelist("agent_1", ["execute_code"])
        agent_tools = self.registry.get_agent_tools("agent_1")

        assert isinstance(agent_tools, dict)
        assert "execute_code" in agent_tools
        assert "read_file" in agent_tools  # Shared tools included
        assert callable(agent_tools["execute_code"])

    def test_get_agent_tools_no_whitelist(self):
        """Test getting tools for agent without whitelist."""
        with pytest.raises(ValueError, match="No tool whitelist found for agent"):
            self.registry.get_agent_tools("nonexistent_agent")

    def test_remove_agent_whitelist_success(self):
        """Test successful agent whitelist removal."""
        self.registry.create_tool_whitelist("agent_1", ["execute_code"])

        # Verify whitelist exists
        assert self.registry.get_agent_whitelist("agent_1") is not None

        # Remove whitelist
        result = self.registry.remove_agent_whitelist("agent_1")
        assert result is True

        # Verify whitelist is gone
        assert self.registry.get_agent_whitelist("agent_1") is None

    def test_remove_agent_whitelist_nonexistent(self):
        """Test removing non-existent agent whitelist."""
        result = self.registry.remove_agent_whitelist("nonexistent_agent")
        assert result is False

    def test_get_tools_for_task_type_web_testing(self):
        """Test getting tools for web testing task."""
        tools = self.registry.get_tools_for_task_type("web_testing")
        assert isinstance(tools, list)
        # Should recommend web_scrape and execute_code if available

    def test_get_tools_for_task_type_research(self):
        """Test getting tools for research task."""
        tools = self.registry.get_tools_for_task_type("research")
        assert isinstance(tools, list)
        # Should recommend search_internet and web_scrape if available

    def test_get_tools_for_task_type_unknown(self):
        """Test getting tools for unknown task type."""
        tools = self.registry.get_tools_for_task_type("unknown_task")
        assert isinstance(tools, list)
        # Should default to execute_code

    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        stats = self.registry.get_registry_stats()

        assert isinstance(stats, dict)
        assert "total_tools" in stats
        assert "shared_tools" in stats
        assert "assignable_tools" in stats
        assert "available_assignable_tools" in stats
        assert "active_agents" in stats
        assert "tools_requiring_api_keys" in stats
        assert "tool_names" in stats

        assert isinstance(stats["total_tools"], int)
        assert stats["total_tools"] > 0
        assert isinstance(stats["tool_names"], list)

    def test_parameter_extraction(self):
        """Test parameter extraction from function signatures."""

        def test_func(required_param: str, optional_param: int = 10, *args, **kwargs):
            """Test function."""
            pass

        self.registry.register_custom_tool(
            name="param_test_tool", tool_type=ToolType.ASSIGNABLE, function=test_func
        )

        tool_info = self.registry.get_tool("param_test_tool")
        assert "required_param" in tool_info.required_params
        assert "optional_param" in tool_info.optional_params
        # *args and **kwargs should not be included in required params

    def test_function_description_extraction(self):
        """Test description extraction from function docstrings."""

        def documented_func():
            """This is a documented function.

            It has multiple lines in the docstring.
            """
            pass

        def undocumented_func():
            pass

        # Test with docstring
        description1 = self.registry._get_function_description(documented_func)
        assert description1 == "This is a documented function."

        # Test without docstring
        description2 = self.registry._get_function_description(undocumented_func)
        assert "undocumented_func" in description2

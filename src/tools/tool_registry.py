"""Tool registry system for managing and assigning tools to subagents."""

from typing import Dict, List, Set, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
import functools
from ..config import AgentConfig
from ..models.virtual_file_system import VirtualFileSystem
from .file_tools import create_file_tools
from .assignable_tools import create_assignable_tools


class ToolType(Enum):
    """Types of tools available in the registry."""

    SHARED = "shared"  # Available to all agents
    ASSIGNABLE = "assignable"  # Can be selectively assigned to subagents


@dataclass
class ToolInfo:
    """Information about a registered tool."""

    name: str
    tool_type: ToolType
    function: Callable
    description: str
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    requires_api_key: bool = False
    api_service: Optional[str] = None


class BaseTool(ABC):
    """Base interface for consistent tool implementation."""

    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get the tool description."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the tool parameters schema."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass


class ToolRegistry:
    """
    Registry for managing tools and their assignment to subagents.

    Provides centralized tool management, validation, and selective assignment
    capabilities for the supervisor agent system.
    """

    def __init__(self, config: AgentConfig, vfs: VirtualFileSystem):
        """
        Initialize the tool registry.

        Args:
            config: Agent configuration
            vfs: Virtual file system instance
        """
        self.config = config
        self.vfs = vfs
        self.tools: Dict[str, ToolInfo] = {}
        self.whitelists: Dict[str, Set[str]] = {}  # agent_id -> set of tool names
        self._tracer = None  # Will be set by supervisor when available

        # Initialize with default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default shared and assignable tools."""
        # Register shared file tools
        file_tools = create_file_tools(self.vfs)
        for name, func in file_tools.items():
            self._register_tool(
                name=name,
                tool_type=ToolType.SHARED,
                function=func,
                description=self._get_function_description(func),
                required_params=self._extract_required_params(func),
                optional_params=self._extract_optional_params(func),
            )

        # Register assignable tools
        try:
            assignable_tools = create_assignable_tools(self.config)
            for name, func in assignable_tools.items():
                requires_api = name in ["search_internet", "web_scrape"]
                api_service = None
                if name == "search_internet":
                    api_service = "Tavily"
                elif name == "web_scrape":
                    api_service = "Firecrawl"

                self._register_tool(
                    name=name,
                    tool_type=ToolType.ASSIGNABLE,
                    function=func,
                    description=self._get_function_description(func),
                    required_params=self._extract_required_params(func),
                    optional_params=self._extract_optional_params(func),
                    requires_api_key=requires_api,
                    api_service=api_service,
                )
        except Exception as e:
            # Some assignable tools might not be available due to missing API keys
            print(f"Warning: Some assignable tools not available: {e}")

    def _register_tool(
        self,
        name: str,
        tool_type: ToolType,
        function: Callable,
        description: str,
        required_params: List[str] = None,
        optional_params: List[str] = None,
        requires_api_key: bool = False,
        api_service: Optional[str] = None,
    ) -> None:
        """Register a tool in the registry."""
        tool_info = ToolInfo(
            name=name,
            tool_type=tool_type,
            function=function,
            description=description,
            required_params=required_params or [],
            optional_params=optional_params or [],
            requires_api_key=requires_api_key,
            api_service=api_service,
        )
        self.tools[name] = tool_info

    def _get_function_description(self, func: Callable) -> str:
        """Extract description from function docstring."""
        if func.__doc__:
            # Get first line of docstring
            return func.__doc__.strip().split("\n")[0]
        return f"Tool function: {func.__name__}"

    def _extract_required_params(self, func: Callable) -> List[str]:
        """Extract required parameters from function signature."""
        sig = inspect.signature(func)
        required = []

        for param_name, param in sig.parameters.items():
            if (
                param.default == inspect.Parameter.empty
                and param.kind != inspect.Parameter.VAR_KEYWORD
            ):
                required.append(param_name)

        return required

    def _extract_optional_params(self, func: Callable) -> List[str]:
        """Extract optional parameters from function signature."""
        sig = inspect.signature(func)
        optional = []

        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                optional.append(param_name)

        return optional

    def register_custom_tool(
        self,
        name: str,
        tool_type: ToolType,
        function: Callable,
        description: str = None,
        requires_api_key: bool = False,
        api_service: Optional[str] = None,
    ) -> None:
        """
        Register a custom tool.

        Args:
            name: Tool name
            tool_type: Type of tool (shared or assignable)
            function: Tool function
            description: Tool description
            requires_api_key: Whether tool requires API key
            api_service: Name of API service if applicable

        Raises:
            ValueError: If tool name already exists or is invalid
        """
        if not name or not name.strip():
            raise ValueError("Tool name cannot be empty")

        if name in self.tools:
            raise ValueError(f"Tool '{name}' already exists")

        if not callable(function):
            raise ValueError("Tool function must be callable")

        self._register_tool(
            name=name,
            tool_type=tool_type,
            function=function,
            description=description or self._get_function_description(function),
            required_params=self._extract_required_params(function),
            optional_params=self._extract_optional_params(function),
            requires_api_key=requires_api_key,
            api_service=api_service,
        )

    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """
        Get tool information by name.

        Args:
            name: Tool name

        Returns:
            ToolInfo object or None if not found
        """
        return self.tools.get(name)

    def get_all_tools(self) -> Dict[str, ToolInfo]:
        """Get all registered tools."""
        return self.tools.copy()

    def get_tools_by_type(self, tool_type: ToolType) -> Dict[str, ToolInfo]:
        """
        Get tools by type.

        Args:
            tool_type: Type of tools to retrieve

        Returns:
            Dictionary of tools of the specified type
        """
        return {
            name: tool
            for name, tool in self.tools.items()
            if tool.tool_type == tool_type
        }

    def get_shared_tools(self) -> Dict[str, ToolInfo]:
        """Get all shared tools (available to all agents)."""
        return self.get_tools_by_type(ToolType.SHARED)

    def get_assignable_tools(self) -> Dict[str, ToolInfo]:
        """Get all assignable tools."""
        return self.get_tools_by_type(ToolType.ASSIGNABLE)

    def get_available_assignable_tools(self) -> Dict[str, ToolInfo]:
        """Get assignable tools that are actually available (have required API keys)."""
        assignable = self.get_assignable_tools()
        available = {}

        for name, tool in assignable.items():
            if not tool.requires_api_key:
                available[name] = tool
            elif tool.api_service == "Tavily" and self.config.tavily_api_key:
                available[name] = tool
            elif tool.api_service == "Firecrawl" and self.config.firecrawl_api_key:
                available[name] = tool

        return available

    def validate_tool_assignment(self, tool_names: List[str]) -> Dict[str, str]:
        """
        Validate a list of tool names for assignment.

        Args:
            tool_names: List of tool names to validate

        Returns:
            Dictionary with validation results: {tool_name: status}
            Status can be: "valid", "not_found", "not_assignable", "api_key_missing"
        """
        results = {}

        for tool_name in tool_names:
            if tool_name not in self.tools:
                results[tool_name] = "not_found"
                continue

            tool = self.tools[tool_name]

            if tool.tool_type == ToolType.SHARED:
                results[tool_name] = "not_assignable"
                continue

            if tool.requires_api_key:
                if tool.api_service == "Tavily" and not self.config.tavily_api_key:
                    results[tool_name] = "api_key_missing"
                    continue
                elif (
                    tool.api_service == "Firecrawl"
                    and not self.config.firecrawl_api_key
                ):
                    results[tool_name] = "api_key_missing"
                    continue

            results[tool_name] = "valid"

        return results

    def create_tool_whitelist(
        self, agent_id: str, tool_names: List[str]
    ) -> Dict[str, Any]:
        """
        Create a tool whitelist for a specific agent.

        Args:
            agent_id: Unique identifier for the agent
            tool_names: List of tool names to whitelist

        Returns:
            Dictionary with whitelist creation results

        Raises:
            ValueError: If agent_id is invalid
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")

        # Validate tools
        validation_results = self.validate_tool_assignment(tool_names)

        # Only include valid tools in whitelist
        valid_tools = {
            name for name, status in validation_results.items() if status == "valid"
        }

        # Always include shared tools
        shared_tools = set(self.get_shared_tools().keys())

        # Create final whitelist
        self.whitelists[agent_id] = shared_tools | valid_tools

        return {
            "agent_id": agent_id,
            "whitelisted_tools": list(self.whitelists[agent_id]),
            "validation_results": validation_results,
            "shared_tools_included": list(shared_tools),
            "assignable_tools_included": list(valid_tools),
        }

    def get_agent_whitelist(self, agent_id: str) -> Optional[Set[str]]:
        """
        Get the tool whitelist for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Set of whitelisted tool names or None if agent not found
        """
        return self.whitelists.get(agent_id)

    def set_tracer(self, tracer) -> None:
        """Set the tracer for tool usage tracking."""
        self._tracer = tracer

    def _wrap_tool_with_tracing(
        self, tool_name: str, tool_function: Callable, agent_id: Optional[str] = None
    ) -> Callable:
        """Wrap a tool function with tracing if tracer is available."""
        if not self._tracer or not self._tracer.is_enabled():
            return tool_function

        def traced_tool(*args, **kwargs):
            # Apply tracing decorator
            @self._tracer.tool_trace(
                tool_name=tool_name,
                agent_id=agent_id,
                metadata={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
            )
            def _execute_tool():
                return tool_function(*args, **kwargs)

            return _execute_tool()

        # Set the function name for testing
        traced_tool.__name__ = "traced_tool"
        return traced_tool

    def get_agent_tools(self, agent_id: str) -> Dict[str, Callable]:
        """
        Get the actual tool functions for a specific agent based on their whitelist.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary of tool functions the agent can use

        Raises:
            ValueError: If agent has no whitelist
        """
        whitelist = self.get_agent_whitelist(agent_id)
        if whitelist is None:
            raise ValueError(f"No tool whitelist found for agent '{agent_id}'")

        agent_tools = {}
        for tool_name in whitelist:
            if tool_name in self.tools:
                # Wrap tool with tracing
                original_function = self.tools[tool_name].function
                traced_function = self._wrap_tool_with_tracing(
                    tool_name, original_function, agent_id
                )
                agent_tools[tool_name] = traced_function

        return agent_tools

    def remove_agent_whitelist(self, agent_id: str) -> bool:
        """
        Remove the tool whitelist for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            True if whitelist was removed, False if agent not found
        """
        if agent_id in self.whitelists:
            del self.whitelists[agent_id]
            return True
        return False

    def get_tools_for_task_type(self, task_type: str) -> List[str]:
        """
        Get recommended tools for a specific task type.

        Args:
            task_type: Type of task (e.g., "web_testing", "research", "analysis")

        Returns:
            List of recommended tool names
        """
        # Define task type to tool mappings
        task_mappings = {
            "web_testing": ["web_scrape", "execute_code"],
            "research": ["search_internet", "web_scrape"],
            "analysis": ["execute_code"],
            "code_analysis": ["execute_code"],
            "data_processing": ["execute_code"],
            "web_scraping": ["web_scrape"],
            "internet_search": ["search_internet"],
            "file_processing": [],  # Only shared file tools
            "general": ["execute_code"],
        }

        recommended = task_mappings.get(task_type.lower(), ["execute_code"])

        # Filter to only include available tools
        available_assignable = set(self.get_available_assignable_tools().keys())

        return [tool for tool in recommended if tool in available_assignable]

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tool registry.

        Returns:
            Dictionary with registry statistics
        """
        shared_tools = self.get_shared_tools()
        assignable_tools = self.get_assignable_tools()
        available_assignable = self.get_available_assignable_tools()

        return {
            "total_tools": len(self.tools),
            "shared_tools": len(shared_tools),
            "assignable_tools": len(assignable_tools),
            "available_assignable_tools": len(available_assignable),
            "active_agents": len(self.whitelists),
            "tools_requiring_api_keys": len(
                [t for t in self.tools.values() if t.requires_api_key]
            ),
            "tool_names": list(self.tools.keys()),
            "shared_tool_names": list(shared_tools.keys()),
            "assignable_tool_names": list(assignable_tools.keys()),
            "available_assignable_tool_names": list(available_assignable.keys()),
        }

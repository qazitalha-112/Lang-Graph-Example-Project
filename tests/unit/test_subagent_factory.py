"""Unit tests for the SubAgent factory and prompt generation system."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import uuid

from src.agents.subagent_factory import SubAgentFactory, PromptTemplate
from src.models.data_models import Task, TaskResult, SubAgent, TaskStatus, TaskType
from src.tools.tool_registry import ToolRegistry, ToolType, ToolInfo
from src.models.virtual_file_system import VirtualFileSystem
from src.config import AgentConfig


class TestPromptTemplate:
    """Test cases for the PromptTemplate class."""

    def test_generate_basic_prompt(self):
        """Test basic prompt generation with minimal inputs."""
        task = Task(
            id="test_task_1",
            description="Test task description",
            task_type="general",
            priority=1,
            success_criteria="Complete successfully",
        )

        available_tools = ["read_file", "write_file"]
        tool_descriptions = {
            "read_file": "Read content from a file",
            "write_file": "Write content to a file",
        }

        prompt = PromptTemplate.generate_prompt(
            task=task,
            available_tools=available_tools,
            tool_descriptions=tool_descriptions,
        )

        # Verify key components are in the prompt
        assert "Test task description" in prompt
        assert "general" in prompt
        assert "Complete successfully" in prompt
        assert "read_file: Read content from a file" in prompt
        assert "write_file: Write content to a file" in prompt
        assert "**Task Status:**" in prompt

    def test_generate_prompt_with_context(self):
        """Test prompt generation with previous context and file system state."""
        task = Task(
            id="test_task_2",
            description="Analysis task",
            task_type="analysis",
            priority=1,
        )

        available_tools = ["execute_code"]
        tool_descriptions = {"execute_code": "Execute Python code"}
        previous_context = "Previous task completed successfully"
        file_system_context = "Files: report.md, data.json"

        prompt = PromptTemplate.generate_prompt(
            task=task,
            available_tools=available_tools,
            tool_descriptions=tool_descriptions,
            previous_context=previous_context,
            file_system_context=file_system_context,
        )

        assert "Previous task completed successfully" in prompt
        assert "Files: report.md, data.json" in prompt

    def test_task_specific_templates(self):
        """Test that task-specific templates are included for known task types."""
        task_types = [
            "web_testing",
            "research",
            "analysis",
            "code_execution",
            "file_operation",
        ]

        for task_type in task_types:
            task = Task(
                id=f"test_{task_type}",
                description=f"Test {task_type} task",
                task_type=task_type,
                priority=1,
            )

            prompt = PromptTemplate.generate_prompt(
                task=task,
                available_tools=["execute_code"],
                tool_descriptions={"execute_code": "Execute code"},
            )

            # Each task type should have specific instructions
            assert (
                f"{task_type.replace('_', ' ').title()} Specific Instructions" in prompt
            )

    def test_unknown_task_type(self):
        """Test prompt generation for unknown task types."""
        task = Task(
            id="test_unknown",
            description="Unknown task type",
            task_type="unknown_type",
            priority=1,
        )

        prompt = PromptTemplate.generate_prompt(
            task=task,
            available_tools=["execute_code"],
            tool_descriptions={"execute_code": "Execute code"},
        )

        # Should still generate a valid prompt without task-specific instructions
        assert "Unknown task type" in prompt
        assert "unknown_type" in prompt


class TestSubAgentFactory:
    """Test cases for the SubAgentFactory class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=AgentConfig)
        config.llm_model = "gpt-4"
        config.tool_timeout = 30
        config.max_subagents = 5
        return config

    @pytest.fixture
    def mock_vfs(self):
        """Create a mock virtual file system."""
        vfs = Mock(spec=VirtualFileSystem)
        vfs.list_files.return_value = ["test.txt", "data.json"]
        vfs.read_file.return_value = "Test file content"
        return vfs

    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock tool registry."""
        registry = Mock(spec=ToolRegistry)

        # Mock tool info
        execute_code_tool = ToolInfo(
            name="execute_code",
            tool_type=ToolType.ASSIGNABLE,
            function=Mock(),
            description="Execute Python code",
        )

        read_file_tool = ToolInfo(
            name="read_file",
            tool_type=ToolType.SHARED,
            function=Mock(),
            description="Read file content",
        )

        registry.get_tool.side_effect = lambda name: {
            "execute_code": execute_code_tool,
            "read_file": read_file_tool,
        }.get(name)

        registry.get_tools_for_task_type.return_value = ["execute_code"]
        registry.get_available_assignable_tools.return_value = {
            "execute_code": execute_code_tool
        }

        registry.create_tool_whitelist.return_value = {
            "agent_id": "test_agent",
            "whitelisted_tools": ["read_file", "write_file", "execute_code"],
            "validation_results": {"execute_code": "valid"},
            "shared_tools_included": ["read_file", "write_file"],
            "assignable_tools_included": ["execute_code"],
        }

        return registry

    @pytest.fixture
    def factory(self, mock_config, mock_tool_registry, mock_vfs):
        """Create a SubAgentFactory instance with mocked dependencies."""
        return SubAgentFactory(mock_config, mock_tool_registry, mock_vfs)

    def test_factory_initialization(self, mock_config, mock_tool_registry, mock_vfs):
        """Test factory initialization."""
        factory = SubAgentFactory(mock_config, mock_tool_registry, mock_vfs)

        assert factory.config == mock_config
        assert factory.tool_registry == mock_tool_registry
        assert factory.vfs == mock_vfs
        assert factory.created_agents == {}

    def test_create_agent_basic(self, factory):
        """Test basic agent creation."""
        task = Task(
            id="test_task",
            description="Test task",
            task_type="general",
            priority=1,
        )

        agent = factory.create_agent(task)

        assert isinstance(agent, SubAgent)
        assert agent.task == task
        assert agent.status == "created"
        assert len(agent.available_tools) > 0
        assert agent.prompt is not None
        assert len(agent.prompt) > 0
        assert agent.id in factory.created_agents

    def test_create_agent_with_custom_tools(self, factory):
        """Test agent creation with custom tool assignment."""
        task = Task(
            id="test_task",
            description="Test task",
            task_type="general",
            priority=1,
        )

        custom_tools = ["execute_code"]
        agent = factory.create_agent(task, custom_tools=custom_tools)

        # Verify the tool registry was called with custom tools
        factory.tool_registry.create_tool_whitelist.assert_called()
        call_args = factory.tool_registry.create_tool_whitelist.call_args
        assert custom_tools == call_args[0][1]  # Second argument should be custom_tools

    def test_create_agent_with_previous_results(self, factory):
        """Test agent creation with previous task results for context."""
        task = Task(
            id="test_task",
            description="Test task",
            task_type="analysis",
            priority=1,
        )

        previous_results = [
            TaskResult(
                task_id="prev_task_1",
                status=TaskStatus.COMPLETED.value,
                output="Previous task completed successfully",
                artifacts=["report.md"],
            ),
            TaskResult(
                task_id="prev_task_2",
                status=TaskStatus.COMPLETED.value,
                output="Another task result with longer output that should be truncated"
                * 10,
                artifacts=["data.json"],
            ),
        ]

        agent = factory.create_agent(task, previous_results=previous_results)

        # Verify context is included in prompt
        assert "Previous task completed successfully" in agent.prompt
        assert "prev_task_1" in agent.prompt

    def test_create_agent_invalid_task(self, factory):
        """Test agent creation with invalid task."""
        with pytest.raises(ValueError, match="Task must be a Task instance"):
            factory.create_agent("not_a_task")

    def test_determine_tools_for_task(self, factory):
        """Test tool determination logic for different task types."""
        # Test with required tools
        task_with_required = Task(
            id="test_task",
            description="Test task",
            task_type="general",
            priority=1,
            required_tools=["execute_code"],
        )

        tools = factory._determine_tools_for_task(task_with_required)
        assert "execute_code" in tools

        # Test with task type mapping
        task_research = Task(
            id="research_task",
            description="Research task",
            task_type="research",
            priority=1,
        )

        factory.tool_registry.get_tools_for_task_type.return_value = ["search_internet"]
        factory.tool_registry.get_available_assignable_tools.return_value = {
            "search_internet": Mock()
        }

        tools = factory._determine_tools_for_task(task_research)
        factory.tool_registry.get_tools_for_task_type.assert_called_with("research")

    def test_get_tool_descriptions(self, factory):
        """Test tool description retrieval."""
        tool_names = ["execute_code", "read_file"]

        descriptions = factory._get_tool_descriptions(tool_names)

        assert "execute_code" in descriptions
        assert "read_file" in descriptions
        assert descriptions["execute_code"] == "Execute Python code"
        assert descriptions["read_file"] == "Read file content"

    def test_generate_previous_context_empty(self, factory):
        """Test previous context generation with no results."""
        context = factory._generate_previous_context([])
        assert "No previous tasks have been completed" in context

    def test_generate_previous_context_with_results(self, factory):
        """Test previous context generation with task results."""
        results = [
            TaskResult(
                task_id="task1",
                status=TaskStatus.COMPLETED.value,
                output="Short output",
                artifacts=["file1.txt"],
            ),
            TaskResult(
                task_id="task2",
                status=TaskStatus.FAILED.value,
                output="A" * 300,  # Long output that should be truncated
                artifacts=[],
            ),
        ]

        context = factory._generate_previous_context(results)

        assert "task1" in context
        assert "task2" in context
        assert "Short output" in context
        assert "file1.txt" in context
        assert "..." in context  # Truncation indicator

    def test_generate_file_system_context(self, factory):
        """Test file system context generation."""
        context = factory._generate_file_system_context()

        factory.vfs.list_files.assert_called_once()
        factory.vfs.read_file.assert_called()
        assert "test.txt" in context
        assert "data.json" in context

    def test_generate_file_system_context_empty(self, factory):
        """Test file system context generation when empty."""
        factory.vfs.list_files.return_value = []

        context = factory._generate_file_system_context()

        assert "empty" in context.lower()

    def test_generate_file_system_context_error(self, factory):
        """Test file system context generation with errors."""
        factory.vfs.list_files.side_effect = Exception("VFS Error")

        context = factory._generate_file_system_context()

        assert "Error accessing file system" in context

    def test_get_agent(self, factory):
        """Test retrieving created agents."""
        task = Task(
            id="test_task",
            description="Test task",
            task_type="general",
            priority=1,
        )

        agent = factory.create_agent(task)
        retrieved_agent = factory.get_agent(agent.id)

        assert retrieved_agent == agent

        # Test non-existent agent
        assert factory.get_agent("non_existent") is None

    def test_get_all_agents(self, factory):
        """Test retrieving all created agents."""
        task1 = Task(id="task1", description="Task 1", task_type="general", priority=1)
        task2 = Task(id="task2", description="Task 2", task_type="research", priority=2)

        agent1 = factory.create_agent(task1)
        agent2 = factory.create_agent(task2)

        all_agents = factory.get_all_agents()

        assert len(all_agents) == 2
        assert agent1.id in all_agents
        assert agent2.id in all_agents

    def test_update_agent_status(self, factory):
        """Test updating agent status."""
        task = Task(
            id="test_task",
            description="Test task",
            task_type="general",
            priority=1,
        )

        agent = factory.create_agent(task)
        assert agent.status == "created"

        # Update status
        result = factory.update_agent_status(agent.id, "running")
        assert result is True
        assert factory.created_agents[agent.id].status == "running"

        # Try to update non-existent agent
        result = factory.update_agent_status("non_existent", "completed")
        assert result is False

    def test_cleanup_agent(self, factory):
        """Test agent cleanup."""
        task = Task(
            id="test_task",
            description="Test task",
            task_type="general",
            priority=1,
        )

        agent = factory.create_agent(task)
        agent_id = agent.id

        # Verify agent exists
        assert agent_id in factory.created_agents

        # Cleanup agent
        result = factory.cleanup_agent(agent_id)
        assert result is True
        assert agent_id not in factory.created_agents

        # Verify tool registry cleanup was called
        factory.tool_registry.remove_agent_whitelist.assert_called_with(agent_id)

        # Try to cleanup non-existent agent
        result = factory.cleanup_agent("non_existent")
        assert result is False

    def test_get_factory_stats(self, factory):
        """Test factory statistics generation."""
        # Create agents with different statuses and task types
        task1 = Task(id="task1", description="Task 1", task_type="research", priority=1)
        task2 = Task(id="task2", description="Task 2", task_type="analysis", priority=2)
        task3 = Task(id="task3", description="Task 3", task_type="research", priority=3)

        agent1 = factory.create_agent(task1)
        agent2 = factory.create_agent(task2)
        agent3 = factory.create_agent(task3)

        # Update some statuses
        factory.update_agent_status(agent1.id, "running")
        factory.update_agent_status(agent2.id, "completed")

        stats = factory.get_factory_stats()

        assert stats["total_agents_created"] == 3
        assert stats["agents_by_status"]["created"] == 1
        assert stats["agents_by_status"]["running"] == 1
        assert stats["agents_by_status"]["completed"] == 1
        assert stats["agents_by_task_type"]["research"] == 2
        assert stats["agents_by_task_type"]["analysis"] == 1
        assert len(stats["active_agent_ids"]) == 3

    @patch("uuid.uuid4")
    def test_agent_id_generation(self, mock_uuid, factory):
        """Test that agent IDs are generated correctly."""
        mock_uuid.return_value.hex = "abcd1234" * 4  # 32 chars

        task = Task(
            id="test_task",
            description="Test task",
            task_type="general",
            priority=1,
        )

        agent = factory.create_agent(task)

        expected_id = "agent_test_task_abcd1234"
        assert agent.id == expected_id

    def test_llm_config_generation(self, factory):
        """Test that LLM configuration is properly set."""
        task = Task(
            id="test_task",
            description="Test task",
            task_type="general",
            priority=1,
        )

        agent = factory.create_agent(task)

        assert agent.llm_config["model"] == "gpt-4"
        assert agent.llm_config["temperature"] == 0.1
        assert agent.llm_config["max_tokens"] == 4000
        assert agent.llm_config["timeout"] == 30

    def test_tool_assignment_warnings(self, factory, capfd):
        """Test that warnings are printed for failed tool assignments."""
        # Mock validation results with failures
        factory.tool_registry.create_tool_whitelist.return_value = {
            "agent_id": "test_agent",
            "whitelisted_tools": ["read_file", "write_file"],
            "validation_results": {
                "execute_code": "valid",
                "search_internet": "api_key_missing",
                "invalid_tool": "not_found",
            },
            "shared_tools_included": ["read_file", "write_file"],
            "assignable_tools_included": ["execute_code"],
        }

        task = Task(
            id="test_task",
            description="Test task",
            task_type="general",
            priority=1,
        )

        factory.create_agent(
            task, custom_tools=["execute_code", "search_internet", "invalid_tool"]
        )

        # Check that warning was printed
        captured = capfd.readouterr()
        assert "Warning: Some tools could not be assigned" in captured.out
        assert "search_internet" in captured.out
        assert "invalid_tool" in captured.out

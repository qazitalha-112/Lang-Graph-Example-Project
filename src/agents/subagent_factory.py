"""SubAgent factory for dynamic agent creation with task-specific prompts and tools."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import json

from ..models.data_models import Task, SubAgent, TaskResult
from ..tools.tool_registry import ToolRegistry
from ..models.virtual_file_system import VirtualFileSystem
from ..config import AgentConfig


class PromptTemplate:
    """Template for generating task-specific prompts."""

    BASE_TEMPLATE = """You are a specialized AI assistant created to execute a specific task within a larger objective.

## Your Task
{task_description}

## Task Type
{task_type}

## Success Criteria
{success_criteria}

## Available Tools
{available_tools}

## Context from Previous Tasks
{previous_context}

## File System State
{file_system_context}

## Instructions
1. Focus solely on completing the assigned task
2. Use the available tools effectively to gather information and perform actions
3. Create artifacts using write_file when your work produces deliverable content
4. Provide clear, structured output that can be used by the supervisor
5. If you encounter errors, explain what went wrong and suggest alternatives

## Constraints
- Only use the tools listed in the "Available Tools" section
- Stay within the scope of your assigned task
- Do not attempt to modify the overall plan or execute other tasks
- Ensure all file operations use appropriate paths and naming conventions

## Output Format
Provide your final result in this format:

**Task Status:** [COMPLETED/FAILED/PARTIAL]
**Summary:** [Brief summary of what was accomplished]
**Artifacts Created:** [List of files created, if any]
**Key Findings:** [Important information discovered]
**Recommendations:** [Any suggestions for follow-up tasks]

Begin working on your task now."""

    TASK_SPECIFIC_TEMPLATES = {
        "web_testing": """
## Web Testing Specific Instructions
- Use web_scrape to examine web pages and forms
- Use execute_code to write test scripts or analyze responses
- Document any bugs, issues, or unexpected behavior found
- Create a structured test report with findings
""",
        "research": """
## Research Specific Instructions
- Use search_internet to find relevant information
- Use web_scrape to extract detailed content from sources
- Synthesize information from multiple sources
- Create a comprehensive research report with citations
""",
        "analysis": """
## Analysis Specific Instructions
- Use execute_code to perform data analysis or code examination
- Break down complex problems into smaller components
- Provide detailed explanations of your analytical process
- Create visualizations or summaries as appropriate
""",
        "code_execution": """
## Code Execution Specific Instructions
- Use execute_code to run and test code
- Ensure code is properly formatted and documented
- Test edge cases and error conditions
- Provide clear output and error handling
""",
        "file_operation": """
## File Operation Specific Instructions
- Use read_file, write_file, and edit_file as needed
- Maintain proper file organization and naming
- Ensure file operations are atomic and safe
- Document any file structure changes made
""",
    }

    @classmethod
    def generate_prompt(
        cls,
        task: Task,
        available_tools: List[str],
        tool_descriptions: Dict[str, str],
        previous_context: str = "",
        file_system_context: str = "",
    ) -> str:
        """
        Generate a task-specific prompt for a subagent.

        Args:
            task: The task to be executed
            available_tools: List of tool names available to the agent
            tool_descriptions: Descriptions of available tools
            previous_context: Context from previous task executions
            file_system_context: Current state of the file system

        Returns:
            Generated prompt string
        """
        # Format available tools with descriptions
        tools_section = "\n".join(
            [
                f"- {tool}: {tool_descriptions.get(tool, 'No description available')}"
                for tool in available_tools
            ]
        )

        # Get task-specific instructions
        task_specific = cls.TASK_SPECIFIC_TEMPLATES.get(task.task_type, "")

        # Format the base template
        prompt = cls.BASE_TEMPLATE.format(
            task_description=task.description,
            task_type=task.task_type,
            success_criteria=task.success_criteria or "Complete the task successfully",
            available_tools=tools_section,
            previous_context=previous_context or "No previous context available",
            file_system_context=file_system_context or "File system is empty",
        )

        # Add task-specific instructions if available
        if task_specific:
            prompt += task_specific

        return prompt


class SubAgentFactory:
    """
    Factory for creating specialized subagents with task-specific prompts and tools.

    This factory dynamically creates subagents at runtime, assigns appropriate tools
    based on task requirements, and generates tailored prompts that include context
    from previous tasks and the current file system state.
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        vfs: VirtualFileSystem,
    ):
        """
        Initialize the SubAgent factory.

        Args:
            config: Agent configuration
            tool_registry: Registry for tool management and assignment
            vfs: Virtual file system for context injection
        """
        self.config = config
        self.tool_registry = tool_registry
        self.vfs = vfs
        self.created_agents: Dict[str, SubAgent] = {}

    def create_agent(
        self,
        task: Task,
        previous_results: List[TaskResult] = None,
        custom_tools: List[str] = None,
    ) -> SubAgent:
        """
        Create a specialized subagent for the given task.

        Args:
            task: The task to be executed by the subagent
            previous_results: Results from previously completed tasks
            custom_tools: Custom list of tools to assign (overrides automatic selection)

        Returns:
            Created SubAgent instance

        Raises:
            ValueError: If task is invalid or tool assignment fails
        """
        if not isinstance(task, Task):
            raise ValueError("Task must be a Task instance")

        # Generate unique agent ID
        agent_id = f"agent_{task.id}_{uuid.uuid4().hex[:8]}"

        # Determine tools to assign
        if custom_tools:
            assigned_tools = custom_tools
        else:
            assigned_tools = self._determine_tools_for_task(task)

        # Validate and create tool whitelist
        whitelist_result = self.tool_registry.create_tool_whitelist(
            agent_id, assigned_tools
        )

        # Check for any tool assignment issues
        validation_results = whitelist_result["validation_results"]
        failed_tools = [
            tool for tool, status in validation_results.items() if status != "valid"
        ]

        if failed_tools:
            print(
                f"Warning: Some tools could not be assigned to agent {agent_id}: {failed_tools}"
            )

        # Get tool descriptions for prompt generation
        tool_descriptions = self._get_tool_descriptions(
            whitelist_result["whitelisted_tools"]
        )

        # Generate context from previous tasks
        previous_context = self._generate_previous_context(previous_results or [])

        # Generate file system context
        file_system_context = self._generate_file_system_context()

        # Generate the agent prompt
        prompt = PromptTemplate.generate_prompt(
            task=task,
            available_tools=whitelist_result["whitelisted_tools"],
            tool_descriptions=tool_descriptions,
            previous_context=previous_context,
            file_system_context=file_system_context,
        )

        # Create LLM configuration
        llm_config = {
            "model": self.config.llm_model,
            "temperature": 0.1,  # Lower temperature for more focused task execution
            "max_tokens": 4000,
            "timeout": self.config.tool_timeout,
        }

        # Create the subagent
        subagent = SubAgent(
            id=agent_id,
            task=task,
            prompt=prompt,
            available_tools=whitelist_result["whitelisted_tools"],
            llm_config=llm_config,
            created_at=datetime.now(),
            status="created",
        )

        # Store the created agent
        self.created_agents[agent_id] = subagent

        return subagent

    def _determine_tools_for_task(self, task: Task) -> List[str]:
        """
        Determine appropriate tools for a task based on its type and requirements.

        Args:
            task: The task to analyze

        Returns:
            List of recommended tool names
        """
        # Start with tools explicitly required by the task
        required_tools = set(task.required_tools)

        # Add tools based on task type
        type_based_tools = set(
            self.tool_registry.get_tools_for_task_type(task.task_type)
        )

        # Combine required and type-based tools
        recommended_tools = required_tools | type_based_tools

        # Add execute_code as a default versatile tool if not already included
        if not recommended_tools:
            recommended_tools.add("execute_code")

        # Filter to only include available tools
        available_assignable = set(
            self.tool_registry.get_available_assignable_tools().keys()
        )

        final_tools = list(recommended_tools & available_assignable)

        # Ensure we have at least one tool (execute_code if available)
        if not final_tools and "execute_code" in available_assignable:
            final_tools = ["execute_code"]

        return final_tools

    def _get_tool_descriptions(self, tool_names: List[str]) -> Dict[str, str]:
        """
        Get descriptions for the specified tools.

        Args:
            tool_names: List of tool names

        Returns:
            Dictionary mapping tool names to descriptions
        """
        descriptions = {}
        for tool_name in tool_names:
            tool_info = self.tool_registry.get_tool(tool_name)
            if tool_info:
                descriptions[tool_name] = tool_info.description
            else:
                descriptions[tool_name] = "No description available"

        return descriptions

    def _generate_previous_context(self, previous_results: List[TaskResult]) -> str:
        """
        Generate context string from previous task results.

        Args:
            previous_results: List of completed task results

        Returns:
            Formatted context string
        """
        if not previous_results:
            return "No previous tasks have been completed."

        context_parts = []
        for i, result in enumerate(previous_results[-5:], 1):  # Last 5 results
            context_parts.append(f"""
Task {i}: {result.task_id}
Status: {result.status}
Summary: {result.output[:200]}{"..." if len(result.output) > 200 else ""}
Artifacts: {", ".join(result.artifacts) if result.artifacts else "None"}
""")

        return "\n".join(context_parts)

    def _generate_file_system_context(self) -> str:
        """
        Generate context string from current file system state.

        Returns:
            Formatted file system context string
        """
        try:
            files = self.vfs.list_files()
            if not files:
                return "The virtual file system is currently empty."

            # Limit to most recent or important files
            file_summaries = []
            for file_path in files[:10]:  # Limit to 10 files
                try:
                    content = self.vfs.read_file(file_path)
                    size = len(content)
                    preview = content[:100] + "..." if len(content) > 100 else content
                    file_summaries.append(f"- {file_path} ({size} bytes): {preview}")
                except Exception:
                    file_summaries.append(f"- {file_path} (unable to read)")

            return f"Current files in the system:\n" + "\n".join(file_summaries)

        except Exception as e:
            return f"Error accessing file system: {str(e)}"

    def get_agent(self, agent_id: str) -> Optional[SubAgent]:
        """
        Get a created agent by ID.

        Args:
            agent_id: The agent identifier

        Returns:
            SubAgent instance or None if not found
        """
        return self.created_agents.get(agent_id)

    def get_all_agents(self) -> Dict[str, SubAgent]:
        """
        Get all created agents.

        Returns:
            Dictionary of all created agents
        """
        return self.created_agents.copy()

    def update_agent_status(self, agent_id: str, status: str) -> bool:
        """
        Update the status of an agent.

        Args:
            agent_id: The agent identifier
            status: New status value

        Returns:
            True if updated successfully, False if agent not found
        """
        if agent_id in self.created_agents:
            self.created_agents[agent_id].status = status
            return True
        return False

    def cleanup_agent(self, agent_id: str) -> bool:
        """
        Clean up resources for a completed agent.

        Args:
            agent_id: The agent identifier

        Returns:
            True if cleaned up successfully, False if agent not found
        """
        if agent_id in self.created_agents:
            # Remove tool whitelist
            self.tool_registry.remove_agent_whitelist(agent_id)
            # Remove agent from tracking
            del self.created_agents[agent_id]
            return True
        return False

    def get_factory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the factory and created agents.

        Returns:
            Dictionary with factory statistics
        """
        agents_by_status = {}
        agents_by_task_type = {}

        for agent in self.created_agents.values():
            # Count by status
            status = agent.status
            agents_by_status[status] = agents_by_status.get(status, 0) + 1

            # Count by task type
            task_type = agent.task.task_type
            agents_by_task_type[task_type] = agents_by_task_type.get(task_type, 0) + 1

        return {
            "total_agents_created": len(self.created_agents),
            "agents_by_status": agents_by_status,
            "agents_by_task_type": agents_by_task_type,
            "active_agent_ids": list(self.created_agents.keys()),
        }

#!/usr/bin/env python3
"""
Demo script that shows the system components without requiring API keys.
This demonstrates the architecture and basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def demo_virtual_file_system():
    """Demonstrate virtual file system capabilities."""
    print("🗂️  Virtual File System Demo")
    print("-" * 30)

    from src.models.virtual_file_system import VirtualFileSystem

    vfs = VirtualFileSystem()

    # Create some files
    vfs.write_file("project/README.md", "# My Project\n\nThis is a sample project.")
    vfs.write_file("project/src/main.py", "def main():\n    print('Hello, World!')")
    vfs.write_file("project/config.json", '{"name": "demo", "version": "1.0.0"}')

    print(f"📁 Created {len(vfs.list_files())} files")

    # List files
    print("\n📋 File listing:")
    for file_path in vfs.list_files():
        size = len(vfs.read_file(file_path))
        print(f"  {file_path} ({size} bytes)")

    # Edit a file
    edits = [{"find": "Hello, World!", "replace": "Hello from LangGraph!"}]
    result = vfs.edit_file("project/src/main.py", edits)
    print(f"\n✏️  Edit successful - diff available")

    # Show updated content
    updated_content = vfs.read_file("project/src/main.py")
    print(f"📄 Updated content:\n{updated_content}")

    return vfs


def demo_tool_registry():
    """Demonstrate tool registry functionality."""
    print("\n🔧 Tool Registry Demo")
    print("-" * 30)

    from src.config import AgentConfig
    from src.tools.tool_registry import ToolRegistry
    from src.models.virtual_file_system import VirtualFileSystem

    config = AgentConfig()
    vfs = VirtualFileSystem()
    registry = ToolRegistry(config, vfs)

    # Show available tools
    shared_tools = registry.get_shared_tools()
    assignable_tools = registry.get_assignable_tools()

    print(f"📚 Shared tools ({len(shared_tools)}):")
    for tool_name in shared_tools.keys():
        print(f"  - {tool_name}")

    print(f"\n🎯 Assignable tools ({len(assignable_tools)}):")
    for tool_name in assignable_tools.keys():
        print(f"  - {tool_name}")

    # Demonstrate tool assignment
    research_tools = registry.get_tools_for_task_type("research")
    coding_tools = registry.get_tools_for_task_type("coding")

    print(f"\n🔍 Research task tools: {research_tools}")
    print(f"💻 Coding task tools: {coding_tools}")

    return registry


def demo_configuration():
    """Demonstrate configuration management."""
    print("\n⚙️  Configuration Demo")
    print("-" * 30)

    from src.config import AgentConfig

    # Default configuration
    config = AgentConfig()
    print(f"🔧 Default model: {config.llm_model}")
    print(f"🔄 Max iterations: {config.max_iterations}")
    print(f"👥 Max subagents: {config.max_subagents}")
    print(f"⏱️  Tool timeout: {config.tool_timeout}s")

    # Custom configuration
    custom_config = AgentConfig(
        llm_model="gpt-3.5-turbo", max_iterations=5, max_subagents=3
    )
    print(f"\n🎛️  Custom configuration:")
    print(f"   Model: {custom_config.llm_model}")
    print(f"   Iterations: {custom_config.max_iterations}")
    print(f"   Subagents: {custom_config.max_subagents}")

    return config


def demo_workflow_structure():
    """Demonstrate workflow structure without API calls."""
    print("\n🏗️  Workflow Structure Demo")
    print("-" * 30)

    from src.config import AgentConfig
    from src.workflow_simple import SimpleWorkflow

    config = AgentConfig(max_iterations=1)
    workflow = SimpleWorkflow(config)

    print("✅ Workflow components initialized:")
    print(f"   - Virtual File System: {type(workflow.vfs).__name__}")
    print(f"   - Tool Registry: {type(workflow.tool_registry).__name__}")
    print(f"   - Supervisor: {type(workflow.supervisor).__name__}")
    print(f"   - Subagent Factory: {type(workflow.subagent_factory).__name__}")

    # Show workflow capabilities
    print(f"\n📊 Workflow capabilities:")
    print(f"   - File operations: {len(workflow.list_files())} files tracked")
    print(
        f"   - Available tools: {len(workflow.tool_registry.get_shared_tools())} shared"
    )
    print(f"   - Configuration: {workflow.config.llm_model} model")

    return workflow


def demo_task_decomposition():
    """Demonstrate task decomposition logic (without API calls)."""
    print("\n📋 Task Decomposition Demo")
    print("-" * 30)

    from src.agents.supervisor import SupervisorAgent
    from src.agents.subagent_factory import SubAgentFactory
    from src.config import AgentConfig
    from src.tools.tool_registry import ToolRegistry
    from src.models.virtual_file_system import VirtualFileSystem

    config = AgentConfig()
    vfs = VirtualFileSystem()
    tool_registry = ToolRegistry(config, vfs)
    subagent_factory = SubAgentFactory(config, tool_registry, vfs)
    supervisor = SupervisorAgent(config, tool_registry, vfs, subagent_factory)

    # Example objectives and their expected task types
    objectives = [
        "Research AI trends and create a report",
        "Analyze code quality and suggest improvements",
        "Create a web scraping script for data collection",
        "Test a web application for security vulnerabilities",
    ]

    print("🎯 Example objectives and expected task patterns:")
    for i, objective in enumerate(objectives, 1):
        print(f"\n{i}. Objective: {objective}")

        # Simulate task type identification
        if "research" in objective.lower():
            task_types = ["research", "analysis", "documentation"]
        elif "code" in objective.lower() or "script" in objective.lower():
            task_types = ["analysis", "coding", "testing"]
        elif "test" in objective.lower():
            task_types = ["testing", "analysis", "documentation"]
        else:
            task_types = ["analysis", "planning", "execution"]

        print(f"   Expected task types: {', '.join(task_types)}")

        # Show tool assignment
        tools_needed = []
        for task_type in task_types:
            tools_needed.extend(tool_registry.get_tools_for_task_type(task_type))

        unique_tools = list(set(tools_needed))
        print(
            f"   Tools needed: {', '.join(unique_tools[:3])}{'...' if len(unique_tools) > 3 else ''}"
        )


def main():
    """Run all demonstrations."""
    print("🚀 LangGraph Supervisor Agent - Component Demo")
    print("=" * 60)
    print("This demo shows the system architecture without requiring API keys.\n")

    # Run demonstrations
    vfs = demo_virtual_file_system()
    registry = demo_tool_registry()
    config = demo_configuration()
    workflow = demo_workflow_structure()
    demo_task_decomposition()

    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Add your OpenAI API key to .env file")
    print("2. Run: python examples/run_simple_workflow.py")
    print("3. Or start LangGraph Studio: langgraph studio")
    print("\nFor more information, see README.md and docs/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LangGraph Studio demonstration script.

This script shows how to use the Supervisor Agent with LangGraph Studio
for development, debugging, and testing.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow import SupervisorWorkflow
from config import AgentConfig
from studio_config import StudioDebugger, StudioVisualizer, create_studio_config


def demo_basic_execution():
    """Demonstrate basic workflow execution."""
    print("🎯 Basic Execution Demo")
    print("=" * 50)

    # Create configuration
    config = AgentConfig()

    # Create workflow
    workflow = SupervisorWorkflow(config)

    # Simple objective for demonstration
    objective = "Create a simple Python script that prints 'Hello, World!' and save it to a file"

    print(f"Objective: {objective}")
    print("\nExecuting workflow...")

    try:
        result = workflow.run(objective)

        print("\n✅ Execution Complete!")
        print(f"Iterations: {result['iterations']}")
        print(f"Completed Tasks: {result['completed_tasks']}")
        print(f"Files Created: {len(result['file_system'])}")

        # Show created files
        if result["file_system"]:
            print("\n📁 Created Files:")
            for path, content in result["file_system"].items():
                print(f"  {path}: {len(content)} characters")

        print(f"\n📋 Final Result:\n{result['final_result']}")

    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return False

    return True


def demo_debugging_features():
    """Demonstrate debugging features."""
    print("\n🔧 Debugging Features Demo")
    print("=" * 50)

    # Create configuration with debug mode
    config = AgentConfig()
    workflow = SupervisorWorkflow(config)

    # Create debugger
    debugger = StudioDebugger(workflow)

    # Add some breakpoints
    debugger.add_breakpoint("supervisor")
    debugger.add_breakpoint("execute_task")

    print("Added breakpoints on 'supervisor' and 'execute_task' nodes")

    # Simulate some execution logging
    mock_state = {
        "user_objective": "Test objective",
        "todo_list": [
            {"id": "task_1", "description": "Test task", "status": "pending"}
        ],
        "completed_tasks": [],
        "iteration_count": 1,
        "file_system": {},
    }

    # Log execution
    debugger.log_execution("supervisor", mock_state, {"action": "plan_creation"})
    debugger.capture_state_snapshot(mock_state, "initial_state")

    # Show debugging info
    summary = debugger.get_execution_summary()
    print(f"\nExecution Summary:")
    print(f"  Total Executions: {summary['total_executions']}")
    print(f"  Nodes Executed: {summary['nodes_executed']}")
    print(f"  Active Breakpoints: {summary['active_breakpoints']}")
    print(f"  Snapshots: {summary['snapshot_count']}")


def demo_visualization():
    """Demonstrate visualization features."""
    print("\n📊 Visualization Demo")
    print("=" * 50)

    visualizer = StudioVisualizer()

    # Generate Mermaid diagram
    print("Generated Mermaid diagram for workflow visualization")

    # Show state inspector config
    inspector_config = visualizer.generate_state_inspector_config()
    print(f"\nState Inspector Panels: {len(inspector_config['inspector_panels'])}")
    for panel in inspector_config["inspector_panels"]:
        print(f"  - {panel['name']}: {panel['fields']}")

    # Show timeline config
    timeline_config = visualizer.generate_execution_timeline_config()
    print(f"\nTimeline Event Types: {len(timeline_config['event_types'])}")
    for event_type, config in timeline_config["event_types"].items():
        print(f"  - {event_type}: {config['color']}")


def demo_environment_configs():
    """Demonstrate different environment configurations."""
    print("\n🌍 Environment Configurations Demo")
    print("=" * 50)

    environments = {
        "development": {
            "description": "Full debugging enabled",
            "max_iterations": 5,
            "debug_mode": True,
        },
        "testing": {
            "description": "Controlled execution",
            "max_iterations": 3,
            "debug_mode": False,
        },
        "production": {
            "description": "Optimized performance",
            "max_iterations": 20,
            "debug_mode": False,
        },
    }

    for env_name, env_config in environments.items():
        print(f"\n{env_name.title()} Environment:")
        print(f"  Description: {env_config['description']}")
        print(f"  Max Iterations: {env_config['max_iterations']}")
        print(f"  Debug Mode: {env_config['debug_mode']}")


def demo_studio_config_generation():
    """Demonstrate studio configuration generation."""
    print("\n⚙️ Studio Configuration Demo")
    print("=" * 50)

    # Create configuration
    config = AgentConfig()

    # Generate studio config
    studio_config = create_studio_config(config)

    print("Generated comprehensive studio configuration:")
    print(f"  Graph Config: ✓")
    print(f"  Visualization: ✓")
    print(f"  Debugging: ✓")
    print(f"  Environments: {len(studio_config['environments'])}")

    # Show some details
    graph_metadata = studio_config["graph_config"]["metadata"]
    print(f"\nGraph Metadata:")
    print(f"  Name: {graph_metadata['name']}")
    print(f"  Description: {graph_metadata['description']}")
    print(f"  Version: {graph_metadata['version']}")
    print(f"  Nodes: {len(graph_metadata['nodes'])}")


def show_studio_instructions():
    """Show instructions for using LangGraph Studio."""
    print("\n📚 LangGraph Studio Instructions")
    print("=" * 50)

    instructions = """
To use this demo with LangGraph Studio:

1. Install LangGraph Studio:
   pip install langgraph-studio

2. Setup the environment:
   python scripts/setup_studio.py

3. Start LangGraph Studio:
   langgraph studio

4. Open your browser to: http://localhost:8000

5. Select 'supervisor_agent' from the graph dropdown

6. Try these objectives:
   - "Create a simple Python script and save it to a file"
   - "Research AI trends and write a summary"
   - "Analyze a text file and create a report"

7. Use debugging features:
   - Set breakpoints on nodes
   - Inspect state in real-time
   - View execution timeline
   - Monitor performance metrics

8. Switch between environments:
   - Development: Full debugging
   - Testing: Controlled execution  
   - Production: Optimized performance

For detailed documentation, see:
- docs/studio_usage.md
- docs/studio_debugging.md
"""

    print(instructions)


def main():
    """Main demonstration function."""
    print("🎯 LangGraph Studio Demonstration")
    print("=" * 60)

    # Check if we can run the demos
    try:
        config = AgentConfig()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("Please ensure your .env file is properly configured.")
        return 1

    # Run demonstrations
    demos = [
        ("Basic Execution", demo_basic_execution),
        ("Debugging Features", demo_debugging_features),
        ("Visualization", demo_visualization),
        ("Environment Configs", demo_environment_configs),
        ("Studio Config Generation", demo_studio_config_generation),
    ]

    for demo_name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")

    # Show instructions
    show_studio_instructions()

    print("\n🎉 Demo complete! Ready to use LangGraph Studio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

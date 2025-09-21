"""Example script demonstrating the Simple Supervisor Agent workflow."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow_simple import SimpleWorkflow
from src.config import AgentConfig


def run_simple_example():
    """Run a simple workflow example."""
    print("=== Simple Research Example ===")

    # Create configuration
    config = AgentConfig()

    # Create workflow
    workflow = SimpleWorkflow(config)

    # Run a simple objective
    objective = "Research the latest trends in artificial intelligence"
    print(f"Objective: {objective}")
    print("Running workflow...")

    try:
        result = workflow.run(objective)

        print("\n=== Results ===")
        print(f"Success: {result.success}")
        print(f"Completed in {result.iterations} iterations")
        print(f"Tasks completed: {result.completed_tasks}")
        print(f"Artifacts created: {len(result.artifacts)}")
        print("\nFinal Result:")
        print(result.final_result)

        if result.file_system:
            print(f"\nFiles created: {list(result.file_system.keys())}")

    except Exception as e:
        print(f"Error running workflow: {e}")


def run_web_testing_example():
    """Run a web application testing example."""
    print("\n=== Web Application Testing Example ===")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    objective = "Test my web application for bugs and usability issues"
    print(f"Objective: {objective}")
    print("Running workflow...")

    try:
        result = workflow.run(objective)

        print("\n=== Results ===")
        print(f"Success: {result.success}")
        print(f"Completed in {result.iterations} iterations")
        print(f"Tasks completed: {result.completed_tasks}")
        print("\nFinal Result:")
        print(result.final_result)

        # Show workflow statistics
        stats = workflow.get_workflow_stats()
        print(f"\nWorkflow Statistics:")
        print(f"- Total tasks: {stats.get('total_tasks', 0)}")
        print(f"- Completed tasks: {stats.get('completed_tasks', 0)}")
        print(f"- Pending tasks: {stats.get('pending_tasks', 0)}")
        print(f"- Failed tasks: {stats.get('failed_tasks', 0)}")

    except Exception as e:
        print(f"Error running workflow: {e}")


def run_code_analysis_example():
    """Run a code analysis example."""
    print("\n=== Code Analysis Example ===")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    objective = "Analyze this Python codebase and provide improvement recommendations"
    print(f"Objective: {objective}")
    print("Running workflow...")

    try:
        result = workflow.run(objective)

        print("\n=== Results ===")
        print(f"Success: {result.success}")
        print(f"Completed in {result.iterations} iterations")
        print(f"Tasks completed: {result.completed_tasks}")
        print("\nFinal Result:")
        print(result.final_result)

        # Show task details
        tasks = workflow.get_task_details()
        print(f"\nTask Breakdown ({len(tasks)} tasks):")
        for i, task in enumerate(tasks[:5], 1):  # Show first 5 tasks
            print(
                f"{i}. {task.get('description', 'No description')} - {task.get('status', 'unknown')}"
            )

    except Exception as e:
        print(f"Error running workflow: {e}")


def demonstrate_file_operations():
    """Demonstrate file operations in the workflow."""
    print("\n=== File Operations Example ===")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    objective = "Create a comprehensive project documentation"
    print(f"Objective: {objective}")
    print("Running workflow...")

    try:
        result = workflow.run(objective)

        print("\n=== Results ===")
        print(f"Success: {result.success}")
        print(f"Files created: {len(result.file_system)}")

        # List all files
        files = workflow.list_files()
        if files:
            print("\nFiles in virtual file system:")
            for file_path in files:
                content = workflow.get_file_contents(file_path)
                size = len(content) if content else 0
                print(f"- {file_path} ({size} bytes)")

                # Show preview of first file
                if file_path == files[0] and content:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"  Preview: {preview}")
        else:
            print("No files were created during execution.")

    except Exception as e:
        print(f"Error running workflow: {e}")


def demonstrate_workflow_features():
    """Demonstrate various workflow features."""
    print("\n=== Workflow Features Demo ===")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    # Test multiple objectives
    objectives = [
        "Create a simple test plan",
        "Generate a status report",
        "Analyze system performance",
    ]

    for i, objective in enumerate(objectives, 1):
        print(f"\n--- Running Objective {i}: {objective} ---")

        try:
            result = workflow.run(objective)
            print(f"✓ Completed in {result.iterations} iterations")
            print(f"✓ {result.completed_tasks} tasks completed")

            # Reset for next objective
            if i < len(objectives):
                workflow.reset()
                print("✓ Workflow reset for next objective")

        except Exception as e:
            print(f"✗ Error: {e}")


def main():
    """Main function to run all examples."""
    print("Simple Supervisor Agent Workflow Examples")
    print("=" * 50)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Using mock configuration.")
        os.environ["OPENAI_API_KEY"] = "test-key"

    # Disable tracing for examples
    os.environ["ENABLE_TRACING"] = "false"

    try:
        # Run examples
        run_simple_example()
        run_web_testing_example()
        run_code_analysis_example()
        demonstrate_file_operations()
        demonstrate_workflow_features()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

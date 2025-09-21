"""
Comprehensive end-to-end demonstration of the Supervisor Agent workflow.

This script demonstrates the complete workflow implementation including:
- Complex multi-step objectives
- Iteration management
- Termination conditions
- Artifact collection
- Error handling and recovery
- Performance metrics
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow_simple import SimpleWorkflow
from src.config import AgentConfig


def setup_environment():
    """Set up the environment for demonstration."""
    # Set up API keys (use mock values for demo)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Using mock configuration for demonstration.")
        os.environ["OPENAI_API_KEY"] = "demo-key"

    # Disable tracing for cleaner demo output
    os.environ["ENABLE_TRACING"] = "false"


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_workflow_result(result, workflow_type: str = "Simple"):
    """Print formatted workflow results."""
    print(f"\n📊 {workflow_type} Workflow Results:")
    print("-" * 40)

    if hasattr(result, "success"):  # SimpleWorkflow result
        print(f"✅ Success: {result.success}")
        print(f"🎯 Objective: {result.objective}")
        print(f"🔄 Iterations: {result.iterations}")
        print(f"✅ Tasks Completed: {result.completed_tasks}")
        print(
            f"📁 Artifacts Created: {len(result.artifacts) if result.artifacts else 0}"
        )
        print(f"📄 Files in VFS: {len(result.file_system)}")

        if result.error_message:
            print(f"❌ Error: {result.error_message}")

        print(f"\n📋 Final Result Summary:")
        print("-" * 30)
        # Show first 500 characters of final result
        summary = (
            result.final_result[:500] + "..."
            if len(result.final_result) > 500
            else result.final_result
        )
        print(summary)

    else:  # LangGraph workflow result
        print(f"🎯 Objective: {result.get('objective', 'Unknown')}")
        print(f"🔄 Iterations: {result.get('iterations', 0)}")
        print(f"✅ Tasks Completed: {result.get('completed_tasks', 0)}")
        print(f"📁 Artifacts: {len(result.get('artifacts', {}))}")
        print(f"📄 Files in VFS: {len(result.get('file_system', {}))}")

        print(f"\n📋 Final Result Summary:")
        print("-" * 30)
        final_result = result.get("final_result", "No result available")
        summary = (
            final_result[:500] + "..." if len(final_result) > 500 else final_result
        )
        print(summary)


def demo_research_workflow():
    """Demonstrate a comprehensive research workflow."""
    print_section_header("Research Workflow Demonstration")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    objective = "Research the latest developments in large language models and their applications in software development"

    print(f"🎯 Objective: {objective}")
    print("🚀 Starting research workflow...")

    start_time = time.time()

    try:
        result = workflow.run(objective)
        execution_time = time.time() - start_time

        print_workflow_result(result)
        print(f"⏱️  Total Execution Time: {execution_time:.2f} seconds")

        # Show workflow statistics
        stats = workflow.get_workflow_stats()
        print(f"\n📈 Workflow Statistics:")
        print(f"   - Total Tasks: {stats.get('total_tasks', 0)}")
        print(f"   - Completed: {stats.get('completed_tasks', 0)}")
        print(f"   - Pending: {stats.get('pending_tasks', 0)}")
        print(f"   - Failed: {stats.get('failed_tasks', 0)}")
        print(f"   - Subagents Created: {stats.get('subagents_created', 0)}")

        # Show created files
        files = workflow.list_files()
        if files:
            print(f"\n📁 Files Created ({len(files)}):")
            for file_path in files[:5]:  # Show first 5 files
                content = workflow.get_file_contents(file_path)
                size = len(content) if content else 0
                print(f"   - {file_path} ({size} bytes)")

        return result

    except Exception as e:
        print(f"❌ Error in research workflow: {e}")
        return None


def demo_web_testing_workflow():
    """Demonstrate a web application testing workflow."""
    print_section_header("Web Application Testing Workflow")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    objective = "Conduct comprehensive testing of my e-commerce web application including security, performance, and usability testing"

    print(f"🎯 Objective: {objective}")
    print("🚀 Starting web testing workflow...")

    start_time = time.time()

    try:
        result = workflow.run(objective)
        execution_time = time.time() - start_time

        print_workflow_result(result)
        print(f"⏱️  Total Execution Time: {execution_time:.2f} seconds")

        # Show task breakdown
        tasks = workflow.get_task_details()
        if tasks:
            print(f"\n📋 Task Breakdown ({len(tasks)} tasks):")
            for i, task in enumerate(tasks[:8], 1):  # Show first 8 tasks
                status_emoji = {
                    "completed": "✅",
                    "pending": "⏳",
                    "failed": "❌",
                    "in_progress": "🔄",
                }.get(task.get("status", "unknown"), "❓")
                print(
                    f"   {i}. {status_emoji} {task.get('description', 'No description')[:60]}..."
                )

        return result

    except Exception as e:
        print(f"❌ Error in web testing workflow: {e}")
        return None


def demo_code_analysis_workflow():
    """Demonstrate a code analysis and improvement workflow."""
    print_section_header("Code Analysis Workflow")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    objective = "Analyze this Python codebase for code quality, security vulnerabilities, performance issues, and provide detailed improvement recommendations"

    print(f"🎯 Objective: {objective}")
    print("🚀 Starting code analysis workflow...")

    # Pre-populate some sample code files for analysis
    workflow.vfs.write_file(
        "sample_code.py",
        """
def process_data(data):
    # Sample code with potential issues
    result = []
    for item in data:
        if item != None:  # Should use 'is not None'
            result.append(item * 2)
    return result

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_data(self, item):
        self.data.append(item)  # No validation
    
    def process_all(self):
        return [self.process_item(item) for item in self.data]
    
    def process_item(self, item):
        return item ** 2  # Potential overflow for large numbers
""",
    )

    workflow.vfs.write_file(
        "config.py",
        """
# Configuration with potential security issues
DATABASE_PASSWORD = "hardcoded_password"  # Security issue
API_KEY = "12345"  # Another security issue

DEBUG = True  # Should not be True in production
""",
    )

    start_time = time.time()

    try:
        result = workflow.run(objective)
        execution_time = time.time() - start_time

        print_workflow_result(result)
        print(f"⏱️  Total Execution Time: {execution_time:.2f} seconds")

        # Show file analysis results
        files = workflow.list_files()
        print(f"\n📁 Files Analyzed ({len(files)}):")
        for file_path in files:
            content = workflow.get_file_contents(file_path)
            if content:
                lines = len(content.split("\n"))
                print(f"   - {file_path} ({lines} lines)")

        return result

    except Exception as e:
        print(f"❌ Error in code analysis workflow: {e}")
        return None


def demo_max_iterations_handling():
    """Demonstrate workflow behavior with max iterations limit."""
    print_section_header("Max Iterations Handling")

    config = AgentConfig()
    config.max_iterations = 3  # Set low limit for demonstration
    workflow = SimpleWorkflow(config)

    objective = (
        "Create an infinitely complex project that would normally take many iterations"
    )

    print(f"🎯 Objective: {objective}")
    print(f"⚙️  Max Iterations Set To: {config.max_iterations}")
    print("🚀 Starting workflow with iteration limit...")

    start_time = time.time()

    try:
        result = workflow.run(objective)
        execution_time = time.time() - start_time

        print_workflow_result(result)
        print(f"⏱️  Total Execution Time: {execution_time:.2f} seconds")

        # Verify iteration limit was respected
        if result.iterations <= config.max_iterations:
            print(
                f"✅ Iteration limit properly enforced: {result.iterations}/{config.max_iterations}"
            )
        else:
            print(
                f"❌ Iteration limit exceeded: {result.iterations}/{config.max_iterations}"
            )

        return result

    except Exception as e:
        print(f"❌ Error in max iterations demo: {e}")
        return None


def demo_error_handling_and_recovery():
    """Demonstrate error handling and recovery mechanisms."""
    print_section_header("Error Handling and Recovery")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    objective = "Test error handling by attempting operations that may fail and recovering gracefully"

    print(f"🎯 Objective: {objective}")
    print("🚀 Starting error handling demonstration...")

    start_time = time.time()

    try:
        result = workflow.run(objective)
        execution_time = time.time() - start_time

        print_workflow_result(result)
        print(f"⏱️  Total Execution Time: {execution_time:.2f} seconds")

        # Check if workflow handled errors gracefully
        if result.success:
            print("✅ Workflow completed successfully despite potential errors")
        else:
            print(f"⚠️  Workflow completed with errors: {result.error_message}")

        return result

    except Exception as e:
        print(f"❌ Error in error handling demo: {e}")
        return None


def demo_file_system_operations():
    """Demonstrate file system operations and persistence."""
    print_section_header("File System Operations")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    # Pre-populate file system with various file types
    workflow.vfs.write_file(
        "project_requirements.md",
        """
# Project Requirements

## Overview
This project aims to create a comprehensive documentation system.

## Features
- Automated documentation generation
- Multi-format output support
- Version control integration
- Real-time collaboration

## Technical Requirements
- Python 3.9+
- FastAPI framework
- PostgreSQL database
- Redis for caching
""",
    )

    workflow.vfs.write_file(
        "api_spec.json",
        """{
    "openapi": "3.0.0",
    "info": {
        "title": "Documentation API",
        "version": "1.0.0"
    },
    "paths": {
        "/docs": {
            "get": {
                "summary": "Get documentation",
                "responses": {
                    "200": {
                        "description": "Success"
                    }
                }
            }
        }
    }
}""",
    )

    objective = "Create comprehensive project documentation based on existing requirements and API specifications"

    print(f"🎯 Objective: {objective}")
    print(f"📁 Pre-populated files: {len(workflow.list_files())}")
    print("🚀 Starting file system operations workflow...")

    start_time = time.time()

    try:
        result = workflow.run(objective)
        execution_time = time.time() - start_time

        print_workflow_result(result)
        print(f"⏱️  Total Execution Time: {execution_time:.2f} seconds")

        # Show detailed file system state
        final_files = workflow.list_files()
        print(f"\n📁 Final File System State ({len(final_files)} files):")

        for file_path in final_files:
            content = workflow.get_file_contents(file_path)
            if content:
                size = len(content)
                lines = len(content.split("\n"))
                file_type = file_path.split(".")[-1] if "." in file_path else "txt"
                print(f"   - {file_path} ({size} bytes, {lines} lines, {file_type})")

                # Show preview for markdown files
                if file_path.endswith(".md") and size > 100:
                    preview = content[:150] + "..." if len(content) > 150 else content
                    print(f"     Preview: {preview.replace(chr(10), ' ')}")

        return result

    except Exception as e:
        print(f"❌ Error in file system demo: {e}")
        return None


def demo_performance_metrics():
    """Demonstrate performance metrics collection."""
    print_section_header("Performance Metrics Collection")

    config = AgentConfig()
    workflow = SimpleWorkflow(config)

    objective = "Execute a series of tasks to demonstrate performance metrics collection and analysis"

    print(f"🎯 Objective: {objective}")
    print("🚀 Starting performance metrics demonstration...")

    start_time = time.time()

    try:
        result = workflow.run(objective)
        execution_time = time.time() - start_time

        print_workflow_result(result)
        print(f"⏱️  Total Execution Time: {execution_time:.2f} seconds")

        # Extract and display performance metrics
        artifacts = result.artifacts
        if artifacts and "consolidated_results" in str(artifacts):
            print(f"\n📊 Performance Metrics:")
            print(f"   - Workflow Execution Time: {execution_time:.2f}s")
            print(
                f"   - Tasks per Second: {result.completed_tasks / execution_time:.2f}"
            )
            print(
                f"   - Average Task Time: {execution_time / max(result.completed_tasks, 1):.2f}s"
            )
            print(
                f"   - Iterations per Second: {result.iterations / execution_time:.2f}"
            )

            # Memory usage (simulated)
            file_system_size = sum(
                len(content) for content in result.file_system.values()
            )
            print(f"   - File System Usage: {file_system_size} bytes")
            print(f"   - Artifacts Generated: {len(artifacts) if artifacts else 0}")

        return result

    except Exception as e:
        print(f"❌ Error in performance metrics demo: {e}")
        return None


def run_comprehensive_demo():
    """Run all demonstration scenarios."""
    print_section_header("Supervisor Agent Workflow - Comprehensive Demonstration")

    print("""
This demonstration showcases the complete end-to-end workflow implementation
including iteration management, termination conditions, artifact collection,
error handling, and performance metrics.

The following scenarios will be demonstrated:
1. Research Workflow - Complex multi-step research task
2. Web Testing Workflow - Comprehensive application testing
3. Code Analysis Workflow - Codebase analysis and improvement
4. Max Iterations Handling - Workflow termination conditions
5. Error Handling - Recovery mechanisms
6. File System Operations - File management and persistence
7. Performance Metrics - Metrics collection and analysis
""")

    input("Press Enter to start the demonstration...")

    # Track overall demo metrics
    demo_start_time = time.time()
    demo_results = {}

    # Run all demonstration scenarios
    scenarios = [
        ("Research", demo_research_workflow),
        ("Web Testing", demo_web_testing_workflow),
        ("Code Analysis", demo_code_analysis_workflow),
        ("Max Iterations", demo_max_iterations_handling),
        ("Error Handling", demo_error_handling_and_recovery),
        ("File System", demo_file_system_operations),
        ("Performance", demo_performance_metrics),
    ]

    for scenario_name, demo_func in scenarios:
        try:
            print(f"\n🎬 Starting {scenario_name} demonstration...")
            result = demo_func()
            demo_results[scenario_name] = {
                "success": result is not None
                and (not hasattr(result, "success") or result.success),
                "result": result,
            }

            if demo_results[scenario_name]["success"]:
                print(f"✅ {scenario_name} demonstration completed successfully")
            else:
                print(f"⚠️  {scenario_name} demonstration completed with issues")

        except Exception as e:
            print(f"❌ {scenario_name} demonstration failed: {e}")
            demo_results[scenario_name] = {"success": False, "error": str(e)}

        # Pause between demonstrations
        if scenario_name != scenarios[-1][0]:  # Not the last scenario
            input(f"\nPress Enter to continue to the next demonstration...")

    # Final summary
    demo_total_time = time.time() - demo_start_time

    print_section_header("Demonstration Summary")

    successful_demos = sum(1 for result in demo_results.values() if result["success"])
    total_demos = len(demo_results)

    print(f"📊 Overall Results:")
    print(f"   - Total Demonstrations: {total_demos}")
    print(f"   - Successful: {successful_demos}")
    print(f"   - Failed: {total_demos - successful_demos}")
    print(f"   - Success Rate: {successful_demos / total_demos * 100:.1f}%")
    print(f"   - Total Demo Time: {demo_total_time:.2f} seconds")

    print(f"\n📋 Individual Results:")
    for scenario_name, result in demo_results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"   - {scenario_name}: {status}")
        if not result["success"] and "error" in result:
            print(f"     Error: {result['error']}")

    print(f"\n🎉 Comprehensive demonstration completed!")
    print(f"   The Supervisor Agent workflow has been successfully demonstrated")
    print(f"   across {total_demos} different scenarios, showcasing:")
    print(f"   - Complete supervisor loop with iteration management")
    print(f"   - Final result consolidation and artifact collection")
    print(f"   - Workflow termination conditions and success detection")
    print(f"   - Integration of all components into cohesive system")
    print(f"   - End-to-end tests with complex multi-step objectives")


def main():
    """Main function to run the comprehensive demonstration."""
    setup_environment()

    try:
        run_comprehensive_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

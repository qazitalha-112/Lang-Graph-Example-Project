#!/usr/bin/env python3
"""Setup script for LangGraph Studio configuration."""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import AgentConfig, validate_environment
from studio_config import create_studio_config, export_studio_config


def setup_environment_file():
    """Create .env file from .env.example if it doesn't exist."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists() and env_example_path.exists():
        print("Creating .env file from .env.example...")
        with open(env_example_path, "r") as src, open(env_path, "w") as dst:
            content = src.read()
            dst.write(content)
        print("✅ .env file created. Please update it with your API keys.")
        return False
    elif not env_path.exists():
        print("❌ No .env file found and no .env.example to copy from.")
        return False

    return True


def validate_studio_requirements():
    """Validate that all requirements for LangGraph Studio are met."""
    print("Validating studio requirements...")

    # Check environment configuration
    if not validate_environment():
        print("❌ Environment validation failed. Please check your .env file.")
        return False

    # Check required packages
    try:
        import langgraph
        import langchain_openai

        print(f"✅ LangGraph version: {langgraph.__version__}")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        return False

    # Check API keys
    config = AgentConfig()
    if not config.openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False

    if config.enable_tracing and not config.langsmith_api_key:
        print("❌ LANGSMITH_API_KEY not found but tracing is enabled")
        return False

    print("✅ All requirements validated")
    return True


def create_studio_files():
    """Create necessary files for LangGraph Studio."""
    print("Creating studio configuration files...")

    # Create studio config
    config = AgentConfig()
    export_studio_config(config, "studio_config.json")

    # Create a simple run script for studio
    run_script = '''#!/usr/bin/env python3
"""Simple run script for LangGraph Studio."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workflow import SupervisorWorkflow
from config import AgentConfig

def main():
    """Main entry point for studio execution."""
    config = AgentConfig()
    workflow = SupervisorWorkflow(config)
    
    # Example objective for testing
    objective = "Research the latest trends in AI and write a summary report"
    
    print(f"Running workflow with objective: {objective}")
    result = workflow.run(objective)
    
    print("\\n=== EXECUTION COMPLETE ===")
    print(f"Final Result: {result['final_result']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Completed Tasks: {result['completed_tasks']}")
    
    return result

if __name__ == "__main__":
    main()
'''

    with open("run_studio.py", "w") as f:
        f.write(run_script)

    # Make it executable
    os.chmod("run_studio.py", 0o755)

    print("✅ Studio files created:")
    print("  - studio_config.json")
    print("  - run_studio.py")


def create_development_configs():
    """Create configuration files for different environments."""
    print("Creating environment-specific configurations...")

    configs = {
        "development": {
            "model": "gpt-4",
            "max_iterations": 5,
            "max_subagents": 3,
            "enable_tracing": True,
            "debug_mode": True,
            "tool_timeout": 60,
        },
        "testing": {
            "model": "gpt-3.5-turbo",
            "max_iterations": 3,
            "max_subagents": 2,
            "enable_tracing": False,
            "debug_mode": False,
            "tool_timeout": 30,
        },
        "production": {
            "model": "gpt-4",
            "max_iterations": 20,
            "max_subagents": 10,
            "enable_tracing": True,
            "debug_mode": False,
            "tool_timeout": 120,
        },
    }

    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    for env_name, config in configs.items():
        config_path = configs_dir / f"{env_name}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  - configs/{env_name}.json")

    print("✅ Environment configurations created")


def print_studio_instructions():
    """Print instructions for using LangGraph Studio."""
    instructions = """
🎯 LangGraph Studio Setup Complete!

📋 Next Steps:

1. Install LangGraph Studio (if not already installed):
   pip install langgraph-studio

2. Start LangGraph Studio:
   langgraph studio

3. Open your browser to the studio URL (usually http://localhost:8000)

4. Load the supervisor_agent graph from the dropdown

🔧 Configuration:

- Main config: langgraph.json
- Studio config: studio_config.json  
- Environment configs: configs/
- Run script: run_studio.py

🎮 Studio Features:

✅ Graph Visualization
  - Hierarchical layout with color-coded nodes
  - Real-time execution highlighting
  - Interactive node inspection

✅ State Inspector
  - Live state monitoring
  - Structured data views
  - File system browser

✅ Debugging Tools
  - Breakpoint support
  - Step-through execution
  - Execution timeline
  - State snapshots

✅ Multiple Environments
  - Development (debug enabled)
  - Testing (controlled execution)
  - Production (optimized settings)

🚀 Quick Test:

Run the example script to test your setup:
python run_studio.py

📚 For more information, see the documentation in docs/studio_usage.md
"""
    print(instructions)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup LangGraph Studio for Supervisor Agent"
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip environment validation"
    )
    parser.add_argument(
        "--config-only", action="store_true", help="Only create configuration files"
    )

    args = parser.parse_args()

    print("🎯 Setting up LangGraph Studio for Supervisor Agent...")
    print("=" * 60)

    # Setup environment file
    if not args.config_only:
        if not setup_environment_file():
            print(
                "\n❌ Setup incomplete. Please configure your .env file and run again."
            )
            return 1

    # Validate requirements
    if not args.skip_validation and not args.config_only:
        if not validate_studio_requirements():
            print(
                "\n❌ Setup incomplete. Please fix the validation errors and run again."
            )
            return 1

    # Create studio files
    create_studio_files()

    # Create environment configs
    create_development_configs()

    # Print instructions
    print_studio_instructions()

    return 0


if __name__ == "__main__":
    sys.exit(main())

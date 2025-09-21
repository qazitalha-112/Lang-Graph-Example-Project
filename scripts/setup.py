#!/usr/bin/env python3
"""
Simple setup script for the LangGraph Supervisor Agent project.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_step(message):
    """Print a step message."""
    print(f"🔧 {message}")


def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")


def check_python_version():
    """Check Python version."""
    print_step("Checking Python version...")
    version = sys.version_info
    if version >= (3, 9):
        print_success(f"Python {version.major}.{version.minor} - OK")
        return True
    else:
        print_error(
            f"Python {version.major}.{version.minor} detected. Python 3.9+ required."
        )
        return False


def install_dependencies():
    """Install project dependencies."""
    print_step("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print_success("Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to install dependencies")
        return False


def setup_environment():
    """Set up environment file."""
    print_step("Setting up environment file...")

    project_root = Path(__file__).parent.parent
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"

    if env_file.exists():
        print_success(".env file already exists")
        return True

    if env_example.exists():
        shutil.copy2(env_example, env_file)
        print_success("Created .env file from template")
        print("⚠️  Please edit .env file with your API keys")
        return True
    else:
        print_error(".env.example not found")
        return False


def create_directories():
    """Create necessary directories."""
    print_step("Creating directories...")

    project_root = Path(__file__).parent.parent
    directories = ["logs", "data", "outputs"]

    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)

    print_success("Directories created")
    return True


def test_imports():
    """Test basic imports."""
    print_step("Testing imports...")

    try:
        import langgraph
        import langchain
        import openai

        print_success("Core dependencies imported successfully")
        return True
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return False


def main():
    """Run setup."""
    print("🚀 LangGraph Supervisor Agent Setup")
    print("=" * 40)

    steps = [
        check_python_version,
        install_dependencies,
        setup_environment,
        create_directories,
        test_imports,
    ]

    for step in steps:
        if not step():
            print_error("Setup failed")
            sys.exit(1)

    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python examples/run_simple_workflow.py")
    print("3. Or start LangGraph Studio: langgraph studio")


if __name__ == "__main__":
    main()

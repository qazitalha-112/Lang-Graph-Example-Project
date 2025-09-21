# LangGraph Supervisor Agent

A hierarchical AI agent system built with LangGraph that decomposes complex
objectives into manageable tasks and executes them using specialized subagents.

## Overview

This project implements a supervisor architecture where:

- A **Supervisor Agent** receives user objectives and breaks them down into
  actionable tasks
- **Subagents** are created dynamically with specialized prompts and tool access
- A **Virtual File System** provides context persistence across agent executions
- **Tool Registry** manages selective tool assignment to subagents
- **LangGraph** orchestrates the workflow with proper state management

## Features

- 🎯 **Hierarchical Architecture**: Central supervisor coordinates specialized
  subagents
- 🔧 **Dynamic Tool Assignment**: Tools are selectively assigned based on task
  requirements
- 📁 **Virtual File System**: In-memory file operations for context engineering
- 🔄 **Error Handling**: Robust retry mechanisms and state recovery
- 📊 **LangSmith Integration**: Optional tracing and evaluation capabilities
- 🎨 **LangGraph Studio**: Visual workflow development and debugging

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd langgraph-supervisor-agent
   ```

2. **Run setup script**

   ```bash
   python scripts/setup.py
   ```

3. **Edit environment file with your API keys**

   ```bash
   # Edit .env with your actual API keys
   nano .env
   ```

4. **Test the system (without API keys)**

   ```bash
   python examples/demo_without_api.py
   ```

5. **Run a full example (requires OpenAI API key)**
   ```bash
   python examples/run_simple_workflow.py
   ```

### Basic Usage

```python
from src.workflow_simple import SimpleWorkflow
from src.config import AgentConfig

# Create configuration
config = AgentConfig()

# Initialize workflow
workflow = SimpleWorkflow(config)

# Run an objective
result = workflow.run("Research AI trends and create a summary report")

print(f"Success: {result.success}")
print(f"Files created: {len(result.file_system)}")
```

## Configuration

### Required Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
LANGSMITH_API_KEY=your_langsmith_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

### Configuration Options

```python
config = AgentConfig(
    llm_model="gpt-4",          # LLM model to use
    max_iterations=10,          # Maximum workflow iterations
    max_subagents=5,            # Maximum concurrent subagents
    tool_timeout=30,            # Tool execution timeout
    enable_tracing=False,       # LangSmith tracing
    debug_mode=False            # Debug logging
)
```

## LangGraph Studio

For visual development and debugging:

```bash
# Install LangGraph Studio
pip install langgraph-studio

# Start studio
langgraph studio

# Open browser to http://localhost:8000
```

## Examples

The `examples/` directory contains various usage scenarios:

- `run_simple_workflow.py` - Basic workflow execution
- `end_to_end_demo.py` - Comprehensive demonstration
- `studio_demo.py` - LangGraph Studio examples

## Architecture

```
User Objective
     ↓
Supervisor Agent (decomposes into tasks)
     ↓
Task Queue → Subagent Factory → Specialized Subagents
     ↓                              ↓
Tool Registry ← Tool Assignment ← Task Execution
     ↓                              ↓
Virtual File System ← Results ← Task Completion
     ↓
Final Output
```

## Available Tools

### Shared Tools (All Agents)

- `read_file` - Read file contents
- `write_file` - Write file with metadata
- `edit_file` - Edit file with find/replace

### Assignable Tools (Selective)

- `execute_code` - Run code in sandbox
- `search_internet` - Search using Tavily API
- `web_scrape` - Scrape content using Firecrawl API

## Development

### Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src
```

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/
```

## Project Structure

```
├── src/                    # Source code
│   ├── agents/            # Agent implementations
│   ├── models/            # Data models
│   ├── tools/             # Tool implementations
│   ├── error_handling/    # Error management
│   ├── tracing/           # LangSmith integration
│   ├── config.py          # Configuration
│   ├── workflow.py        # Main LangGraph workflow
│   └── workflow_simple.py # Simplified interface
├── examples/              # Usage examples
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── .env.example          # Environment template
├── langgraph.json        # LangGraph configuration
└── pyproject.toml        # Project configuration
```

## Troubleshooting

### Common Issues

1. **Missing API Key**

   ```bash
   # Check if API key is set
   echo $OPENAI_API_KEY
   ```

2. **Import Errors**

   ```bash
   # Reinstall dependencies
   pip install -e .
   ```

3. **Workflow Hangs**
   ```python
   # Use debug mode
   config = AgentConfig(debug_mode=True, max_iterations=5)
   ```

### Debug Mode

Enable detailed logging:

```python
config = AgentConfig(
    debug_mode=True,
    log_level="DEBUG"
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com/)
- Tracing by [LangSmith](https://smith.langchain.com/)
- Search by [Tavily](https://tavily.com/)
- Web scraping by [Firecrawl](https://firecrawl.dev/)

---

**Note**: This project was developed with AI assistance using Kiro IDE. See
[docs/AI_ASSISTANCE_SUMMARY.md](docs/AI_ASSISTANCE_SUMMARY.md) for details about
the development process.

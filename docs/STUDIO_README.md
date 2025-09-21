# LangGraph Studio Setup for Supervisor Agent

This document provides a quick start guide for setting up and using LangGraph
Studio with the Supervisor Agent system.

## Quick Start

### 1. Prerequisites

- Python 3.9+
- OpenAI API key
- LangSmith API key (optional, for tracing)
- Tavily API key (optional, for internet search)
- Firecrawl API key (optional, for web scraping)

### 2. Installation

```bash
# Install LangGraph Studio
pip install langgraph-studio

# Run the setup script
python scripts/setup_studio.py
```

### 3. Configuration

1. **Update your .env file** with your API keys:

   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

2. **Verify configuration**:
   ```bash
   python scripts/setup_studio.py --skip-validation
   ```

### 4. Start Studio

```bash
# Start LangGraph Studio
langgraph studio

# Open browser to http://localhost:8000
# Select 'supervisor_agent' from the dropdown
```

## Studio Features

### 🎯 Graph Visualization

- **Hierarchical layout** with color-coded nodes
- **Real-time execution** highlighting
- **Interactive node inspection**

### 📊 State Inspector

- **Live state monitoring** with organized panels
- **File system browser** for virtual files
- **Structured data views** for complex objects

### 🔧 Debugging Tools

- **Breakpoint support** on any node
- **Step-through execution** with state inspection
- **Execution timeline** with performance metrics
- **State snapshots** at key points

### 🌍 Multiple Environments

- **Development**: Full debugging enabled
- **Testing**: Controlled execution parameters
- **Production**: Optimized for performance

## Configuration Files

| File                       | Purpose                      |
| -------------------------- | ---------------------------- |
| `langgraph.json`           | Main LangGraph configuration |
| `studio_config.json`       | Enhanced studio settings     |
| `configs/development.json` | Development environment      |
| `configs/testing.json`     | Testing environment          |
| `configs/production.json`  | Production environment       |

## Example Objectives

Try these objectives in the studio:

### Simple Tasks

```
Create a Python script that prints "Hello, World!" and save it to a file
```

### Research Tasks

```
Research the latest trends in artificial intelligence and write a summary report
```

### Analysis Tasks

```
Analyze the structure of this codebase and create a documentation outline
```

### Complex Tasks

```
Test my web application for bugs by exploring the main features and documenting any issues found
```

## Debugging Workflow

### 1. Set Breakpoints

- Click on any node in the graph
- Select "Add Breakpoint" from context menu
- Execution will pause at the breakpoint

### 2. Inspect State

- Use the State Inspector panel
- Expand nested objects for details
- Monitor file system changes

### 3. Step Through Execution

- Use "Step" button to advance one node
- Use "Continue" to resume normal execution
- Use "Reset" to start over

### 4. Analyze Performance

- Check execution timeline for bottlenecks
- Monitor token usage in LangSmith
- Review tool performance metrics

## Environment Switching

### Via Studio Interface

1. Go to Configuration panel
2. Select environment from dropdown
3. Click "Apply Configuration"

### Via Command Line

```bash
# Development environment
langgraph studio --config configs/development.json

# Testing environment
langgraph studio --config configs/testing.json

# Production environment
langgraph studio --config configs/production.json
```

## Common Issues

### Studio Won't Start

```bash
# Check if port is in use
lsof -i :8000

# Try different port
langgraph studio --port 8001
```

### Graph Not Loading

1. Check `langgraph.json` syntax
2. Verify import paths
3. Check for missing dependencies
4. Review console errors

### Execution Hangs

1. Set breakpoints to isolate issue
2. Check for infinite loops in routing
3. Verify tool timeouts are reasonable
4. Monitor resource usage

### API Errors

1. Verify API keys in `.env` file
2. Check API key permissions and quotas
3. Test external service connectivity
4. Review error logs in artifacts

## Performance Tips

### Development

- Use `gpt-3.5-turbo` for faster iteration
- Set lower `max_iterations` for testing
- Enable debug mode for detailed logging

### Testing

- Disable tracing for faster execution
- Use minimal tool sets
- Set strict timeouts

### Production

- Use `gpt-4` for best results
- Enable LangSmith tracing
- Set appropriate resource limits
- Monitor performance metrics

## Advanced Features

### Custom Visualizations

Modify `src/studio_config.py` to customize:

- Node colors and icons
- State inspector panels
- Execution timeline events

### Custom Debug Hooks

Add custom debugging logic:

```python
from src.studio_config import StudioDebugger

debugger = StudioDebugger(workflow)
debugger.register_hook("task_failure", custom_handler)
```

### External Integrations

Connect with external monitoring:

- DataDog for metrics
- Slack for notifications
- Custom webhooks for events

## Documentation

- **Usage Guide**: `docs/studio_usage.md`
- **Debugging Guide**: `docs/studio_debugging.md`
- **API Reference**: See source code documentation

## Support

For issues and questions:

1. Check the troubleshooting section in `docs/studio_debugging.md`
2. Review execution logs and state snapshots
3. Use breakpoints to isolate problems
4. Enable debug mode for detailed information

## Demo Script

Run the demo to test your setup:

```bash
python examples/studio_demo.py
```

This will demonstrate all major features and verify your configuration is
working correctly.

---

🎉 **You're ready to use LangGraph Studio with the Supervisor Agent!**

Start with simple objectives and gradually work up to more complex tasks as you
become familiar with the debugging and visualization features.

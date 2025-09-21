"""Unit tests for assignable tools with mocked external services."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import subprocess
import tempfile
import os
from src.tools.assignable_tools import (
    CodeExecutionTool,
    InternetSearchTool,
    WebScrapeTool,
    AssignableTools,
    create_assignable_tools,
)
from src.config import AgentConfig


class TestCodeExecutionTool:
    """Test cases for CodeExecutionTool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=AgentConfig)
        self.config.tool_timeout = 30
        self.tool = CodeExecutionTool(self.config)

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_execute_code_success(self, mock_tempfile, mock_subprocess):
        """Test successful code execution."""
        # Setup mocks
        mock_file = Mock()
        mock_file.name = "/tmp/test.py"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello, World!\n"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test
        result = self.tool.execute_code("print('Hello, World!')")

        # Assert
        assert result["status"] == "success"
        assert result["stdout"] == "Hello, World!\n"
        assert result["stderr"] == ""
        assert result["return_code"] == 0

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_execute_code_error(self, mock_tempfile, mock_subprocess):
        """Test code execution with error."""
        # Setup mocks
        mock_file = Mock()
        mock_file.name = "/tmp/test.py"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "SyntaxError: invalid syntax\n"
        mock_subprocess.return_value = mock_result

        # Test
        result = self.tool.execute_code("invalid python code")

        # Assert
        assert result["status"] == "error"
        assert result["stdout"] == ""
        assert "SyntaxError" in result["stderr"]
        assert result["return_code"] == 1

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_execute_code_timeout(self, mock_tempfile, mock_subprocess):
        """Test code execution timeout."""
        # Setup mocks
        mock_file = Mock()
        mock_file.name = "/tmp/test.py"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        mock_subprocess.side_effect = subprocess.TimeoutExpired("python", 30)

        # Test
        result = self.tool.execute_code("import time; time.sleep(60)")

        # Assert
        assert result["status"] == "timeout"
        assert "timed out" in result["stderr"]
        assert result["return_code"] == -1

    def test_execute_code_unsupported_language(self):
        """Test execution with unsupported language."""
        with pytest.raises(ValueError, match="Unsupported language"):
            self.tool.execute_code("console.log('test')", "javascript")

    def test_execute_code_empty_code(self):
        """Test execution with empty code."""
        with pytest.raises(ValueError, match="Code cannot be empty"):
            self.tool.execute_code("")

        with pytest.raises(ValueError, match="Code cannot be empty"):
            self.tool.execute_code("   ")


class TestInternetSearchTool:
    """Test cases for InternetSearchTool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=AgentConfig)
        self.config.tavily_api_key = "test_api_key"
        self.config.tool_timeout = 30
        self.tool = InternetSearchTool(self.config)

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        config = Mock(spec=AgentConfig)
        config.tavily_api_key = None
        config.tool_timeout = 30

        with pytest.raises(ValueError, match="Tavily API key is required"):
            InternetSearchTool(config)

    @patch("requests.post")
    def test_search_internet_success(self, mock_post):
        """Test successful internet search."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "Test content 1",
                    "score": 0.9,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "Test content 2",
                    "score": 0.8,
                },
            ]
        }
        mock_post.return_value = mock_response

        # Test
        results = self.tool.search_internet("test query")

        # Assert
        assert len(results) == 2
        assert results[0]["title"] == "Test Result 1"
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["content"] == "Test content 1"
        assert results[0]["score"] == 0.9

    @patch("requests.post")
    def test_search_internet_api_error(self, mock_post):
        """Test internet search with API error."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        # Test
        with pytest.raises(RuntimeError, match="Invalid Tavily API key"):
            self.tool.search_internet("test query")

    @patch("requests.post")
    def test_search_internet_rate_limit(self, mock_post):
        """Test internet search with rate limit."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_post.return_value = mock_response

        # Test
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            self.tool.search_internet("test query")

    @patch("requests.post")
    def test_search_internet_timeout(self, mock_post):
        """Test internet search timeout."""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout()

        # Test
        with pytest.raises(RuntimeError, match="timed out"):
            self.tool.search_internet("test query")

    def test_search_internet_empty_query(self):
        """Test search with empty query."""
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            self.tool.search_internet("")

        with pytest.raises(ValueError, match="Search query cannot be empty"):
            self.tool.search_internet("   ")

    def test_search_internet_invalid_max_results(self):
        """Test search with invalid max_results."""
        with pytest.raises(ValueError, match="max_results must be between 1 and 20"):
            self.tool.search_internet("test", max_results=0)

        with pytest.raises(ValueError, match="max_results must be between 1 and 20"):
            self.tool.search_internet("test", max_results=25)


class TestWebScrapeTool:
    """Test cases for WebScrapeTool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=AgentConfig)
        self.config.firecrawl_api_key = "test_api_key"
        self.config.tool_timeout = 30
        self.tool = WebScrapeTool(self.config)

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        config = Mock(spec=AgentConfig)
        config.firecrawl_api_key = None
        config.tool_timeout = 30

        with pytest.raises(ValueError, match="Firecrawl API key is required"):
            WebScrapeTool(config)

    @patch("requests.post")
    def test_web_scrape_success(self, mock_post):
        """Test successful web scraping."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "# Test Page\n\nThis is test content.",
                "html": "<h1>Test Page</h1><p>This is test content.</p>",
                "metadata": {"title": "Test Page", "description": "A test page"},
            },
        }
        mock_post.return_value = mock_response

        # Test
        result = self.tool.web_scrape("https://example.com")

        # Assert
        assert result["status"] == "success"
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"
        assert "This is test content" in result["content"]
        assert result["selector_used"] is None

    @patch("requests.post")
    def test_web_scrape_with_selector(self, mock_post):
        """Test web scraping with CSS selector."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "Selected content",
                "html": "<div>Selected content</div>",
                "metadata": {"title": "Test Page"},
            },
        }
        mock_post.return_value = mock_response

        # Test
        result = self.tool.web_scrape("https://example.com", selector=".content")

        # Assert
        assert result["status"] == "success"
        assert result["selector_used"] == ".content"

    @patch("requests.post")
    def test_web_scrape_api_error(self, mock_post):
        """Test web scraping with API error."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        # Test
        with pytest.raises(RuntimeError, match="Invalid Firecrawl API key"):
            self.tool.web_scrape("https://example.com")

    @patch("requests.post")
    def test_web_scrape_scraping_failed(self, mock_post):
        """Test web scraping when scraping fails."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "error": "Failed to scrape the page",
        }
        mock_post.return_value = mock_response

        # Test
        with pytest.raises(RuntimeError, match="Firecrawl scraping failed"):
            self.tool.web_scrape("https://example.com")

    @patch("requests.post")
    def test_web_scrape_timeout(self, mock_post):
        """Test web scraping timeout."""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout()

        # Test
        with pytest.raises(RuntimeError, match="timed out"):
            self.tool.web_scrape("https://example.com")

    def test_web_scrape_empty_url(self):
        """Test scraping with empty URL."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            self.tool.web_scrape("")

        with pytest.raises(ValueError, match="URL cannot be empty"):
            self.tool.web_scrape("   ")

    def test_web_scrape_invalid_url(self):
        """Test scraping with invalid URL."""
        with pytest.raises(ValueError, match="URL must start with http"):
            self.tool.web_scrape("ftp://example.com")

        with pytest.raises(ValueError, match="URL must start with http"):
            self.tool.web_scrape("example.com")


class TestAssignableTools:
    """Test cases for AssignableTools container."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=AgentConfig)
        self.config.tool_timeout = 30
        self.config.tavily_api_key = "tavily_key"
        self.config.firecrawl_api_key = "firecrawl_key"

    @patch("src.tools.assignable_tools.WebScrapeTool")
    @patch("src.tools.assignable_tools.InternetSearchTool")
    @patch("src.tools.assignable_tools.CodeExecutionTool")
    def test_init_all_tools_available(self, mock_code, mock_search, mock_scrape):
        """Test initialization when all tools are available."""
        # Test
        tools = AssignableTools(self.config)

        # Assert
        assert tools.code_tool is not None
        assert tools.search_tool is not None
        assert tools.scrape_tool is not None

    def test_init_missing_api_keys(self):
        """Test initialization with missing API keys."""
        config = Mock(spec=AgentConfig)
        config.tool_timeout = 30
        config.tavily_api_key = None
        config.firecrawl_api_key = None

        # Test
        tools = AssignableTools(config)

        # Assert
        assert tools.code_tool is not None
        assert tools.search_tool is None
        assert tools.scrape_tool is None

    @patch("src.tools.assignable_tools.WebScrapeTool")
    @patch("src.tools.assignable_tools.InternetSearchTool")
    @patch("src.tools.assignable_tools.CodeExecutionTool")
    def test_get_available_tools_all(self, mock_code, mock_search, mock_scrape):
        """Test getting available tools when all are configured."""
        tools = AssignableTools(self.config)
        available = tools.get_available_tools()

        assert "execute_code" in available
        assert "search_internet" in available
        assert "web_scrape" in available

    def test_get_available_tools_code_only(self):
        """Test getting available tools when only code execution is available."""
        config = Mock(spec=AgentConfig)
        config.tool_timeout = 30
        config.tavily_api_key = None
        config.firecrawl_api_key = None

        tools = AssignableTools(config)
        available = tools.get_available_tools()

        assert "execute_code" in available
        assert "search_internet" not in available
        assert "web_scrape" not in available

    def test_search_internet_not_available(self):
        """Test search_internet when tool is not available."""
        config = Mock(spec=AgentConfig)
        config.tool_timeout = 30
        config.tavily_api_key = None
        config.firecrawl_api_key = None

        tools = AssignableTools(config)

        with pytest.raises(RuntimeError, match="Internet search tool not available"):
            tools.search_internet("test query")

    def test_web_scrape_not_available(self):
        """Test web_scrape when tool is not available."""
        config = Mock(spec=AgentConfig)
        config.tool_timeout = 30
        config.tavily_api_key = None
        config.firecrawl_api_key = None

        tools = AssignableTools(config)

        with pytest.raises(RuntimeError, match="Web scrape tool not available"):
            tools.web_scrape("https://example.com")


class TestCreateAssignableTools:
    """Test cases for create_assignable_tools function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=AgentConfig)
        self.config.tool_timeout = 30

    @patch("src.tools.assignable_tools.AssignableTools")
    def test_create_assignable_tools_all_available(self, mock_assignable_tools):
        """Test creating tool functions when all tools are available."""
        # Setup mock
        mock_tools_instance = Mock()
        mock_tools_instance.search_tool = Mock()  # Available
        mock_tools_instance.scrape_tool = Mock()  # Available
        mock_assignable_tools.return_value = mock_tools_instance

        # Test
        tool_functions = create_assignable_tools(self.config)

        # Assert
        assert "execute_code" in tool_functions
        assert "search_internet" in tool_functions
        assert "web_scrape" in tool_functions

    @patch("src.tools.assignable_tools.AssignableTools")
    def test_create_assignable_tools_code_only(self, mock_assignable_tools):
        """Test creating tool functions when only code execution is available."""
        # Setup mock
        mock_tools_instance = Mock()
        mock_tools_instance.search_tool = None  # Not available
        mock_tools_instance.scrape_tool = None  # Not available
        mock_assignable_tools.return_value = mock_tools_instance

        # Test
        tool_functions = create_assignable_tools(self.config)

        # Assert
        assert "execute_code" in tool_functions
        assert "search_internet" not in tool_functions
        assert "web_scrape" not in tool_functions

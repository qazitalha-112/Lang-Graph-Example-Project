"""Assignable tools for subagents with external API integrations."""

import subprocess
import sys
import tempfile
import os
import requests
from typing import Dict, Any, List, Optional
from ..config import AgentConfig
from ..error_handling import ErrorHandler, create_retry_manager
from ..error_handling.exceptions import ToolExecutionError, NetworkError, TimeoutError


class CodeExecutionTool:
    """Tool for executing Python code in CodeAct style."""

    def __init__(
        self, config: AgentConfig, error_handler: Optional[ErrorHandler] = None
    ):
        """
        Initialize code execution tool.

        Args:
            config: Agent configuration containing timeout settings
            error_handler: Optional error handler for error management
        """
        self.config = config
        self.timeout = config.tool_timeout
        self.error_handler = error_handler or ErrorHandler(config)
        self.retry_manager = create_retry_manager(
            max_attempts=config.max_retry_attempts,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )

    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code and return results with error handling and recovery.

        Args:
            code: Code to execute
            language: Programming language (currently only supports "python")

        Returns:
            Dictionary with execution results

        Raises:
            ToolExecutionError: If code execution fails after retries
            ValueError: If language is not supported or code is invalid
        """
        if language.lower() != "python":
            raise ValueError(
                f"Unsupported language: {language}. Only 'python' is supported."
            )

        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        def _execute_code_impl():
            temp_file = None
            try:
                # Create temporary file for code execution
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as f:
                    f.write(code)
                    temp_file = f.name

                # Execute the code with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=os.getcwd(),
                )

                return {
                    "status": "success" if result.returncode == 0 else "error",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "execution_time": "completed within timeout",
                }

            except subprocess.TimeoutExpired as e:
                raise TimeoutError("execute_code", self.timeout, original_error=e)
            except Exception as e:
                raise ToolExecutionError(
                    "execute_code", f"Failed to execute code: {e}", original_error=e
                )
            finally:
                # Clean up temporary file
                if temp_file:
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass

        try:
            # Use retry manager for execution
            return self.retry_manager.execute_with_retry(
                _execute_code_impl, "execute_code"
            )
        except Exception as e:
            # Handle error through error handler
            context = {"code_length": len(code), "language": language}
            return self.error_handler.handle_tool_error("execute_code", e, context)


class InternetSearchTool:
    """Tool for searching the internet using Tavily API."""

    def __init__(
        self, config: AgentConfig, error_handler: Optional[ErrorHandler] = None
    ):
        """
        Initialize internet search tool.

        Args:
            config: Agent configuration containing API keys and timeout
            error_handler: Optional error handler for error management

        Raises:
            ValueError: If Tavily API key is not configured
        """
        self.config = config
        self.api_key = config.tavily_api_key
        self.timeout = config.tool_timeout
        self.base_url = "https://api.tavily.com"
        self.error_handler = error_handler or ErrorHandler(config)
        self.retry_manager = create_retry_manager(
            max_attempts=config.max_retry_attempts,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )

        if not self.api_key:
            raise ValueError("Tavily API key is required for internet search")

    def search_internet(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the internet for information with error handling and retry logic.

        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 5)

        Returns:
            List of search results with title, url, and content

        Raises:
            ValueError: If query is invalid or max_results is out of range
            ToolExecutionError: If API request fails after retries
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if max_results <= 0 or max_results > 20:
            raise ValueError("max_results must be between 1 and 20")

        def _search_impl():
            # Prepare API request
            headers = {"Content-Type": "application/json"}

            payload = {
                "api_key": self.api_key,
                "query": query.strip(),
                "search_depth": "basic",
                "include_answer": True,
                "include_images": False,
                "include_raw_content": False,
                "max_results": max_results,
            }

            # Make API request with timeout
            response = requests.post(
                f"{self.base_url}/search",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                # Process search results
                for result in data.get("results", []):
                    results.append(
                        {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "content": result.get("content", ""),
                            "score": result.get("score", 0.0),
                        }
                    )

                return results

            elif response.status_code == 401:
                raise NetworkError(
                    "search_internet", "Invalid Tavily API key", status_code=401
                )
            elif response.status_code == 429:
                raise NetworkError(
                    "search_internet", "Tavily API rate limit exceeded", status_code=429
                )
            else:
                raise NetworkError(
                    "search_internet",
                    f"Tavily API error: {response.status_code} - {response.text}",
                    status_code=response.status_code,
                )

        def _fallback_search():
            """Fallback to return empty results with warning."""
            return [
                {
                    "title": "Search Unavailable",
                    "url": "",
                    "content": f"Internet search for '{query}' is currently unavailable. Please try again later.",
                    "score": 0.0,
                }
            ]

        try:
            # Use retry manager with circuit breaker
            return self.retry_manager.execute_with_retry(
                _search_impl, "search_internet"
            )
        except Exception as e:
            # Handle error through error handler with fallback
            context = {"query": query, "max_results": max_results}
            result = self.error_handler.handle_tool_error(
                "search_internet", e, context, fallback_func=_fallback_search
            )

            # Return the result if recovery was successful
            if result.get("recovery_status") in [
                "retry_successful",
                "fallback_successful",
            ]:
                return result.get("result", _fallback_search())
            else:
                # Re-raise as ToolExecutionError if no recovery possible
                raise ToolExecutionError(
                    "search_internet", str(e), original_error=e, context=context
                )


class WebScrapeTool:
    """Tool for scraping web content using Firecrawl API."""

    def __init__(
        self, config: AgentConfig, error_handler: Optional[ErrorHandler] = None
    ):
        """
        Initialize web scrape tool.

        Args:
            config: Agent configuration containing API keys and timeout
            error_handler: Optional error handler for error management

        Raises:
            ValueError: If Firecrawl API key is not configured
        """
        self.config = config
        self.api_key = config.firecrawl_api_key
        self.timeout = config.tool_timeout
        self.base_url = "https://api.firecrawl.dev"
        self.error_handler = error_handler or ErrorHandler(config)
        self.retry_manager = create_retry_manager(
            max_attempts=config.max_retry_attempts,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )

        if not self.api_key:
            raise ValueError("Firecrawl API key is required for web scraping")

    def web_scrape(self, url: str, selector: Optional[str] = None) -> Dict[str, Any]:
        """
        Scrape content from a web page with error handling and retry logic.

        Args:
            url: URL to scrape
            selector: Optional CSS selector to extract specific content

        Returns:
            Dictionary with scraped content and metadata

        Raises:
            ValueError: If URL is invalid
            ToolExecutionError: If scraping fails after retries
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        # Basic URL validation
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")

        def _scrape_impl():
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "url": url.strip(),
                "formats": ["markdown", "html"],
                "onlyMainContent": True,
            }

            # Add selector if provided
            if selector and selector.strip():
                payload["extract"] = {"selector": selector.strip()}

            # Make API request with timeout
            response = requests.post(
                f"{self.base_url}/v1/scrape",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("success"):
                    content_data = data.get("data", {})
                    return {
                        "status": "success",
                        "url": url,
                        "title": content_data.get("metadata", {}).get("title", ""),
                        "content": content_data.get("markdown", ""),
                        "html": content_data.get("html", ""),
                        "metadata": content_data.get("metadata", {}),
                        "selector_used": selector,
                    }
                else:
                    error_msg = data.get("error", "Unknown error")
                    raise NetworkError(
                        "web_scrape", f"Firecrawl scraping failed: {error_msg}"
                    )

            elif response.status_code == 401:
                raise NetworkError(
                    "web_scrape", "Invalid Firecrawl API key", status_code=401
                )
            elif response.status_code == 429:
                raise NetworkError(
                    "web_scrape", "Firecrawl API rate limit exceeded", status_code=429
                )
            else:
                raise NetworkError(
                    "web_scrape",
                    f"Firecrawl API error: {response.status_code} - {response.text}",
                    status_code=response.status_code,
                )

        def _fallback_scrape():
            """Fallback to return basic page info."""
            return {
                "status": "fallback",
                "url": url,
                "title": "Content Unavailable",
                "content": f"Unable to scrape content from {url}. The page may be temporarily unavailable.",
                "html": "",
                "metadata": {"error": "scraping_unavailable"},
                "selector_used": selector,
            }

        try:
            # Use retry manager with circuit breaker
            return self.retry_manager.execute_with_retry(_scrape_impl, "web_scrape")
        except Exception as e:
            # Handle error through error handler with fallback
            context = {"url": url, "selector": selector}
            result = self.error_handler.handle_tool_error(
                "web_scrape", e, context, fallback_func=_fallback_scrape
            )

            # Return the result if recovery was successful
            if result.get("recovery_status") in [
                "retry_successful",
                "fallback_successful",
            ]:
                return result.get("result", _fallback_scrape())
            else:
                # Re-raise as ToolExecutionError if no recovery possible
                raise ToolExecutionError(
                    "web_scrape", str(e), original_error=e, context=context
                )


class AssignableTools:
    """Container for all assignable tools that can be given to subagents."""

    def __init__(
        self, config: AgentConfig, error_handler: Optional[ErrorHandler] = None
    ):
        """
        Initialize all assignable tools.

        Args:
            config: Agent configuration
            error_handler: Optional error handler for error management
        """
        self.config = config
        self.error_handler = error_handler or ErrorHandler(config)
        self.code_tool = CodeExecutionTool(config, self.error_handler)

        # Initialize optional tools based on API key availability
        self.search_tool = None
        self.scrape_tool = None

        try:
            if config.tavily_api_key:
                self.search_tool = InternetSearchTool(config, self.error_handler)
        except ValueError:
            pass  # Tavily not configured

        try:
            if config.firecrawl_api_key:
                self.scrape_tool = WebScrapeTool(config, self.error_handler)
        except ValueError:
            pass  # Firecrawl not configured

    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.

        Returns:
            List of available tool names
        """
        tools = ["execute_code"]

        if self.search_tool:
            tools.append("search_internet")

        if self.scrape_tool:
            tools.append("web_scrape")

        return tools

    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code using the code execution tool."""
        return self.code_tool.execute_code(code, language)

    def search_internet(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search internet using the search tool."""
        if not self.search_tool:
            raise RuntimeError(
                "Internet search tool not available (Tavily API key not configured)"
            )
        return self.search_tool.search_internet(query, max_results)

    def web_scrape(self, url: str, selector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape web content using the scrape tool."""
        if not self.scrape_tool:
            raise RuntimeError(
                "Web scrape tool not available (Firecrawl API key not configured)"
            )
        return self.scrape_tool.web_scrape(url, selector)


def create_assignable_tools(
    config: AgentConfig, error_handler: Optional[ErrorHandler] = None
) -> Dict[str, callable]:
    """
    Create assignable tool functions that can be used with LangGraph/LangChain.

    Args:
        config: Agent configuration
        error_handler: Optional error handler for error management

    Returns:
        Dictionary of available tool functions
    """
    tools_container = AssignableTools(config, error_handler)

    tool_functions = {"execute_code": tools_container.execute_code}

    # Add optional tools if available
    if tools_container.search_tool:
        tool_functions["search_internet"] = tools_container.search_internet

    if tools_container.scrape_tool:
        tool_functions["web_scrape"] = tools_container.web_scrape

    return tool_functions

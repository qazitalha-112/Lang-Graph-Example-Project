"""File operation tools that integrate with VirtualFileSystem."""

from typing import Dict, Any, Union, List, Optional
from ..models.virtual_file_system import VirtualFileSystem
from ..error_handling import ErrorHandler
from ..error_handling.exceptions import ToolExecutionError


class FileTools:
    """
    File operation tools that provide a consistent interface for file operations
    using the VirtualFileSystem backend.
    """

    def __init__(
        self, vfs: VirtualFileSystem, error_handler: Optional[ErrorHandler] = None
    ):
        """
        Initialize file tools with a VirtualFileSystem instance.

        Args:
            vfs: VirtualFileSystem instance to use for file operations
            error_handler: Optional error handler for error management
        """
        self.vfs = vfs
        self.error_handler = error_handler

    def read_file(self, path: str) -> str:
        """
        Read content from a file with error handling.

        Args:
            path: Path to the file to read

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
            ToolExecutionError: If unexpected error occurs
        """
        try:
            return self.vfs.read_file(path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cannot read file: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid file path: {e}")
        except Exception as e:
            if self.error_handler:
                context = {"path": path, "operation": "read"}
                self.error_handler.handle_tool_error("read_file", e, context)
            raise ToolExecutionError(
                "read_file",
                f"Unexpected error reading file '{path}': {e}",
                original_error=e,
            )

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file with error handling.

        Args:
            path: Path to the file to write
            content: Content to write to the file

        Returns:
            Dictionary with path and bytes_written

        Raises:
            ValueError: If path is invalid or content is invalid
            ToolExecutionError: If unexpected error occurs
        """
        try:
            if not isinstance(content, str):
                content = str(content)

            result = self.vfs.write_file(path, content)
            return {
                "path": result["path"],
                "bytes_written": result["bytes_written"],
                "status": "success",
            }
        except ValueError as e:
            raise ValueError(f"Cannot write file: {e}")
        except Exception as e:
            if self.error_handler:
                context = {
                    "path": path,
                    "content_length": len(str(content)),
                    "operation": "write",
                }
                self.error_handler.handle_tool_error("write_file", e, context)
            raise ToolExecutionError(
                "write_file",
                f"Unexpected error writing file '{path}': {e}",
                original_error=e,
            )

    def edit_file(
        self, path: str, edits: Union[List[Dict[str, str]], str]
    ) -> Dict[str, Any]:
        """
        Edit a file using find/replace operations or whole-file replacement with error handling.

        Args:
            path: Path to the file to edit
            edits: Either a list of {"find": str, "replace": str} dicts for
                  find/replace operations, or a string for whole-file replacement

        Returns:
            Dictionary with path, diff, and status

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid or edits format is wrong
            ToolExecutionError: If unexpected error occurs
        """
        try:
            # Validate edits format
            if isinstance(edits, list):
                for i, edit in enumerate(edits):
                    if not isinstance(edit, dict):
                        raise ValueError(f"Edit {i} must be a dictionary")
                    if "find" not in edit or "replace" not in edit:
                        raise ValueError(
                            f"Edit {i} must have 'find' and 'replace' keys"
                        )
                    if not isinstance(edit["find"], str) or not isinstance(
                        edit["replace"], str
                    ):
                        raise ValueError(
                            f"Edit {i} 'find' and 'replace' values must be strings"
                        )
            elif not isinstance(edits, str):
                raise ValueError(
                    "Edits must be either a string or list of find/replace dictionaries"
                )

            result = self.vfs.edit_file(path, edits)
            return {"path": result["path"], "diff": result["diff"], "status": "success"}
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cannot edit file: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid edit operation: {e}")
        except Exception as e:
            if self.error_handler:
                context = {
                    "path": path,
                    "edits_type": type(edits).__name__,
                    "operation": "edit",
                }
                self.error_handler.handle_tool_error("edit_file", e, context)
            raise ToolExecutionError(
                "edit_file",
                f"Unexpected error editing file '{path}': {e}",
                original_error=e,
            )

    def list_files(self, directory: str = "/") -> List[str]:
        """
        List files in a directory.

        Args:
            directory: Directory to list (default: root)

        Returns:
            List of file paths

        Raises:
            ValueError: If directory path is invalid
        """
        try:
            return self.vfs.list_files(directory)
        except ValueError as e:
            raise ValueError(f"Invalid directory path: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error listing directory '{directory}': {e}")

    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists.

        Args:
            path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            return self.vfs.file_exists(path)
        except Exception:
            return False

    def delete_file(self, path: str) -> Dict[str, Any]:
        """
        Delete a file.

        Args:
            path: Path to the file to delete

        Returns:
            Dictionary with deletion status

        Raises:
            ValueError: If path is invalid
        """
        try:
            deleted = self.vfs.delete_file(path)
            return {
                "path": path,
                "deleted": deleted,
                "status": "success" if deleted else "file_not_found",
            }
        except ValueError as e:
            raise ValueError(f"Invalid file path: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error deleting file '{path}': {e}")


# Tool function wrappers for use with LangGraph/LangChain
def create_file_tools(
    vfs: VirtualFileSystem, error_handler: Optional[ErrorHandler] = None
) -> Dict[str, callable]:
    """
    Create file tool functions that can be used with LangGraph/LangChain.

    Args:
        vfs: VirtualFileSystem instance
        error_handler: Optional error handler for error management

    Returns:
        Dictionary of tool functions
    """
    file_tools = FileTools(vfs, error_handler)

    def read_file_tool(path: str) -> str:
        """Read content from a file."""
        return file_tools.read_file(path)

    def write_file_tool(path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        return file_tools.write_file(path, content)

    def edit_file_tool(
        path: str, edits: Union[List[Dict[str, str]], str]
    ) -> Dict[str, Any]:
        """Edit a file using find/replace operations or whole-file replacement."""
        return file_tools.edit_file(path, edits)

    return {
        "read_file": read_file_tool,
        "write_file": write_file_tool,
        "edit_file": edit_file_tool,
    }

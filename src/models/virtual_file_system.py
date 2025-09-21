"""Virtual File System implementation for context engineering and artifact persistence."""

import os
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass, field


@dataclass
class FileMetadata:
    """Metadata for virtual files."""

    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    size: int = 0
    content_type: str = "text/plain"

    def update_modified(self, size: int) -> None:
        """Update modification time and size."""
        self.modified_at = datetime.now()
        self.size = size


class VirtualFileSystem:
    """
    Simulated file system for context engineering and artifact persistence.

    Provides file operations that work entirely in memory, allowing agents
    to create, read, write, and edit files for context sharing and artifact
    storage without affecting the actual file system.
    """

    def __init__(self, root: str = "/virtual"):
        """
        Initialize the virtual file system.

        Args:
            root: Root directory path for the virtual file system
        """
        self.root = root
        self.files: Dict[str, str] = {}
        self.metadata: Dict[str, FileMetadata] = {}

    def _normalize_path(self, path: str) -> str:
        """
        Normalize and validate file path.

        Args:
            path: File path to normalize

        Returns:
            Normalized absolute path

        Raises:
            ValueError: If path is invalid or contains dangerous patterns
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Check for dangerous patterns first
        if any(dangerous in path for dangerous in ["../", "..", "~"]):
            raise ValueError(f"Invalid path contains dangerous components: {path}")

        # Remove leading slash if present
        if path.startswith("/"):
            path = path.lstrip("/")

        # Normalize the path
        path = os.path.normpath(path)

        # Check again after normalization for any dangerous patterns
        if any(dangerous in path for dangerous in ["../", "..", "~"]):
            raise ValueError(f"Invalid path contains dangerous components: {path}")

        # Create absolute path within virtual root
        normalized = os.path.join(self.root, path).replace("\\", "/")
        return normalized

    def _validate_path(self, path: str) -> str:
        """
        Validate and normalize a file path.

        Args:
            path: Path to validate

        Returns:
            Validated normalized path

        Raises:
            ValueError: If path is invalid
        """
        try:
            normalized = self._normalize_path(path)
            return normalized
        except Exception as e:
            raise ValueError(f"Invalid path '{path}': {e}")

    def read_file(self, path: str) -> str:
        """
        Read content from a virtual file.

        Args:
            path: Path to the file to read

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
        """
        normalized_path = self._validate_path(path)

        if normalized_path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")

        return self.files[normalized_path]

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a virtual file.

        Args:
            path: Path to the file to write
            content: Content to write to the file

        Returns:
            Dictionary with path and bytes_written

        Raises:
            ValueError: If path is invalid
        """
        normalized_path = self._validate_path(path)

        # Ensure content is string
        if not isinstance(content, str):
            content = str(content)

        # Store file content
        self.files[normalized_path] = content

        # Update metadata
        bytes_written = len(content.encode("utf-8"))
        if normalized_path in self.metadata:
            self.metadata[normalized_path].update_modified(bytes_written)
        else:
            metadata = FileMetadata(size=bytes_written)
            self.metadata[normalized_path] = metadata

        return {"path": path, "bytes_written": bytes_written}

    def edit_file(
        self, path: str, edits: Union[List[Dict[str, str]], str]
    ) -> Dict[str, Any]:
        """
        Edit a virtual file using find/replace operations or whole-file replacement.

        Args:
            path: Path to the file to edit
            edits: Either a list of {"find": str, "replace": str} dicts for
                  find/replace operations, or a string for whole-file replacement

        Returns:
            Dictionary with path and diff showing changes made

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid or edits format is wrong
        """
        normalized_path = self._validate_path(path)

        if normalized_path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")

        original_content = self.files[normalized_path]

        # Handle whole-file replacement
        if isinstance(edits, str):
            new_content = edits
        elif isinstance(edits, list):
            # Handle find/replace operations
            new_content = original_content

            for edit in edits:
                if (
                    not isinstance(edit, dict)
                    or "find" not in edit
                    or "replace" not in edit
                ):
                    raise ValueError(
                        "Each edit must be a dict with 'find' and 'replace' keys"
                    )

                find_text = edit["find"]
                replace_text = edit["replace"]

                if find_text not in new_content:
                    raise ValueError(f"Text to find not found in file: '{find_text}'")

                new_content = new_content.replace(find_text, replace_text)
        else:
            raise ValueError(
                "Edits must be either a string or list of find/replace dicts"
            )

        # Generate diff
        diff = self._generate_diff(original_content, new_content, path)

        # Update file content
        self.files[normalized_path] = new_content

        # Update metadata
        bytes_written = len(new_content.encode("utf-8"))
        if normalized_path in self.metadata:
            self.metadata[normalized_path].update_modified(bytes_written)
        else:
            metadata = FileMetadata(size=bytes_written)
            self.metadata[normalized_path] = metadata

        return {"path": path, "diff": diff}

    def _generate_diff(self, original: str, new: str, filename: str) -> str:
        """
        Generate a unified diff between original and new content.

        Args:
            original: Original file content
            new: New file content
            filename: Name of the file for diff header

        Returns:
            Unified diff as string
        """
        import difflib

        original_lines = original.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                lineterm="",
            )
        )

        return "".join(diff_lines)

    def list_files(self, directory: str = "/") -> List[str]:
        """
        List all files in the virtual file system or a specific directory.

        Args:
            directory: Directory to list (default: root)

        Returns:
            List of file paths
        """
        if directory == "/":
            directory = self.root
        else:
            directory = self._validate_path(directory)

        # Find all files that start with the directory path
        matching_files = []
        for file_path in self.files.keys():
            if file_path.startswith(directory):
                # Get relative path from directory
                relative_path = file_path[len(directory) :].lstrip("/")
                if relative_path and "/" not in relative_path:
                    matching_files.append(relative_path)

        return sorted(matching_files)

    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists in the virtual file system.

        Args:
            path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            normalized_path = self._validate_path(path)
            return normalized_path in self.files
        except ValueError:
            return False

    def get_file_metadata(self, path: str) -> Optional[FileMetadata]:
        """
        Get metadata for a file.

        Args:
            path: Path to the file

        Returns:
            FileMetadata object or None if file doesn't exist
        """
        try:
            normalized_path = self._validate_path(path)
            return self.metadata.get(normalized_path)
        except ValueError:
            return None

    def delete_file(self, path: str) -> bool:
        """
        Delete a file from the virtual file system.

        Args:
            path: Path to the file to delete

        Returns:
            True if file was deleted, False if it didn't exist
        """
        try:
            normalized_path = self._validate_path(path)
            if normalized_path in self.files:
                del self.files[normalized_path]
                if normalized_path in self.metadata:
                    del self.metadata[normalized_path]
                return True
            return False
        except ValueError:
            return False

    def clear(self) -> None:
        """Clear all files from the virtual file system."""
        self.files.clear()
        self.metadata.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the virtual file system.

        Returns:
            Dictionary with file count, total size, etc.
        """
        total_size = sum(
            len(content.encode("utf-8")) for content in self.files.values()
        )

        return {
            "file_count": len(self.files),
            "total_size_bytes": total_size,
            "files": list(self.files.keys()),
        }

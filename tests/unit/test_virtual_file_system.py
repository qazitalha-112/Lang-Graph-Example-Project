"""Unit tests for VirtualFileSystem."""

import pytest
from datetime import datetime
from src.models.virtual_file_system import VirtualFileSystem, FileMetadata


class TestVirtualFileSystem:
    """Test cases for VirtualFileSystem class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.vfs = VirtualFileSystem()

    def test_init(self):
        """Test VirtualFileSystem initialization."""
        vfs = VirtualFileSystem()
        assert vfs.root == "/virtual"
        assert vfs.files == {}
        assert vfs.metadata == {}

        # Test custom root
        custom_vfs = VirtualFileSystem("/custom")
        assert custom_vfs.root == "/custom"

    def test_normalize_path(self):
        """Test path normalization."""
        # Test basic path normalization
        assert self.vfs._normalize_path("test.txt") == "/virtual/test.txt"
        assert self.vfs._normalize_path("folder/test.txt") == "/virtual/folder/test.txt"

        # Test absolute path handling
        assert self.vfs._normalize_path("/test.txt") == "/virtual/test.txt"

        # Test dangerous path rejection
        with pytest.raises(
            ValueError, match="Invalid path contains dangerous components"
        ):
            self.vfs._normalize_path("../test.txt")

        with pytest.raises(
            ValueError, match="Invalid path contains dangerous components"
        ):
            self.vfs._normalize_path("folder/../test.txt")

        with pytest.raises(
            ValueError, match="Invalid path contains dangerous components"
        ):
            self.vfs._normalize_path("~/test.txt")

    def test_validate_path(self):
        """Test path validation."""
        # Valid paths
        assert self.vfs._validate_path("test.txt") == "/virtual/test.txt"
        assert self.vfs._validate_path("folder/test.txt") == "/virtual/folder/test.txt"

        # Invalid paths
        with pytest.raises(ValueError, match="Path cannot be empty"):
            self.vfs._validate_path("")

        with pytest.raises(ValueError, match="Invalid path"):
            self.vfs._validate_path("../test.txt")

    def test_write_file_basic(self):
        """Test basic file writing functionality."""
        content = "Hello, World!"
        result = self.vfs.write_file("test.txt", content)

        # Check return value
        assert result["path"] == "test.txt"
        assert result["bytes_written"] == len(content.encode("utf-8"))

        # Check internal storage
        normalized_path = "/virtual/test.txt"
        assert normalized_path in self.vfs.files
        assert self.vfs.files[normalized_path] == content

        # Check metadata
        assert normalized_path in self.vfs.metadata
        metadata = self.vfs.metadata[normalized_path]
        assert metadata.size == len(content.encode("utf-8"))
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.modified_at, datetime)

    def test_write_file_overwrite(self):
        """Test overwriting existing files."""
        # Write initial content
        self.vfs.write_file("test.txt", "Initial content")
        initial_metadata = self.vfs.metadata["/virtual/test.txt"]

        # Small delay to ensure time difference
        import time

        time.sleep(0.01)

        # Overwrite with new content
        new_content = "New content"
        result = self.vfs.write_file("test.txt", new_content)

        assert result["path"] == "test.txt"
        assert result["bytes_written"] == len(new_content.encode("utf-8"))
        assert self.vfs.files["/virtual/test.txt"] == new_content

        # Check metadata was updated
        updated_metadata = self.vfs.metadata["/virtual/test.txt"]
        assert updated_metadata.size == len(new_content.encode("utf-8"))
        # Since we're reusing the same metadata object, check that it's been updated
        assert updated_metadata is initial_metadata  # Same object
        assert updated_metadata.size != len(
            "Initial content".encode("utf-8")
        )  # Size changed

    def test_write_file_non_string(self):
        """Test writing non-string content."""
        result = self.vfs.write_file("test.txt", 12345)
        assert self.vfs.files["/virtual/test.txt"] == "12345"
        assert result["bytes_written"] == 5

    def test_read_file_success(self):
        """Test successful file reading."""
        content = "Test content for reading"
        self.vfs.write_file("test.txt", content)

        read_content = self.vfs.read_file("test.txt")
        assert read_content == content

    def test_read_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found: nonexistent.txt"):
            self.vfs.read_file("nonexistent.txt")

    def test_read_file_invalid_path(self):
        """Test reading with invalid path."""
        with pytest.raises(ValueError, match="Invalid path"):
            self.vfs.read_file("../test.txt")


class TestVirtualFileSystemEdit:
    """Test cases for VirtualFileSystem edit functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.vfs = VirtualFileSystem()
        self.test_content = """Line 1: Hello
Line 2: World
Line 3: Test
Line 4: Content"""
        self.vfs.write_file("test.txt", self.test_content)

    def test_edit_file_find_replace_single(self):
        """Test single find/replace operation."""
        edits = [{"find": "Hello", "replace": "Hi"}]
        result = self.vfs.edit_file("test.txt", edits)

        assert result["path"] == "test.txt"
        assert "diff" in result

        # Check content was updated
        updated_content = self.vfs.read_file("test.txt")
        assert "Hi" in updated_content
        assert "Hello" not in updated_content
        assert "Line 1: Hi" in updated_content

    def test_edit_file_find_replace_multiple(self):
        """Test multiple find/replace operations."""
        edits = [
            {"find": "Hello", "replace": "Hi"},
            {"find": "World", "replace": "Universe"},
            {"find": "Test", "replace": "Example"},
        ]
        result = self.vfs.edit_file("test.txt", edits)

        updated_content = self.vfs.read_file("test.txt")
        assert "Hi" in updated_content
        assert "Universe" in updated_content
        assert "Example" in updated_content
        assert "Hello" not in updated_content
        assert "World" not in updated_content
        assert "Test" not in updated_content

    def test_edit_file_whole_replacement(self):
        """Test whole-file replacement."""
        new_content = "Completely new content"
        result = self.vfs.edit_file("test.txt", new_content)

        assert result["path"] == "test.txt"
        assert "diff" in result

        updated_content = self.vfs.read_file("test.txt")
        assert updated_content == new_content

    def test_edit_file_not_found(self):
        """Test editing non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found: nonexistent.txt"):
            self.vfs.edit_file("nonexistent.txt", [{"find": "test", "replace": "new"}])

    def test_edit_file_find_not_found(self):
        """Test find/replace when text to find doesn't exist."""
        edits = [{"find": "NonexistentText", "replace": "Replacement"}]

        with pytest.raises(ValueError, match="Text to find not found in file"):
            self.vfs.edit_file("test.txt", edits)

    def test_edit_file_invalid_edit_format(self):
        """Test invalid edit format."""
        # Missing 'replace' key
        with pytest.raises(
            ValueError, match="Each edit must be a dict with 'find' and 'replace' keys"
        ):
            self.vfs.edit_file("test.txt", [{"find": "Hello"}])

        # Missing 'find' key
        with pytest.raises(
            ValueError, match="Each edit must be a dict with 'find' and 'replace' keys"
        ):
            self.vfs.edit_file("test.txt", [{"replace": "Hi"}])

        # Invalid type
        with pytest.raises(ValueError, match="Edits must be either a string or list"):
            self.vfs.edit_file("test.txt", 123)

    def test_generate_diff(self):
        """Test diff generation."""
        original = "Line 1\nLine 2\nLine 3"
        new = "Line 1\nModified Line 2\nLine 3"

        diff = self.vfs._generate_diff(original, new, "test.txt")

        assert "--- a/test.txt" in diff
        assert "+++ b/test.txt" in diff
        assert "-Line 2" in diff
        assert "+Modified Line 2" in diff


class TestVirtualFileSystemUtilities:
    """Test cases for VirtualFileSystem utility methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.vfs = VirtualFileSystem()

        # Create some test files
        self.vfs.write_file("file1.txt", "Content 1")
        self.vfs.write_file("file2.txt", "Content 2")
        self.vfs.write_file("folder/file3.txt", "Content 3")
        self.vfs.write_file("folder/subfolder/file4.txt", "Content 4")

    def test_list_files_root(self):
        """Test listing files in root directory."""
        files = self.vfs.list_files()

        # Should only show files directly in root, not in subdirectories
        assert "file1.txt" in files
        assert "file2.txt" in files
        assert len(files) == 2

    def test_list_files_subdirectory(self):
        """Test listing files in subdirectory."""
        files = self.vfs.list_files("folder")

        assert "file3.txt" in files
        assert len(files) == 1  # subfolder/file4.txt should not be included

    def test_file_exists(self):
        """Test file existence checking."""
        assert self.vfs.file_exists("file1.txt") is True
        assert self.vfs.file_exists("folder/file3.txt") is True
        assert self.vfs.file_exists("nonexistent.txt") is False
        assert self.vfs.file_exists("../invalid.txt") is False

    def test_get_file_metadata(self):
        """Test getting file metadata."""
        metadata = self.vfs.get_file_metadata("file1.txt")

        assert metadata is not None
        assert isinstance(metadata, FileMetadata)
        assert metadata.size == len("Content 1".encode("utf-8"))
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.modified_at, datetime)

        # Test non-existent file
        assert self.vfs.get_file_metadata("nonexistent.txt") is None

    def test_delete_file(self):
        """Test file deletion."""
        # Delete existing file
        assert self.vfs.delete_file("file1.txt") is True
        assert not self.vfs.file_exists("file1.txt")

        # Try to delete non-existent file
        assert self.vfs.delete_file("nonexistent.txt") is False

        # Try to delete with invalid path
        assert self.vfs.delete_file("../invalid.txt") is False

    def test_clear(self):
        """Test clearing all files."""
        assert len(self.vfs.files) > 0
        assert len(self.vfs.metadata) > 0

        self.vfs.clear()

        assert len(self.vfs.files) == 0
        assert len(self.vfs.metadata) == 0

    def test_get_stats(self):
        """Test getting file system statistics."""
        stats = self.vfs.get_stats()

        assert stats["file_count"] == 4
        assert stats["total_size_bytes"] > 0
        assert len(stats["files"]) == 4
        assert "/virtual/file1.txt" in stats["files"]
        assert "/virtual/folder/file3.txt" in stats["files"]


class TestFileMetadata:
    """Test cases for FileMetadata class."""

    def test_init(self):
        """Test FileMetadata initialization."""
        metadata = FileMetadata()

        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.modified_at, datetime)
        assert metadata.size == 0
        assert metadata.content_type == "text/plain"

    def test_update_modified(self):
        """Test updating modification time and size."""
        metadata = FileMetadata()
        original_modified = metadata.modified_at

        # Small delay to ensure time difference
        import time

        time.sleep(0.01)

        metadata.update_modified(100)

        assert metadata.size == 100
        assert metadata.modified_at > original_modified

"""Tests for memory export/import functionality."""

import os
import pytest
import asyncio
import json
import csv
from typing import AsyncGenerator, List
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timezone

from core.memory_export import MemoryExporter, MemoryImporter
from core.memory_manager import MemorySegment, MemorySegmentContent, MemoryManager
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface

class DummyLLM(BaseLLMInterface):
    def __init__(self):
        # Don't call super() since we're just using this for testing
        pass

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "dummy response"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"
        yield " stream"

    async def get_embedding(self, text, model=None):
        return [0.1] * 384

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_segments() -> List[MemorySegment]:
    """Create a list of test memory segments."""
    return [
        MemorySegment(
            id=f"test-id-{i}",
            type="test",
            source="unit-test",
            content=MemorySegmentContent(text=f"Test content {i}"),
            importance=0.5 + (i / 10.0),
            embedding=[0.1] * 384 if i % 2 == 0 else None,  # Some with embeddings, some without
            metadata={"index": i, "test": True}
        )
        for i in range(5)
    ]

@pytest.fixture
def mock_memory_manager(test_segments):
    """Create a mock memory manager with test segments."""
    manager = MagicMock(spec=MemoryManager)

    # Mock get_recent_memory to return test segments
    async def mock_get_recent_memory(limit=100, filter_dict=None):
        return test_segments

    # Mock get_memory to simulate looking up by ID
    async def mock_get_memory(segment_id):
        for segment in test_segments:
            if segment.id == segment_id:
                return segment
        return None

    # Mock add_segment to just return the segment ID
    async def mock_add_segment(segment):
        return segment.id

    # Mock add_memory to create a segment
    async def mock_add_memory(type, source, content_text=None, tool_name=None,
                           tool_args=None, tool_output=None, importance=0.5, metadata=None):
        from core.memory_manager import MemorySegment, MemorySegmentContent
        import uuid
        segment = MemorySegment(
            id=str(uuid.uuid4()),
            type=type,
            source=source,
            content=MemorySegmentContent(
                text=content_text,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_output=tool_output
            ),
            importance=importance,
            metadata=metadata or {}
        )
        return segment.id

    manager.get_recent_memory.side_effect = mock_get_recent_memory
    manager.get_memory.side_effect = mock_get_memory
    manager.add_segment.side_effect = mock_add_segment
    manager.add_memory.side_effect = mock_add_memory

    return manager

@pytest.mark.asyncio
async def test_export_to_json(mock_memory_manager, temp_dir):
    """Test exporting memory segments to JSON."""
    exporter = MemoryExporter(mock_memory_manager)

    # Export with embeddings
    output_path = os.path.join(temp_dir, "memory_export.json")
    count = await exporter.export_to_json(output_path)

    # Verify
    assert count == 5
    assert os.path.exists(output_path)

    # Check content
    with open(output_path, 'r') as f:
        data = json.load(f)

    assert "metadata" in data
    assert "segments" in data
    assert len(data["segments"]) == 5
    assert "embedding" in data["segments"][0]

    # Export without embeddings
    output_path2 = os.path.join(temp_dir, "memory_export_no_embeddings.json")
    count = await exporter.export_to_json(output_path2, include_embeddings=False)

    # Check content
    with open(output_path2, 'r') as f:
        data = json.load(f)

    # All embeddings should be None
    for segment in data["segments"]:
        assert segment["embedding"] is None

@pytest.mark.asyncio
async def test_export_to_csv(mock_memory_manager, temp_dir):
    """Test exporting memory segments to CSV."""
    exporter = MemoryExporter(mock_memory_manager)

    output_path = os.path.join(temp_dir, "memory_export.csv")
    count = await exporter.export_to_csv(output_path)

    # Verify
    assert count == 5
    assert os.path.exists(output_path)

    # Check content
    with open(output_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 5
    assert 'id' in rows[0]
    assert 'type' in rows[0]
    assert 'content_text' in rows[0]
    assert 'metadata' in rows[0]

    # Check that metadata is properly JSON encoded
    metadata = json.loads(rows[0]['metadata'])
    assert 'test' in metadata
    assert metadata['test'] is True

@pytest.mark.asyncio
async def test_export_content_only(mock_memory_manager, temp_dir):
    """Test exporting only memory content to text file."""
    exporter = MemoryExporter(mock_memory_manager)

    # Test without metadata
    output_path = os.path.join(temp_dir, "memory_content.txt")
    count = await exporter.export_content_only(output_path)

    # Verify
    assert count == 5
    assert os.path.exists(output_path)

    # Check content
    with open(output_path, 'r') as f:
        content = f.read()

    # Should contain text from all segments, but no metadata headers
    for i in range(5):
        assert f"Test content {i}" in content
    assert "--- Segment" not in content

    # Test with metadata
    output_path2 = os.path.join(temp_dir, "memory_content_with_metadata.txt")
    count = await exporter.export_content_only(output_path2, include_metadata=True)

    # Check content
    with open(output_path2, 'r') as f:
        content = f.read()

    # Should contain both text and metadata headers
    assert "--- Segment" in content
    assert "test/unit-test" in content

@pytest.mark.asyncio
async def test_import_from_json(mock_memory_manager, temp_dir, test_segments):
    """Test importing memory segments from JSON."""
    # First, export segments to a file
    exporter = MemoryExporter(mock_memory_manager)
    output_path = os.path.join(temp_dir, "memory_export.json")
    await exporter.export_to_json(output_path)

    # Now test importing
    importer = MemoryImporter(mock_memory_manager)
    count = await importer.import_from_json(output_path)

    # Verify
    assert count == 0  # All should be skipped as existing

    # Test with skip_existing=False
    mock_memory_manager.add_segment.reset_mock()
    count = await importer.import_from_json(output_path, skip_existing=False)

    # Verify
    assert count == 5
    assert mock_memory_manager.add_segment.call_count == 5

    # Test with regenerate_embeddings=True
    mock_memory_manager.add_segment.reset_mock()
    count = await importer.import_from_json(
        output_path,
        skip_existing=False,
        regenerate_embeddings=True
    )

    # Verify
    assert count == 5
    # Check that embeddings were cleared
    for call in mock_memory_manager.add_segment.call_args_list:
        segment = call[0][0]
        assert segment.embedding is None

@pytest.mark.asyncio
async def test_import_from_csv(mock_memory_manager, temp_dir):
    """Test importing memory segments from CSV."""
    # First, export segments to a file
    exporter = MemoryExporter(mock_memory_manager)
    output_path = os.path.join(temp_dir, "memory_export.csv")
    await exporter.export_to_csv(output_path)

    # Now test importing
    importer = MemoryImporter(mock_memory_manager)
    count = await importer.import_from_csv(output_path)

    # Verify
    assert count == 0  # All should be skipped as existing

    # Test with skip_existing=False
    mock_memory_manager.add_segment.reset_mock()
    count = await importer.import_from_csv(output_path, skip_existing=False)

    # Verify
    assert count == 5
    assert mock_memory_manager.add_segment.call_count == 5

@pytest.mark.asyncio
async def test_import_text_as_segments(mock_memory_manager, temp_dir):
    """Test importing text file as memory segments."""
    # Create a test text file
    text_path = os.path.join(temp_dir, "test_content.txt")
    with open(text_path, 'w') as f:
        f.write("This is segment 1\n")
        f.write("---\n")
        f.write("This is segment 2\n")
        f.write("---\n")
        f.write("This is segment 3\n")

    # Test importing
    importer = MemoryImporter(mock_memory_manager)
    count = await importer.import_text_as_segments(
        text_path,
        segment_type="TEXT_IMPORT",
        source="test",
        metadata={"test_import": True}
    )

    # Verify
    assert count == 3
    assert mock_memory_manager.add_memory.call_count == 3

    # Check the arguments for one call
    call_args = mock_memory_manager.add_memory.call_args_list[0][1]
    assert call_args["type"] == "TEXT_IMPORT"
    assert call_args["source"] == "test"
    assert call_args["metadata"]["test_import"] is True
    assert "source_file" in call_args["metadata"]

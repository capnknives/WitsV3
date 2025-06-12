"""
Memory export/import functionality for WitsV3.

This module provides utilities for exporting memory segments to different formats
and importing them back.
"""

import json
import csv
import logging
import os
from typing import List, Dict, Any, Optional, Union, TextIO
from datetime import datetime
from pathlib import Path

from .memory_manager import MemorySegment, MemoryManager
from .config import WitsV3Config

logger = logging.getLogger(__name__)

class MemoryExporter:
    """Utility for exporting memory segments to different formats."""

    def __init__(self, memory_manager: MemoryManager):
        """Initialize the memory exporter.

        Args:
            memory_manager: Memory manager instance to export from
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)

    async def export_to_json(
        self,
        output_path: Union[str, Path],
        limit: int = 0,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = True
    ) -> int:
        """Export memory segments to a JSON file.

        Args:
            output_path: Path to output JSON file
            limit: Maximum number of segments to export (0 for all)
            filter_dict: Optional filter for memory segments
            include_embeddings: Whether to include embedding vectors

        Returns:
            Number of segments exported
        """
        try:
            # Retrieve segments
            segments = await self.memory_manager.get_recent_memory(
                limit=limit if limit > 0 else 100000,  # Large number to get all
                filter_dict=filter_dict
            )

            if not include_embeddings:
                # Create copies without embeddings
                segments = [segment.model_copy(update={"embedding": None}) for segment in segments]

            # Convert to dictionaries
            segments_data = [segment.model_dump() for segment in segments]

            # Ensure the output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "export_date": datetime.now().isoformat(),
                        "segment_count": len(segments),
                        "format_version": "1.0"
                    },
                    "segments": segments_data
                }, f, indent=2)

            self.logger.info(f"Exported {len(segments)} segments to {output_path}")
            return len(segments)

        except Exception as e:
            self.logger.error(f"Error exporting memory to JSON: {e}")
            raise

    async def export_to_csv(
        self,
        output_path: Union[str, Path],
        limit: int = 0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        """Export memory segments to a CSV file.

        Args:
            output_path: Path to output CSV file
            limit: Maximum number of segments to export (0 for all)
            filter_dict: Optional filter for memory segments

        Returns:
            Number of segments exported
        """
        try:
            # Retrieve segments
            segments = await self.memory_manager.get_recent_memory(
                limit=limit if limit > 0 else 100000,  # Large number to get all
                filter_dict=filter_dict
            )

            # Ensure the output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Define CSV fields
            fields = [
                'id', 'timestamp', 'type', 'source',
                'content_text', 'content_tool_name', 'content_tool_output',
                'importance', 'metadata'
            ]

            # Write to file
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()

                for segment in segments:
                    # Flatten the structure for CSV
                    row = {
                        'id': segment.id,
                        'timestamp': segment.timestamp.isoformat(),
                        'type': segment.type,
                        'source': segment.source,
                        'content_text': segment.content.text,
                        'content_tool_name': segment.content.tool_name,
                        'content_tool_output': segment.content.tool_output,
                        'importance': segment.importance,
                        'metadata': json.dumps(segment.metadata)
                    }
                    writer.writerow(row)

            self.logger.info(f"Exported {len(segments)} segments to {output_path}")
            return len(segments)

        except Exception as e:
            self.logger.error(f"Error exporting memory to CSV: {e}")
            raise

    async def export_content_only(
        self,
        output_path: Union[str, Path],
        limit: int = 0,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False
    ) -> int:
        """Export only the text content of memory segments to a text file.

        Args:
            output_path: Path to output text file
            limit: Maximum number of segments to export (0 for all)
            filter_dict: Optional filter for memory segments
            include_metadata: Whether to include a metadata header for each segment

        Returns:
            Number of segments exported
        """
        try:
            # Retrieve segments
            segments = await self.memory_manager.get_recent_memory(
                limit=limit if limit > 0 else 100000,  # Large number to get all
                filter_dict=filter_dict
            )

            # Ensure the output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in segments:
                    if include_metadata:
                        f.write(f"--- Segment {segment.id} ({segment.type}/{segment.source}) ---\n")

                    # Write text content if available
                    if segment.content.text:
                        f.write(f"{segment.content.text}\n")

                    # Write tool output if available and no text content
                    elif segment.content.tool_output:
                        f.write(f"{segment.content.tool_output}\n")

                    if include_metadata:
                        f.write("\n")  # Add extra spacing between segments

            self.logger.info(f"Exported content of {len(segments)} segments to {output_path}")
            return len(segments)

        except Exception as e:
            self.logger.error(f"Error exporting memory content: {e}")
            raise

class MemoryImporter:
    """Utility for importing memory segments from different formats."""

    def __init__(self, memory_manager: MemoryManager):
        """Initialize the memory importer.

        Args:
            memory_manager: Memory manager instance to import into
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)

    async def import_from_json(
        self,
        input_path: Union[str, Path],
        skip_existing: bool = True,
        regenerate_embeddings: bool = False
    ) -> int:
        """Import memory segments from a JSON file.

        Args:
            input_path: Path to input JSON file
            skip_existing: Whether to skip segments that already exist
            regenerate_embeddings: Whether to regenerate embeddings

        Returns:
            Number of segments imported
        """
        try:
            input_path = Path(input_path)

            if not input_path.exists():
                self.logger.error(f"Input file not found: {input_path}")
                return 0

            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both formats: direct list or nested structure
            if isinstance(data, dict) and "segments" in data:
                segments_data = data["segments"]
            else:
                segments_data = data

            if not isinstance(segments_data, list):
                self.logger.error(f"Invalid JSON format, expected list of segments: {input_path}")
                return 0

            # Import segments
            count = 0
            for segment_data in segments_data:
                # Create segment object
                segment = MemorySegment.model_validate(segment_data)

                # Check if segment already exists
                if skip_existing:
                    existing = await self.memory_manager.get_memory(segment.id)
                    if existing:
                        self.logger.debug(f"Skipping existing segment: {segment.id}")
                        continue

                # Clear embedding if regeneration is requested
                if regenerate_embeddings:
                    segment.embedding = None

                # Add to memory manager
                await self.memory_manager.add_segment(segment)
                count += 1

            self.logger.info(f"Imported {count} segments from {input_path}")
            return count

        except Exception as e:
            self.logger.error(f"Error importing memory from JSON: {e}")
            raise

    async def import_from_csv(
        self,
        input_path: Union[str, Path],
        skip_existing: bool = True
    ) -> int:
        """Import memory segments from a CSV file.

        Args:
            input_path: Path to input CSV file
            skip_existing: Whether to skip segments that already exist

        Returns:
            Number of segments imported
        """
        try:
            input_path = Path(input_path)

            if not input_path.exists():
                self.logger.error(f"Input file not found: {input_path}")
                return 0

            # Import segments
            count = 0
            with open(input_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Check for required fields
                    if not all(field in row for field in ['id', 'type', 'source']):
                        self.logger.warning(f"Skipping row with missing required fields")
                        continue

                    # Check if segment already exists
                    if skip_existing:
                        existing = await self.memory_manager.get_memory(row['id'])
                        if existing:
                            self.logger.debug(f"Skipping existing segment: {row['id']}")
                            continue

                    # Parse timestamp
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                    except (ValueError, KeyError):
                        timestamp = datetime.now()

                    # Parse metadata
                    try:
                        metadata = json.loads(row.get('metadata', '{}'))
                    except json.JSONDecodeError:
                        metadata = {}

                    # Create and add memory segment
                    from .memory_manager import MemorySegmentContent

                    segment = MemorySegment(
                        id=row['id'],
                        timestamp=timestamp,
                        type=row['type'],
                        source=row['source'],
                        content=MemorySegmentContent(
                            text=row.get('content_text'),
                            tool_name=row.get('content_tool_name'),
                            tool_output=row.get('content_tool_output')
                        ),
                        importance=float(row.get('importance', 0.5)),
                        metadata=metadata
                    )

                    # Add to memory manager
                    await self.memory_manager.add_segment(segment)
                    count += 1

            self.logger.info(f"Imported {count} segments from {input_path}")
            return count

        except Exception as e:
            self.logger.error(f"Error importing memory from CSV: {e}")
            raise

    async def import_text_as_segments(
        self,
        input_path: Union[str, Path],
        segment_type: str = "IMPORTED",
        source: str = "import_utility",
        delimiter: str = "---",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Import a text file as multiple memory segments.

        Args:
            input_path: Path to input text file
            segment_type: Type to assign to imported segments
            source: Source to assign to imported segments
            delimiter: String that separates segments in the text file
            importance: Importance score for imported segments
            metadata: Additional metadata to add to segments

        Returns:
            Number of segments imported
        """
        try:
            input_path = Path(input_path)

            if not input_path.exists():
                self.logger.error(f"Input file not found: {input_path}")
                return 0

            # Read the entire file
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by delimiter
            segments_text = content.split(delimiter)

            # Import segments
            count = 0
            for text in segments_text:
                text = text.strip()
                if not text:
                    continue

                # Create metadata with source file info
                segment_metadata = {
                    "source_file": str(input_path),
                    "import_timestamp": datetime.now().isoformat()
                }

                # Add custom metadata
                if metadata:
                    segment_metadata.update(metadata)

                # Add to memory manager
                await self.memory_manager.add_memory(
                    type=segment_type,
                    source=source,
                    content_text=text,
                    importance=importance,
                    metadata=segment_metadata
                )
                count += 1

            self.logger.info(f"Imported {count} text segments from {input_path}")
            return count

        except Exception as e:
            self.logger.error(f"Error importing text as segments: {e}")
            raise


async def export_memory(memory_manager: MemoryManager, output_path: str, limit: int = 0) -> int:
    """Convenience wrapper used by tests to export memory to JSON."""
    exporter = MemoryExporter(memory_manager)
    return await exporter.export_to_json(output_path, limit=limit)

# tools/document_tools.py
"""
Document RAG tools for WitsV3.

Drop files into the configured documents folder (default: ./documents) and
they are chunked, embedded, and stored as DOCUMENT_CHUNK memory segments.
Agents can then answer questions from them via the document_search tool.

Both tools receive their dependencies (config, LLM interface, memory manager)
from WitsV3System during startup via set_dependencies() — they must share the
live MemoryManager instance rather than creating their own, since backends
persist the whole segment list to disk on every write.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.base_tool import BaseTool

logger = logging.getLogger(__name__)

DOCUMENT_SEGMENT_TYPE = "DOCUMENT_CHUNK"

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


def _read_file_text(path: Path) -> Optional[str]:
    """Read a file's text content. Returns None if unsupported/unreadable."""
    if path.suffix.lower() == ".pdf":
        if not HAS_PYPDF:
            logger.warning(f"Skipping {path.name}: PDF support requires 'pip install pypdf'")
            return None
        try:
            reader = PdfReader(str(path))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            logger.warning(f"Skipping {path.name}: failed to read PDF ({e})")
            return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Skipping {path.name}: failed to read ({e})")
        return None


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into chunks of roughly chunk_size characters.

    Splits on paragraph boundaries where possible so chunks stay coherent;
    consecutive chunks share chunk_overlap characters of context.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        # Hard-split paragraphs that are larger than a whole chunk
        while len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = current[-chunk_overlap:] if chunk_overlap else ""
            head, para = para[:chunk_size], para[chunk_size:]
            chunks.append((current + " " + head).strip() if current else head)
            current = head[-chunk_overlap:] if chunk_overlap else ""

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            overlap = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap}\n\n{para}".strip() if overlap else para
        else:
            current = candidate

    if current and (not chunks or current != chunks[-1]):
        chunks.append(current)

    return [c for c in chunks if c.strip()]


class _DocumentToolBase(BaseTool):
    """Shared dependency plumbing for the document tools."""

    def __init__(self, name: str, description: str):
        super().__init__(name=name, description=description)
        self.config = None
        self.llm_interface = None
        self.memory_manager = None

    def set_dependencies(self, config, llm_interface, memory_manager) -> None:
        """Called by WitsV3System after startup wiring."""
        self.config = config
        self.llm_interface = llm_interface
        self.memory_manager = memory_manager

    def _not_ready(self) -> Optional[Dict[str, Any]]:
        if self.memory_manager is None or self.config is None:
            return {
                "success": False,
                "error": "Document tools are not initialized (system startup has not wired dependencies yet)"
            }
        return None


class DocumentIngestTool(_DocumentToolBase):
    """Scan the documents folder and (re-)ingest changed files into memory."""

    def __init__(self):
        super().__init__(
            name="ingest_documents",
            description=(
                "Scan the documents folder and ingest new or changed files into "
                "searchable memory. Returns a summary of what was ingested."
            )
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        not_ready = self._not_ready()
        if not_ready:
            return not_ready

        settings = self.config.document_rag
        docs_dir = Path(settings.documents_path)
        if not docs_dir.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)

        extensions = {ext.lower() for ext in settings.file_extensions}
        summary = {
            "success": True,
            "files_scanned": 0,
            "files_ingested": 0,
            "files_unchanged": 0,
            "files_removed": 0,
            "chunks_added": 0,
            "errors": [],
        }

        # Map of file_path -> file_hash for chunks currently in memory
        existing = await self.memory_manager.get_recent_memory(
            limit=1_000_000, filter_dict={"type": DOCUMENT_SEGMENT_TYPE}
        )
        stored_hashes: Dict[str, str] = {}
        for seg in existing:
            fp = seg.metadata.get("file_path")
            if fp:
                stored_hashes[fp] = seg.metadata.get("file_hash", "")

        seen_paths = set()

        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue

            rel_path = path.relative_to(docs_dir).as_posix()
            seen_paths.add(rel_path)
            summary["files_scanned"] += 1

            try:
                raw = path.read_bytes()
                file_hash = hashlib.sha256(raw).hexdigest()

                if stored_hashes.get(rel_path) == file_hash:
                    summary["files_unchanged"] += 1
                    continue

                text = _read_file_text(path)
                if text is None:
                    continue

                chunks = _chunk_text(text, settings.chunk_size, settings.chunk_overlap)
                if not chunks:
                    summary["files_unchanged"] += 1
                    continue

                # Replace any previous version of this file
                if rel_path in stored_hashes:
                    await self.memory_manager.delete_segments(
                        {"type": DOCUMENT_SEGMENT_TYPE, "file_path": rel_path}
                    )

                ingested_at = datetime.now(timezone.utc).isoformat()
                for i, chunk in enumerate(chunks):
                    await self.memory_manager.add_memory(
                        type=DOCUMENT_SEGMENT_TYPE,
                        source=path.name,
                        content_text=chunk,
                        importance=0.6,
                        metadata={
                            "file_path": rel_path,
                            "file_hash": file_hash,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "ingested_at": ingested_at,
                        },
                    )

                summary["files_ingested"] += 1
                summary["chunks_added"] += len(chunks)
                self.logger.info(f"Ingested {rel_path}: {len(chunks)} chunks")

            except Exception as e:
                self.logger.error(f"Error ingesting {rel_path}: {e}")
                summary["errors"].append(f"{rel_path}: {e}")

        # Remove chunks for files that no longer exist on disk
        for stale_path in set(stored_hashes) - seen_paths:
            removed = await self.memory_manager.delete_segments(
                {"type": DOCUMENT_SEGMENT_TYPE, "file_path": stale_path}
            )
            if removed:
                summary["files_removed"] += 1
                self.logger.info(f"Removed {removed} chunks for deleted file {stale_path}")

        return summary

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


class DocumentSearchTool(_DocumentToolBase):
    """Semantic search over ingested documents."""

    def __init__(self):
        super().__init__(
            name="document_search",
            description=(
                "Search the user's ingested documents by meaning and return the "
                "most relevant passages with their source files. Use this when "
                "the user asks about their documents, notes, or files."
            )
        )

    async def execute(self, query: str = "", max_results: int = 5, min_relevance: float = 0.3) -> Dict[str, Any]:
        not_ready = self._not_ready()
        if not_ready:
            return not_ready

        if not query.strip():
            return {"success": False, "error": "query must not be empty", "results": []}

        try:
            segments = await self.memory_manager.search_memory(
                query_text=query,
                limit=max_results,
                min_relevance=min_relevance,
                filter_dict={"type": DOCUMENT_SEGMENT_TYPE},
            )

            results = [
                {
                    "file": seg.metadata.get("file_path", seg.source),
                    "chunk": f"{seg.metadata.get('chunk_index', 0) + 1}/{seg.metadata.get('total_chunks', 1)}",
                    "relevance": round(seg.relevance_score or 0.0, 3),
                    "text": seg.content.text or "",
                }
                for seg in segments
            ]

            return {
                "success": True,
                "query": query,
                "result_count": len(results),
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"Document search failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for in the documents (natural language)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of passages to return",
                        "default": 5
                    },
                    "min_relevance": {
                        "type": "number",
                        "description": "Minimum similarity score (0-1) for a passage to be included",
                        "default": 0.3
                    },
                },
                "required": ["query"],
            },
        }

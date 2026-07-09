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
from typing import Any

from core.base_tool import BaseTool
from core.document_hybrid_search import fuse_hybrid_scores

logger = logging.getLogger(__name__)

DOCUMENT_SEGMENT_TYPE = "DOCUMENT_CHUNK"

try:
    from pypdf import PdfReader

    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


def _read_file_text(path: Path) -> str | None:
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


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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
    chunks: list[str] = []
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


def _coerce_file_filters(*values: Any) -> list[str]:
    """Normalize LLM file filter args (file_name, file_names, source_files, …)."""
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                out.append(value.strip())
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
    # Preserve order while de-duplicating (case-insensitive).
    seen: set = set()
    unique: list[str] = []
    for name in out:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def _file_matches_filter(file_path: str, filters: list[str]) -> bool:
    """True when *file_path* matches any basename or substring filter."""
    if not filters:
        return True
    path_lower = (file_path or "").lower()
    base_lower = Path(file_path).name.lower()
    for filt in filters:
        fl = filt.lower()
        if fl in (path_lower, base_lower):
            return True
        if fl in path_lower or base_lower.endswith(fl) or path_lower.endswith(fl):
            return True
    return False


def _default_query_from_filename(file_name: str) -> str:
    """Build a searchable query when the LLM only passes a file name."""
    stem = Path(file_name).stem.replace("_", " ").replace("-", " ").strip()
    return stem or file_name


class _DocumentToolBase(BaseTool):
    """Shared dependency plumbing for the document tools."""

    def __init__(self, name: str, description: str):
        super().__init__(name=name, description=description)
        self.config = None
        self.llm_interface = None
        self.memory_manager = None

    def set_dependencies(self, config, llm_interface, memory_manager, **kwargs) -> None:
        """Called by WitsV3System after startup wiring."""
        self.config = config
        self.llm_interface = llm_interface
        self.memory_manager = memory_manager

    def _not_ready(self) -> dict[str, Any] | None:
        if self.memory_manager is None or self.config is None:
            return {
                "success": False,
                "error": "Document tools are not initialized (system startup has not wired dependencies yet)",
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
            ),
        )

    async def execute(self, **kwargs) -> dict[str, Any]:
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
        stored_hashes: dict[str, str] = {}
        existing_counts: dict[str, int] = {}
        for seg in existing:
            fp = seg.metadata.get("file_path")
            if fp:
                stored_hashes[fp] = seg.metadata.get("file_hash", "")
                existing_counts[fp] = existing_counts.get(fp, 0) + 1

        seen_paths = set()
        searchable: dict[str, int] = {}

        ingest_dirs = [docs_dir]
        from core.filesystem_policy import document_ingest_roots

        for extra in document_ingest_roots(self.config):
            if extra.is_dir() and extra not in ingest_dirs:
                ingest_dirs.append(extra)

        for docs_dir in ingest_dirs:
            prefix = "" if docs_dir == Path(settings.documents_path) else f"{docs_dir.name}/"

            for path in sorted(docs_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in extensions:
                    continue

                if docs_dir == Path(settings.documents_path):
                    rel_path = path.relative_to(docs_dir).as_posix()
                else:
                    rel_path = f"{prefix}{path.name}"
                seen_paths.add(rel_path)
                summary["files_scanned"] += 1

                try:
                    raw = path.read_bytes()
                    file_hash = hashlib.sha256(raw).hexdigest()

                    if stored_hashes.get(rel_path) == file_hash:
                        summary["files_unchanged"] += 1
                        searchable[rel_path] = existing_counts.get(rel_path, 0)
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
                    searchable[rel_path] = len(chunks)
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

        # Plain-language summary so the LLM doesn't misread "files_ingested: 0"
        # as "nothing is accessible" — unchanged files are already ingested.
        parts = []
        if summary["files_ingested"]:
            parts.append(
                f"ingested {summary['files_ingested']} new or changed file(s), "
                f"adding {summary['chunks_added']} chunks"
            )
        if summary["files_unchanged"]:
            parts.append(
                f"{summary['files_unchanged']} file(s) were already ingested and are unchanged"
            )
        if summary["files_removed"]:
            parts.append(f"removed {summary['files_removed']} file(s) deleted from disk")
        if searchable:
            listing = ", ".join(
                f"{name} ({count} chunks)" for name, count in sorted(searchable.items())
            )
            parts.append(
                f"ALL of these documents are ingested and searchable via document_search: {listing}"
            )
        else:
            parts.append("no documents are currently ingested")
        if summary["errors"]:
            parts.append(f"{len(summary['errors'])} file(s) failed to ingest")
        summary["searchable_files"] = searchable
        summary["message"] = "Scan complete: " + "; ".join(parts) + "."

        return summary

    def get_schema(self) -> dict[str, Any]:
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
            ),
        )

    async def execute(
        self,
        query: str = "",
        max_results: int = 5,
        min_relevance: float = 0.3,
        file_name: str | None = None,
        file_names: list[str] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        not_ready = self._not_ready()
        if not_ready:
            return not_ready

        file_filters = _coerce_file_filters(
            file_name,
            file_names,
            extra.get("filename"),
            extra.get("file"),
            extra.get("source_files"),
        )

        explicit_query = bool((query or "").strip())
        query = (query or "").strip()
        if not query and file_filters:
            query = _default_query_from_filename(file_filters[0])
        if not query:
            return {"success": False, "error": "query must not be empty", "results": []}

        try:
            if file_filters and not explicit_query:
                segments = await self._segments_for_files(file_filters, max_results)
            else:
                segments = await self._hybrid_segment_search(
                    query=query,
                    max_results=max_results,
                    min_relevance=min_relevance,
                    file_filters=file_filters,
                )

            results = [
                {
                    "file": seg.metadata.get("file_path", seg.source),
                    "chunk": f"{seg.metadata.get('chunk_index', 0) + 1}/{seg.metadata.get('total_chunks', 1)}",
                    "relevance": round(seg.relevance_score or 0.0, 3),
                    "lexical_score": round(float(seg.metadata.get("_lexical_score", 0.0)), 3),
                    "vector_score": round(float(seg.metadata.get("_vector_score", 0.0)), 3),
                    "text": seg.content.text or "",
                }
                for seg in segments
            ]

            return {
                "success": True,
                "query": query,
                "file_name": file_filters[0] if len(file_filters) == 1 else None,
                "file_names": file_filters or None,
                "result_count": len(results),
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"Document search failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    async def _segments_for_files(self, file_filters: list[str], max_results: int) -> list[Any]:
        """Return ordered chunks for file-scoped browse (summarize-by-filename)."""
        all_segments = await self.memory_manager.get_recent_memory(
            limit=1_000_000, filter_dict={"type": DOCUMENT_SEGMENT_TYPE}
        )
        matched = [
            seg
            for seg in all_segments
            if _file_matches_filter(seg.metadata.get("file_path", seg.source), file_filters)
        ]
        matched.sort(
            key=lambda s: (
                s.metadata.get("file_path", ""),
                s.metadata.get("chunk_index", 0),
            )
        )
        return matched[:max_results]

    async def _hybrid_segment_search(
        self,
        *,
        query: str,
        max_results: int,
        min_relevance: float,
        file_filters: list[str],
    ) -> list[Any]:
        """BM25 + vector fusion over ingested document chunks."""
        pool_limit = max(max_results * 4, 20)
        vector_segments = await self.memory_manager.search_memory(
            query_text=query,
            limit=pool_limit,
            min_relevance=0.0,
            filter_dict={"type": DOCUMENT_SEGMENT_TYPE},
        )
        if file_filters:
            vector_segments = [
                seg
                for seg in vector_segments
                if _file_matches_filter(seg.metadata.get("file_path", seg.source), file_filters)
            ]

        corpus = await self.memory_manager.get_recent_memory(
            limit=1_000_000, filter_dict={"type": DOCUMENT_SEGMENT_TYPE}
        )
        if file_filters:
            corpus = [
                seg
                for seg in corpus
                if _file_matches_filter(seg.metadata.get("file_path", seg.source), file_filters)
            ]

        merged: dict[str, Any] = {}
        for seg in corpus:
            merged[seg.id] = seg
            seg.relevance_score = 0.0
        for seg in vector_segments:
            merged[seg.id] = seg

        fused = fuse_hybrid_scores(list(merged.values()), query=query)
        selected: list[Any] = []
        for seg, vector_score, lexical_score, combined in fused:
            if combined < min_relevance:
                continue
            seg.relevance_score = combined
            seg.metadata["_vector_score"] = vector_score
            seg.metadata["_lexical_score"] = lexical_score
            selected.append(seg)
            if len(selected) >= max_results:
                break
        return selected

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for in the documents (natural language)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of passages to return",
                        "default": 5,
                    },
                    "min_relevance": {
                        "type": "number",
                        "description": "Minimum similarity score (0-1) for a passage to be included",
                        "default": 0.3,
                    },
                    "file_name": {
                        "type": "string",
                        "description": (
                            "Optional: limit results to this ingested file (basename or path). "
                            "Always include query too; if you only know the filename, use its "
                            "title words as query."
                        ),
                    },
                },
                "required": ["query"],
            },
        }

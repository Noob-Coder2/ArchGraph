"""
Document Consolidator (Agent-Exclusive Internal Tool)

Purpose: Reduce agent tool-call latency by merging multiple indexed files
into a single Markdown document in one shot. Instead of calling
`read_source_file` N times the agent calls `consolidate_files(paths)` once.

The agent can use Graphify results (god nodes, community clusters, dependency
paths from graphify_query / graphify_path / graphify_explain) to decide
WHICH files to consolidate before calling this.

markitdown is used at document ingestion time to convert .pdf / .docx /
.pptx / .xlsx to Markdown text so all stored content is already text by the
time consolidation is requested.
"""
import io
import os
import logging
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# MarkItDown converts rich document formats (.pdf, .docx, .pptx, .xlsx) to Markdown.
try:
    from markitdown import MarkItDown
    _MARKITDOWN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MARKITDOWN_AVAILABLE = False
    logger.warning(
        "markitdown is not installed. Rich document conversion (.pdf, .docx, .pptx, .xlsx) "
        "will fall back to raw UTF-8 text extraction. Install it with: "
        "pip install markitdown[all]"
    )

# Extensions that markitdown can meaningfully convert to Markdown
MARKITDOWN_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}

# Mapping of code file extensions to fenced-code-block language tags
CODE_LANG_MAP = {
    '.py': 'python', '.js': 'javascript', '.jsx': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript', '.html': 'html',
    '.css': 'css', '.json': 'json', '.sql': 'sql', '.java': 'java',
    '.c': 'c', '.cpp': 'cpp', '.h': 'c', '.hpp': 'cpp',
    '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.sh': 'bash',
    '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml', '.xml': 'xml',
}


class DocumentConsolidator:
    """
    Agent-exclusive file-content consolidator.

    Public API
    ----------
    read_file_content(file_obj)
        Reads a single file-like object.  Uses markitdown for rich formats,
        UTF-8 decode for everything else.  Returns (text, was_converted).

    consolidate_files(graph_engine, file_paths, include_toc=True)
        Reads every requested path from the graph engine's indexed file store
        (graph_engine.file_contents) and merges them into one Markdown
        document.  Falls back to disk for paths not in the in-memory store.
        Returns the merged Markdown string.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Single-file reading
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def read_file_content(cls, file_obj: Any) -> Tuple[str, bool]:
        """
        Reads file content and returns (markdown_text, was_converted).

        For .pdf / .docx / .pptx / .xlsx files markitdown is used.
        For plain text / code files the raw bytes are decoded as UTF-8.

        Args:
            file_obj: A file-like object (must expose .name and .read()).

        Returns:
            (content: str, was_converted: bool)
            was_converted is True when a rich document was converted to
            Markdown by markitdown.
        """
        filename = getattr(file_obj, "name", "unnamed")
        ext = Path(filename).suffix.lower()

        # Reset stream pointer
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        # ── Rich document formats via markitdown ─────────────────────────────
        if ext in MARKITDOWN_EXTENSIONS:
            if _MARKITDOWN_AVAILABLE:
                try:
                    raw_bytes = file_obj.read()
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = tmp.name
                    try:
                        md = MarkItDown()
                        result = md.convert(tmp_path)
                        md_text = result.text_content.strip()
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    if md_text:
                        return md_text, True
                    logger.warning(
                        "markitdown returned empty content for %s, falling back.", filename
                    )
                except Exception as exc:
                    logger.warning(
                        "markitdown failed for %s: %s — falling back to raw bytes.", filename, exc
                    )
            else:
                logger.warning("markitdown not available; reading %s as raw bytes.", filename)

        # ── Plain text / code files ──────────────────────────────────────────
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        if hasattr(file_obj, "read"):
            raw = file_obj.read()
            if isinstance(raw, str):
                return raw, False
            try:
                return raw.decode("utf-8"), False
            except UnicodeDecodeError:
                return raw.decode("utf-8", errors="replace"), False

        return "", False

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-file consolidation (agent tool)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def consolidate_files(
        cls,
        graph_engine: Any,
        file_paths: List[str],
        include_toc: bool = True,
    ) -> str:
        """
        Reads multiple indexed files and merges them into one Markdown document.

        The agent should use graphify_query / graphify_path / graphify_explain
        to identify structurally related file paths first, then pass those paths
        here to receive all content in a single tool-call response — avoiding
        N separate read_source_file round-trips.

        Resolution order for each path:
        1. graph_engine.file_contents  (in-memory indexed store)
        2. Disk (absolute or workspace-relative path)

        Args:
            graph_engine: A GraphifyEngine instance.
            file_paths:   List of file paths to merge.
            include_toc:  Prepend a Table of Contents (default True).

        Returns:
            Merged Markdown string, or an error message if no files resolved.
        """
        if not file_paths:
            return "_No file paths provided to consolidate._"

        file_contents_store: dict = getattr(graph_engine, "file_contents", {}) or {}

        toc_lines: List[str] = ["# Consolidated Files\n\n## Table of Contents\n"]
        body_sections: List[str] = []
        missing: List[str] = []

        for i, raw_path in enumerate(file_paths, start=1):
            # Normalise path separators for lookup
            clean = raw_path.replace("\\", "/").strip()
            content: Optional[str] = None

            # 1. In-memory indexed store — exact match
            if clean in file_contents_store:
                content = file_contents_store[clean]
            else:
                # Suffix / partial match (handles relative vs absolute variants)
                for stored_path, stored_content in file_contents_store.items():
                    sp = stored_path.replace("\\", "/")
                    if sp.endswith(clean) or clean.endswith(sp):
                        content = stored_content
                        clean = stored_path  # use the canonical stored path
                        break

            # 2. Disk fallback
            if content is None:
                disk_path = Path(clean)
                if not disk_path.is_absolute():
                    disk_path = Path.cwd() / clean
                if disk_path.exists() and disk_path.is_file():
                    try:
                        content = disk_path.read_text(encoding="utf-8", errors="replace")
                    except Exception as exc:
                        content = f"_[Error reading file: {exc}]_"

            if content is None:
                missing.append(raw_path)
                continue

            # Build anchor slug
            anchor = f"file-{i}-" + clean.lower().replace("/", "-").replace(".", "-").replace(" ", "-")
            filename = Path(clean).name
            ext = Path(clean).suffix.lower()

            toc_lines.append(f"{i}. [{filename}](#{anchor})")

            # Choose fence language
            lang = CODE_LANG_MAP.get(ext, "")
            # Files already converted from rich formats (pdf/docx etc.) are
            # stored as plain Markdown text — render inline rather than in a fence.
            is_rich_doc = ext in MARKITDOWN_EXTENSIONS

            section_lines = [
                f'<a id="{anchor}"></a>',
                f"## {i}. `{filename}`",
                f"> **Path:** `{clean}`",
                "",
            ]

            if is_rich_doc:
                # Emit as plain Markdown (already converted at ingestion)
                section_lines.append(content)
            elif lang:
                section_lines.append(f"```{lang}\n{content}\n```")
            else:
                section_lines.append(f"```\n{content}\n```")

            body_sections.append("\n".join(section_lines))

        # Nothing resolved
        if not body_sections:
            return (
                f"_None of the requested file paths could be resolved._\n\n"
                f"**Missing paths:** {', '.join(f'`{p}`' for p in missing)}"
            )

        parts: List[str] = []
        if include_toc:
            parts.append("\n".join(toc_lines))
            parts.append("\n---\n")
        parts.append("\n\n---\n\n".join(body_sections))

        if missing:
            parts.append(
                f"\n\n> **⚠️ Could not resolve {len(missing)} path(s):** "
                + ", ".join(f"`{p}`" for p in missing)
            )

        return "\n".join(parts)

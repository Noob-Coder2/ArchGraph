"""
Document Consolidator & Surgical AST Slicing Engine (Sandboxed)

Solves the 3 Core Context-Slicing Edge Cases:
1. Missing Imports/Globals -> `extract_file_preamble()` automatically prepends top-of-file
   imports, constants, and module-level setup to every surgical slice.
2. Missing Decorators -> `find_symbol_span()` includes `@router.post(...)`, `@tool`,
   `@field_validator`, and Rust/TS attributes above functions and classes.
3. Function-less Schema/Config Files -> `build_smart_skeleton()` preserves Pydantic/ORM
   class fields, Enums, and top-level constants/assignments so `schemas.py` & `models.py`
   are never blanked out.
"""
import ast
import io
import os
import re
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from markitdown import MarkItDown
    _MARKITDOWN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MARKITDOWN_AVAILABLE = False
    logger.warning(
        "markitdown is not installed. Rich document conversion (.pdf, .docx, .pptx, .xlsx) "
        "will fall back to raw UTF-8 text extraction."
    )

MARKITDOWN_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}

CODE_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".html": "html",
    ".css": "css", ".json": "json", ".sql": "sql", ".java": "java",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".sh": "bash",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".xml": "xml",
}

SMALL_FILE_LINE_THRESHOLD = 380


class ASTSurgicalSlicer:
    """
    Provides deterministic AST symbol slicing, decorator preservation,
    import preamble extraction, and schema-aware code skeletons.
    """

    @staticmethod
    def extract_file_preamble(lines: List[str], ext: str = ".py", max_preamble_lines: int = 55) -> Tuple[str, int]:
        """
        Edge Case #1 Fix: Extracts top-of-file imports, module docstrings, and global
        constants so surgical function/class slices never lose their import context.
        Returns (formatted_numbered_preamble, last_preamble_line_number).
        """
        if not lines:
            return "", 0

        source = "\n".join(lines)
        if ext == ".py":
            try:
                tree = ast.parse(source)
                preamble_end = 0
                for node in tree.body:
                    # Include module docstring, imports, and top-level constants/type aliases
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        preamble_end = max(preamble_end, getattr(node, "end_lineno", node.lineno))
                    elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                        if preamble_end == 0:
                            preamble_end = getattr(node, "end_lineno", node.lineno)
                    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                        # Keep top-of-file constants/settings before the first def/class
                        end_ln = getattr(node, "end_lineno", node.lineno)
                        if end_ln <= max_preamble_lines:
                            preamble_end = max(preamble_end, end_ln)
                    elif isinstance(node, ast.If):
                        # Keep `if TYPE_CHECKING:` blocks
                        seg = ast.get_source_segment(source, node) or ""
                        if "TYPE_CHECKING" in seg:
                            preamble_end = max(preamble_end, getattr(node, "end_lineno", node.lineno))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        break

                preamble_end = min(preamble_end, max_preamble_lines)
                if preamble_end > 0:
                    numbered = [f"{i}: {lines[i - 1][:300]}" for i in range(1, preamble_end + 1)]
                    return "\n".join(numbered), preamble_end
            except SyntaxError:
                pass

        # Multi-language / fallback import & header scanner
        collected: List[str] = []
        last_ln = 0
        for i, raw_ln in enumerate(lines[:max_preamble_lines], start=1):
            stripped = raw_ln.strip()
            if (
                stripped.startswith(("import ", "from ", "use ", "package ", "#include", "const ", "export type ", "type "))
                or i <= 15
            ):
                collected.append(f"{i}: {raw_ln[:300]}")
                last_ln = i
            elif collected and not stripped:
                collected.append(f"{i}: ")
            elif len(collected) > 8:
                break

        return "\n".join(collected).strip(), last_ln

    @staticmethod
    def find_symbol_span(source: str, symbol: str, ext: str = ".py") -> Optional[Tuple[int, int, str]]:
        """
        Edge Case #2 Fix: Finds exact (start_line, end_line, symbol_kind) for a function,
        class, method (e.g. `ClassName.method` or `method`), or global assignment,
        INCLUDING all preceding `@decorators`!
        """
        target = symbol.strip().split("::")[-1]
        class_hint, _, member_hint = target.partition(".")
        lines = source.splitlines()
        total = len(lines)

        if ext == ".py":
            try:
                tree = ast.parse(source)
                best_match: Optional[Tuple[int, int, str]] = None

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        # Decorator-inclusive start line!
                        decorators = getattr(node, "decorator_list", []) or []
                        start_ln = min([node.lineno] + [d.lineno for d in decorators])
                        end_ln = getattr(node, "end_lineno", node.lineno)
                        kind = "class" if isinstance(node, ast.ClassDef) else "function"

                        if member_hint and isinstance(node, ast.ClassDef) and node.name == class_hint:
                            for sub in node.body:
                                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == member_hint:
                                    sub_decs = getattr(sub, "decorator_list", []) or []
                                    s_ln = min([sub.lineno] + [d.lineno for d in sub_decs])
                                    e_ln = getattr(sub, "end_lineno", sub.lineno)
                                    return max(1, s_ln - 2), min(total, e_ln), "method"

                        if node.name == target or (member_hint and node.name == member_hint):
                            return start_ln, end_ln, kind
                        if node.name.lower() == target.lower() and best_match is None:
                            best_match = (start_ln, end_ln, kind)

                    # Also support slicing top-level constants/dicts/tables (Edge Case #3)
                    elif isinstance(node, ast.Assign):
                        for t in node.targets:
                            if isinstance(t, ast.Name) and t.id == target:
                                return node.lineno, getattr(node, "end_lineno", node.lineno), "constant"

                if best_match:
                    return best_match
            except SyntaxError:
                pass

        # Regex fallback for TS/JS/Go/Rust/Java or broken Python syntax
        escaped = re.escape(member_hint or target)
        pattern = re.compile(
            rf"^\s*(?:export\s+)?(?:public\s+|private\s+|async\s+)*(?:def|class|function|interface|type|const|let|fn|struct|enum)\s+{escaped}\b"
        )
        for idx, line in enumerate(lines):
            if pattern.search(line) or re.search(rf"\b{escaped}\s*[=(:{{]", line):
                start_ln = idx + 1
                # Walk backwards to include @decorators, #[attributes], or doc comments
                while start_ln > 1 and lines[start_ln - 2].strip().startswith(("@", "#[", "///", "/**")):
                    start_ln -= 1
                end_ln = min(total, start_ln + 110)
                return start_ln, end_ln, "symbol"

        return None

    @staticmethod
    def build_smart_skeleton(filepath: str, source: str) -> str:
        """
        Edge Case #3 Fix: Builds a compact, high-signal Structural Blueprint for large files (>380 lines).
        - Preserves imports and module constants.
        - Preserves ALL `@decorators` on routes, tools, and validators.
        - For schema/model classes (`BaseModel`, `Enum`, `dataclass`, ORM models), preserves ALL
          field definitions and type annotations while folding long method bodies!
        """
        lines = source.splitlines()
        total = len(lines)
        ext = Path(filepath).suffix.lower()

        if ext != ".py":
            # Multi-language structural outline fallback
            outline = [f"### Structural Blueprint of `{filepath}` ({total} lines)"]
            for i, ln in enumerate(lines, start=1):
                s = ln.strip()
                if (
                    i <= 35
                    or s.startswith(("import ", "export ", "class ", "interface ", "type ", "function ", "async function ", "fn ", "pub fn ", "struct ", "enum ", "@"))
                ):
                    outline.append(f"{i}: {ln[:250]}")
            outline.append(
                f"\n💡 [Large File Blueprint: Use `read_file(filepath='{filepath}', symbol='Name')` "
                f"or `start_line=...` to inspect a specific implementation body.]"
            )
            return "\n".join(outline)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # If syntax error, return first 250 lines
            body = "\n".join(f"{i}: {lines[i - 1][:300]}" for i in range(1, min(total, 250) + 1))
            return f"{filepath} (lines 1-{min(total, 250)} of {total})\n{body}"

        keep_lines: set[int] = set()

        def _mark_range(s: int, e: int):
            for ln_num in range(max(1, s), min(total, e) + 1):
                keep_lines.add(ln_num)

        def _mark_decorators_and_signature(node: ast.AST):
            decs = getattr(node, "decorator_list", []) or []
            start_ln = min([node.lineno] + [d.lineno for d in decs])
            # Find where the `def ...:` or `class ...:` signature ends
            body_nodes = getattr(node, "body", [])
            sig_end = body_nodes[0].lineno - 1 if body_nodes else getattr(node, "end_lineno", node.lineno)
            _mark_range(start_ln, max(node.lineno, sig_end))
            # Include 1-line or short docstring if present
            if (
                body_nodes
                and isinstance(body_nodes[0], ast.Expr)
                and isinstance(getattr(body_nodes[0], "value", None), ast.Constant)
            ):
                doc_end = getattr(body_nodes[0], "end_lineno", body_nodes[0].lineno)
                if doc_end - body_nodes[0].lineno <= 4:
                    _mark_range(body_nodes[0].lineno, doc_end)

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                _mark_range(node.lineno, getattr(node, "end_lineno", node.lineno))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                # Edge Case #3: Keep top-level constants, schemas, and config assignments (up to 15 lines each)
                end_ln = getattr(node, "end_lineno", node.lineno)
                if end_ln - node.lineno <= 15:
                    _mark_range(node.lineno, end_ln)
                else:
                    _mark_range(node.lineno, node.lineno + 3)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _mark_decorators_and_signature(node)
            elif isinstance(node, ast.ClassDef):
                _mark_decorators_and_signature(node)
                for item in node.body:
                    # Edge Case #3: Keep ALL class attributes / Pydantic fields / Enum items!
                    if isinstance(item, (ast.AnnAssign, ast.Assign)):
                        _mark_range(item.lineno, getattr(item, "end_lineno", item.lineno))
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _mark_decorators_and_signature(item)

        out_lines = [
            f"### AST Blueprint of `{filepath}` ({total} total lines — imports, decorators, schemas & signatures shown)"
        ]
        sorted_kept = sorted(keep_lines)
        prev_ln = 0
        for ln_num in sorted_kept:
            if prev_ln and ln_num > prev_ln + 1:
                folded_count = ln_num - prev_ln - 1
                out_lines.append(f"    ... [lines {prev_ln + 1}-{ln_num - 1} folded ({folded_count} lines) — pass symbol or start_line={prev_ln + 1} to expand]")
            out_lines.append(f"{ln_num}: {lines[ln_num - 1][:300]}")
            prev_ln = ln_num

        if prev_ln < total:
            out_lines.append(f"    ... [lines {prev_ln + 1}-{total} folded — pass start_line={prev_ln + 1} to expand]")

        out_lines.append(
            f"\n💡 To inspect any function/class body above in full, call "
            f"`read_file(filepath='{filepath}', symbol='<name>')` or `consolidate_files(['{filepath}::<name>'])`."
        )
        return "\n".join(out_lines)


class DocumentConsolidator:
    """Sandboxed file reader and surgical multi-file/symbol Markdown consolidator."""

    @classmethod
    def read_file_content(cls, file_obj: Any) -> Tuple[str, bool]:
        """Reads file content and returns (markdown_text, was_converted)."""
        filename = getattr(file_obj, "name", "unnamed")
        ext = Path(filename).suffix.lower()

        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        if ext in MARKITDOWN_EXTENSIONS and _MARKITDOWN_AVAILABLE:
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
            except Exception as exc:
                logger.warning("markitdown failed for %s: %s — falling back to UTF-8.", filename, exc)

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

    @staticmethod
    def _parse_target_spec(spec: str) -> Tuple[str, Optional[str], Optional[Tuple[int, int]]]:
        """
        Parses a target selector into (filepath, symbol_name, line_range):
        - "app/services/insights.py::generate_insight" -> ("app/services/insights.py", "generate_insight", None)
        - "app/services/insights.py:80-180"            -> ("app/services/insights.py", None, (80, 180))
        - "app/models/schemas.py"                      -> ("app/models/schemas.py", None, None)
        """
        raw = (spec or "").strip().strip("`'\"")
        if "::" in raw:
            fp, sym = raw.split("::", 1)
            return fp.strip(), sym.strip(), None
        m = re.match(r"^(.+?):(\d+)(?:-(\d+))?$", raw)
        if m:
            fp = m.group(1).strip()
            s_ln = int(m.group(2))
            e_ln = int(m.group(3)) if m.group(3) else s_ln + 160
            return fp, None, (s_ln, e_ln)
        return raw, None, None

    @classmethod
    def consolidate_files(
        cls,
        graph_engine: Any,
        file_paths: List[str],
        include_toc: bool = True,
        max_chars: int = 55000,
    ) -> str:
        """
        Reads and merges multiple indexed files OR surgical symbol/line slices into one
        structured Markdown payload (up to 55,000 chars / ~14,000 tokens).
        Supports:
          - Full path: `"app/models/schemas.py"`
          - Symbol slice: `"app/services/clustering.py::run_clustering_for_user"`
          - Line range: `"app/services/insights.py:85-210"`
        Automatically prepends file imports/globals and includes `@decorators`.
        """
        if not file_paths:
            return "_No file paths provided to consolidate._"

        file_contents_store = getattr(graph_engine, "file_contents", None)
        if not file_contents_store:
            return "_No indexed files available in this session._"

        known_keys = list(file_contents_store.keys())
        toc_lines: List[str] = ["# Consolidated Codebase Audit Bundle\n\n## Table of Contents\n"]
        body_sections: List[str] = []
        missing: List[str] = []

        per_target_cap = max(8000, max_chars // max(1, min(len(file_paths), 6)))

        for i, raw_spec in enumerate(file_paths[:10], start=1):
            raw_path, symbol_target, line_range = cls._parse_target_spec(raw_spec)
            clean = (raw_path or "").replace("\\", "/").strip().lstrip("/")
            content: Optional[str] = None

            if clean in file_contents_store:
                content = file_contents_store.get(clean)
            else:
                for stored_path in known_keys:
                    sp = stored_path.replace("\\", "/")
                    if sp.endswith("/" + clean) or sp == clean or clean.endswith("/" + sp):
                        content = file_contents_store.get(stored_path)
                        clean = stored_path
                        break

            if content is None:
                missing.append(raw_spec)
                continue

            ext = Path(clean).suffix.lower()
            lines = content.splitlines()
            total_lines = len(lines)
            lang = CODE_LANG_MAP.get(ext, "")
            is_rich_doc = ext in MARKITDOWN_EXTENSIONS

            # Determine how to render this file/target
            mode_label = f"Full File ({total_lines} lines)"
            if symbol_target:
                span = ASTSurgicalSlicer.find_symbol_span(content, symbol_target, ext)
                preamble, pre_end = ASTSurgicalSlicer.extract_file_preamble(lines, ext)
                if span:
                    s_ln, e_ln, kind = span
                    mode_label = f"Surgical {kind.title()} `{symbol_target}` (Lines {s_ln}-{e_ln} of {total_lines} + Imports)"
                    slice_body = "\n".join(f"{ln_no}: {lines[ln_no - 1][:350]}" for ln_no in range(s_ln, e_ln + 1))
                    if preamble and s_ln > pre_end + 1:
                        rendered_code = f"# --- [Module Imports & Globals: Lines 1-{pre_end}] ---\n{preamble}\n\n# --- [{kind.title()} `{symbol_target}`: Lines {s_ln}-{e_ln}] ---\n{slice_body}"
                    else:
                        rendered_code = slice_body
                else:
                    # Symbol not found -> fall back to smart skeleton so the agent sees all real symbol names
                    mode_label = f"AST Blueprint (`{symbol_target}` not found verbatim; showing file structure)"
                    rendered_code = ASTSurgicalSlicer.build_smart_skeleton(clean, content)
            elif line_range:
                s_ln = max(1, line_range[0])
                e_ln = min(total_lines, max(s_ln, line_range[1]))
                preamble, pre_end = ASTSurgicalSlicer.extract_file_preamble(lines, ext)
                mode_label = f"Lines {s_ln}-{e_ln} of {total_lines} (+ Imports)"
                slice_body = "\n".join(f"{ln_no}: {lines[ln_no - 1][:350]}" for ln_no in range(s_ln, e_ln + 1))
                if preamble and s_ln > pre_end + 1:
                    rendered_code = f"# --- [Module Imports & Globals: Lines 1-{pre_end}] ---\n{preamble}\n\n# --- [Requested Lines {s_ln}-{e_ln}] ---\n{slice_body}"
                else:
                    rendered_code = slice_body
            elif total_lines <= SMALL_FILE_LINE_THRESHOLD or is_rich_doc:
                # Small/medium file -> include 100% of numbered lines!
                rendered_code = "\n".join(f"{ln_no}: {lines[:350]}" for ln_no in range(1, total_lines + 1))
            else:
                # Large file (>380 lines) requested without a selector -> provide full AST Blueprint + first 180 lines
                mode_label = f"Large File Smart Blueprint ({total_lines} lines)"
                rendered_code = ASTSurgicalSlicer.build_smart_skeleton(clean, content)

            if len(rendered_code) > per_target_cap:
                rendered_code = (
                    rendered_code[:per_target_cap]
                    + f"\n... [truncated at {per_target_cap} chars; pass `{clean}::SymbolName` or `{clean}:start-end`]"
                )

            anchor = f"file-{i}-" + clean.lower().replace("/", "-").replace(".", "-")
            filename = Path(clean).name
            toc_lines.append(f"{i}. [`{clean}` — {mode_label}](#{anchor})")

            section_lines = [
                f'<a id="{anchor}"></a>',
                f"## {i}. `{filename}` ({mode_label})",
                f"> **Path:** `{clean}`",
                "",
            ]
            if is_rich_doc:
                section_lines.append(rendered_code)
            elif lang:
                section_lines.append(f"```{lang}\n{rendered_code}\n```")
            else:
                section_lines.append(f"```\n{rendered_code}\n```")

            body_sections.append("\n".join(section_lines))

        if not body_sections:
            return (
                f"_None of the requested file paths could be resolved in the indexed session._\n\n"
                f"**Missing paths:** {', '.join(f'`{p}`' for p in missing)}"
            )

        parts: List[str] = []
        if include_toc:
            parts.append("\n".join(toc_lines))
            parts.append("\n---\n")
        parts.append("\n\n---\n\n".join(body_sections))

        if missing:
            parts.append(
                f"\n\n> **Could not resolve {len(missing)} path(s):** "
                + ", ".join(f"`{p}`" for p in missing)
            )

        merged = "\n".join(parts)
        if len(merged) > max_chars:
            merged = merged[:max_chars] + f"\n\n... [Consolidated bundle capped at {max_chars} chars]"
        return merged
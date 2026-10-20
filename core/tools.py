"""
tools.py - High-Bandwidth AST-Aware Agent Tools for Codebase Reasoning.

Key Harness Upgrades:
1. Solves the 3 Slicing Edge Cases via `ASTSurgicalSlicer`:
   - Auto-pins top-of-file imports & constants to every symbol/line slice.
   - Includes all `@decorators` (`@router.post`, `@tool`, `@field_validator`) on symbol spans.
   - Preserves Pydantic/ORM schema fields & constants on function-less files (`schemas.py`).
2. Adaptive Goldilocks `read_file`:
   - Files <= 380 lines are read 100% in full in one call.
   - Files > 380 lines support `symbol="func_or_class"` for surgical extraction, or return a
     complete AST Blueprint (with exact `[L_start-L_end]` numbers) so zero tokens are wasted.
3. High-Bandwidth `consolidate_files` (`55,000` chars / ~14,000 tokens):
   - Accepts both whole files (`"app/models/schemas.py"`) and surgical selectors
     (`"app/services/clustering.py::run_clustering_for_user"` or `"app/services/insights.py:80-200"`).
"""
from __future__ import annotations

import asyncio
import inspect
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, tool
from services.consolidator import DocumentConsolidator, ASTSurgicalSlicer, SMALL_FILE_LINE_THRESHOLD

GRAPH_CHARS = 5000          # graph part of search_code
CHUNK_CHARS = 1500          # each semantic chunk
SEMANTIC_TOP_K = 6
SYMBOL_CHARS = 12000        # explain_symbol now includes the symbol's actual source code!
MAX_TOOL_CHARS = 16000      # hard cap for search_code overall
READ_CHARS = 36000          # ~9,000 tokens per read_file call
CONSOLIDATE_CHARS = 55000   # ~14,000 tokens for multi-file surgical consolidation
DEFAULT_READ_LINES = 400
MAX_READ_LINES = 600
LIST_LIMIT = 80
REPO_MAP_MAX_CHARS = 7500   # ~1.8k tokens injected into system prompt

_IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".next", ".tox",
}


def clip(text: Any, limit: int = MAX_TOOL_CHARS, hint: str = "") -> str:
    """Truncate at a line boundary and tell the model how to get more."""
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    cut = s[:limit]
    nl = cut.rfind("\n")
    if nl > limit * 0.6:
        cut = cut[:nl]
    tail = f" {hint}" if hint else ""
    return f"{cut}\n... [truncated {len(s) - len(cut)} chars]{tail}"


def _norm(path: str) -> str:
    p = (path or "").replace("\\", "/").strip().strip("`'\"")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def _accepts(fn: Any, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _doc_text(doc: Any) -> str:
    getter = getattr(doc, "get_content", None)
    if callable(getter):
        return str(getter() or "")
    return str(getattr(doc, "text", "") or "")


class _Repo:
    """Read-only sandboxed view over everything indexed in the current session."""

    def __init__(self, graph: Any, vector: Any, root: Optional[Any] = None):
        self.graph = graph
        self.vector = vector
        self.root = Path(root).resolve() if root else None

    def _graph_file_keys(self) -> Dict[str, str]:
        fc = getattr(self.graph, "file_contents", None)
        if fc is None or not hasattr(fc, "keys"):
            return {}
        return {_norm(k): k for k in fc.keys()}

    def _doc_files(self) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {}
        for d in (getattr(self.vector, "documents", None) or []):
            fp = (getattr(d, "metadata", None) or {}).get("filepath")
            if fp:
                out.setdefault(_norm(fp), []).append(d)
        return out

    def _walk_root(self) -> List[str]:
        if not self.root:
            return []
        found: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
            for fn in filenames:
                found.append(Path(dirpath, fn).relative_to(self.root).as_posix())
                if len(found) >= 5000:
                    return found
        return found

    def paths(self) -> List[str]:
        known = set(self._graph_file_keys().keys()) | set(self._doc_files().keys())
        if not known:
            known = set(self._walk_root())
        return sorted(p for p in known if p)

    def resolve(self, raw: str) -> Tuple[Optional[str], List[str]]:
        """Map a model-supplied path to an indexed path. -> (path, ambiguous_candidates)."""
        p = _norm(raw.split("::")[0])
        if not p:
            return None, []
        known = self.paths()
        if p in known:
            return p, []
        cands = [k for k in known if k.endswith("/" + p) or p.endswith("/" + k)]
        if len(cands) == 1:
            return cands[0], []
        if cands:
            return None, cands[:8]
        base = p.rsplit("/", 1)[-1]
        cands = [k for k in known if k.rsplit("/", 1)[-1] == base]
        if len(cands) == 1:
            return cands[0], []
        return None, cands[:8]

    def text(self, path: str) -> Optional[str]:
        key_map = self._graph_file_keys()
        if path in key_map:
            fc = getattr(self.graph, "file_contents", {})
            val = fc.get(key_map[path])
            if val is not None:
                return val
        docs = self._doc_files().get(path)
        if docs:
            docs = sorted(docs, key=lambda d: (getattr(d, "metadata", None) or {}).get("page_number") or 0)
            return "\n".join(_doc_text(d) for d in docs)
        if self.root:
            target = (self.root / path).resolve()
            if target.is_file() and target.is_relative_to(self.root):
                return target.read_text(encoding="utf-8", errors="ignore")
        return None


def _format_chunks(results: Any) -> Tuple[str, List[str]]:
    out = []
    discovered_files: List[str] = []
    for i, r in enumerate(list(results)[:SEMANTIC_TOP_K], 1):
        fp = r.get("filepath", "?")
        if fp and fp != "?" and fp not in discovered_files:
            discovered_files.append(fp)
        s, e = r.get("start_line"), r.get("end_line")
        if s:
            loc = f":{s}-{e}" if e else f":{s}"
        elif r.get("file_type") == "pdf" and r.get("page_number"):
            loc = f" (page {r['page_number']})"
        else:
            loc = ""
        score = r.get("score")
        sc = f" score={score:.2f}" if isinstance(score, (int, float)) else ""
        out.append(f"[{i}] `{fp}{loc}`{sc}\n{clip(r.get('content', ''), CHUNK_CHARS)}")
    return "\n\n".join(out), discovered_files


def build_repo_map(
    graph_engine: Any,
    vector_engine: Any,
    root: Optional[Any] = None,
    max_chars: int = REPO_MAP_MAX_CHARS
) -> str:
    """Builds a compact, cached repository overview injected into the System Prompt."""
    repo = _Repo(graph_engine, vector_engine, root)
    paths = repo.paths()
    lines: List[str] = []
    if paths:
        exts = Counter(Path(p).suffix.lower() or "(no ext)" for p in paths)
        top = Counter(p.split("/", 1)[0] if "/" in p else "(root)" for p in paths)
        lines.append(f"Indexed files: {len(paths)}")
        lines.append("File types: " + ", ".join(f"{e} x{c}" for e, c in exts.most_common(6)))
        lines.append("Top-level entries: " + ", ".join(f"{d} ({c})" for d, c in top.most_common(12)))

    summary = ""
    fn = getattr(graph_engine, "get_architectural_summary", None)
    if callable(fn):
        try:
            summary = str(fn() or "")
        except Exception:
            summary = ""
    if summary:
        room = max(600, max_chars - sum(len(x) + 1 for x in lines) - 40)
        lines += ["", clip(summary, room, "(use search_code / explain_symbol for detail)")]

    return "\n".join(lines).strip() or "(no repository overview available)"


def create_codebase_tools(
    graph_engine: Any,
    vector_engine: Any,
    repo_root: Optional[Any] = None
) -> List[BaseTool]:
    """Creates the 6 high-bandwidth, AST-aware tools bound to one session's engines."""
    graph, vector = graph_engine, vector_engine
    repo = _Repo(graph, vector, repo_root)

    async def _run(fn: Any, *args: Any, **kwargs: Any) -> Tuple[Any, Optional[str]]:
        try:
            return await asyncio.to_thread(fn, *args, **kwargs), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    @tool
    async def search_code(query: str, follow_call_flow: bool = False) -> str:
        """Find relevant code, docs, symbols, and call-graph neighbors for a topic.
        START HERE when you do not know exact symbol or file names. Runs AST graph
        traversal and semantic vector search concurrently and returns both with exact
        file paths and line numbers.
        """
        async def graph_part() -> Tuple[Any, Optional[str]]:
            if graph is None:
                return None, None
            kw: Dict[str, Any] = {"mode": "dfs" if follow_call_flow else "bfs"}
            if _accepts(graph.query_traversal, "budget"):
                kw["budget"] = 2000
            return await _run(graph.query_traversal, query, **kw)

        async def vector_part() -> Tuple[Any, Optional[str]]:
            if vector is None:
                return None, None
            return await _run(vector.retrieve, query, top_k=SEMANTIC_TOP_K, file_type=None)

        (g_out, g_err), (v_out, v_err) = await asyncio.gather(graph_part(), vector_part())

        parts: List[str] = []
        if g_out:
            parts.append("## Graph Context & Connected Symbols\n" + clip(g_out, GRAPH_CHARS))
        if v_out:
            formatted_chunks, top_files = _format_chunks(v_out)
            parts.append("## Semantic Code & Doc Matches\n" + formatted_chunks)
            if len(top_files) >= 2:
                sample_list = ", ".join(f'"{fp}"' for fp in top_files[:4])
                parts.append(
                    f"💡 **Next Step Tip:** To inspect these related files together in one call, "
                    f"use `consolidate_files([{sample_list}])` (or `filepath::SymbolName` for huge files)."
                )

        notes = [f"{name} search failed: {err}" for name, err in (("graph", g_err), ("semantic", v_err)) if err]
        if not parts:
            msg = "No results. Try different keywords, an exact symbol name, or list_files()."
            return msg + (" (" + "; ".join(notes) + ")" if notes else "")
        if notes:
            parts.append("Note: " + "; ".join(notes))
        return clip("\n\n".join(parts), MAX_TOOL_CHARS)

    @tool
    async def explain_symbol(symbol: str) -> str:
        """Audits a specific class, function, method, or God Node from the code graph:
        returns its callers, callees, degree centrality, AND automatically extracts its
        exact source code implementation (including @decorators and file imports)!
        """
        if graph is None:
            return "No code graph is available for this session."

        def _do_explain() -> str:
            audit_text = graph.explain_symbol(symbol)
            # Also try to locate the file and slice the exact AST source code of this symbol!
            for marker in ("- **File**: `", "in `"):
                if marker in audit_text:
                    fp = audit_text.split(marker, 1)[1].split("`", 1)[0].strip()
                    resolved_fp, _ = repo.resolve(fp)
                    if resolved_fp:
                        src_text = repo.text(resolved_fp)
                        if src_text:
                            ext = Path(resolved_fp).suffix.lower()
                            span = ASTSurgicalSlicer.find_symbol_span(src_text, symbol, ext)
                            if span:
                                s_ln, e_ln, kind = span
                                lines = src_text.splitlines()
                                preamble, pre_end = ASTSurgicalSlicer.extract_file_preamble(lines, ext)
                                body = "\n".join(f"{i}: {lines[i - 1][:350]}" for i in range(s_ln, e_ln + 1))
                                code_block = (
                                    f"\n\n### Source Implementation (`{resolved_fp}` Lines {s_ln}-{e_ln})\n"
                                    f"```python\n"
                                    + (f"# --- [Imports & Globals: Lines 1-{pre_end}] ---\n{preamble}\n\n" if preamble and s_ln > pre_end + 1 else "")
                                    + f"# --- [{kind.title()} `{symbol}`: Lines {s_ln}-{e_ln}] ---\n{body}\n```"
                                )
                                return audit_text + code_block
            return audit_text

        out, err = await _run(_do_explain)
        if err:
            return f"explain_symbol failed: {err}"
        if not out:
            return f"'{symbol}' was not found in the graph. Use search_code to find the exact name."
        return clip(out, SYMBOL_CHARS, "(use read_file for surrounding functions)")

    @tool
    async def trace_path(source_symbol: str, target_symbol: str) -> str:
        """Shortest dependency or call-chain path between two known symbols, classes or
        modules ("how does A reach B?"). Use exact names.
        """
        if graph is None:
            return "No code graph is available for this session."
        out, err = await _run(graph.find_path, source_symbol, target_symbol)
        if err:
            return f"trace_path failed: {err}"
        if not out:
            return f"No path found between '{source_symbol}' and '{target_symbol}'."
        return clip(out, SYMBOL_CHARS)

    @tool
    async def read_file(
        filepath: str,
        symbol: Optional[str] = None,
        start_line: Optional[int] = None,
        line_count: Optional[int] = DEFAULT_READ_LINES
    ) -> str:
        """Smart AST-aware file reader.
        - If the file is <= 380 lines (and no start_line/symbol is given), returns 100% of the file!
        - If `symbol` is provided (or `filepath` is written as `path.py::SymbolName`), surgically
          extracts that exact function/class/schema WITH its @decorators AND the file's top imports!
        - If the file is > 380 lines and neither `symbol` nor `start_line` is given, returns the
          file's complete AST Structural Blueprint (all imports, decorators, Pydantic/ORM fields,
          and function signatures with exact line numbers) so you can inspect huge files without
          wasting tokens.
        """
        def _do() -> str:
            inline_sym = None
            raw_fp = filepath
            if "::" in filepath:
                raw_fp, inline_sym = filepath.split("::", 1)
            target_sym = (symbol or inline_sym or "").strip() or None

            path, cands = repo.resolve(raw_fp)
            if path is None:
                if cands:
                    return f"'{raw_fp}' is ambiguous. Candidates:\n" + "\n".join(f"- {c}" for c in cands)
                return f"'{raw_fp}' is not in the indexed repository. Use list_files() to browse."
            text = repo.text(path)
            if text is None:
                return f"'{path}' is indexed but its text is unavailable."

            lines = text.splitlines()
            total = len(lines)
            ext = Path(path).suffix.lower()

            # Case 1: Surgical Symbol Extraction (with @decorators + file imports preamble)
            if target_sym:
                span = ASTSurgicalSlicer.find_symbol_span(text, target_sym, ext)
                if span:
                    s_ln, e_ln, kind = span
                    preamble, pre_end = ASTSurgicalSlicer.extract_file_preamble(lines, ext)
                    body = "\n".join(f"{i}: {lines[i - 1][:380]}" for i in range(s_ln, e_ln + 1))
                    header = f"{path} :: {kind} `{target_sym}` (lines {s_ln}-{e_ln} of {total})\n"
                    if preamble and s_ln > pre_end + 1:
                        return (
                            f"{header}# --- [Module Imports & Globals: Lines 1-{pre_end}] ---\n{preamble}\n\n"
                            f"# --- [{kind.title()} `{target_sym}`: Lines {s_ln}-{e_ln}] ---\n{body}"
                        )
                    return header + body
                # If symbol wasn't matched verbatim, return the smart blueprint so the model sees all real symbols
                return (
                    f"Symbol `{target_sym}` was not found verbatim in `{path}`. "
                    f"Here is the AST Blueprint of `{path}` so you can see all exact symbol names and line numbers:\n\n"
                    + ASTSurgicalSlicer.build_smart_skeleton(path, text)
                )

            # Case 2: Explicit line range requested
            if start_line is not None and start_line > 1:
                s = max(1, int(start_line))
                n = min(max(1, int(line_count or DEFAULT_READ_LINES)), MAX_READ_LINES)
                if s > total:
                    return f"'{path}' has only {total} lines."
                e = min(total, s + n - 1)
                preamble, pre_end = ASTSurgicalSlicer.extract_file_preamble(lines, ext)
                body = "\n".join(f"{i}: {ln[:380]}" for i, ln in enumerate(lines[s - 1:e], start=s))
                more = f"\n[more: call again with start_line={e + 1}]" if e < total else ""
                if preamble and s > pre_end + 1:
                    return (
                        f"{path} (lines {s}-{e} of {total}, with top-of-file imports)\n"
                        f"# --- [Module Imports & Globals: Lines 1-{pre_end}] ---\n{preamble}\n\n"
                        f"# --- [Lines {s}-{e}] ---\n{body}{more}"
                    )
                return f"{path} (lines {s}-{e} of {total})\n{body}{more}"

            # Case 3: Small or medium file (<= 380 lines) -> Return 100% of the file in one shot!
            if total <= SMALL_FILE_LINE_THRESHOLD:
                body = "\n".join(f"{i}: {ln[:380]}" for i, ln in enumerate(lines, start=1))
                return f"{path} (full file: lines 1-{total} of {total})\n{body}"

            # Case 4: Large file (> 380 lines) with no symbol or start_line -> Return AST Blueprint!
            if start_line == 1:
                e = min(total, int(line_count or DEFAULT_READ_LINES))
                body = "\n".join(f"{i}: {ln[:380]}" for i, ln in enumerate(lines[:e], start=1))
                return f"{path} (lines 1-{e} of {total})\n{body}\n[more: pass start_line={e + 1} or symbol='FunctionName']"

            return ASTSurgicalSlicer.build_smart_skeleton(path, text)

        out, err = await _run(_do)
        if err:
            return f"read_file failed: {err}"
        return clip(out, READ_CHARS, "(use symbol='...' or start_line=... for specific sections)")

    @tool
    async def list_files(path_prefix: str = "") -> str:
        """List ONE directory level of the indexed repository: sub-folders with file
        counts, then files. Start with '' for the root, then drill down (e.g. 'app/services').
        """
        def _do() -> str:
            prefix = _norm(path_prefix).rstrip("/")
            base = prefix + "/" if prefix else ""
            dirs: Counter = Counter()
            files: List[str] = []
            for p in repo.paths():
                if not p.startswith(base):
                    continue
                rest = p[len(base):]
                if "/" in rest:
                    dirs[rest.split("/", 1)[0]] += 1
                else:
                    files.append(rest)
            if not dirs and not files:
                return f"Nothing indexed under '{prefix}'. Try list_files('') for the root."
            lines = [f"Contents of '{prefix or '/'}':"]
            for name, count in sorted(dirs.items())[:LIST_LIMIT]:
                lines.append(f"- {name}/  ({count} files)")
            for name in sorted(files)[:LIST_LIMIT]:
                lines.append(f"- {name}")
            hidden = max(0, len(dirs) - LIST_LIMIT) + max(0, len(files) - LIST_LIMIT)
            if hidden:
                lines.append(f"... and {hidden} more; narrow with a deeper path_prefix.")
            return "\n".join(lines)

        out, err = await _run(_do)
        return f"list_files failed: {err}" if err else out

    @tool
    async def consolidate_files(file_paths: List[str], include_toc: bool = True) -> str:
        """High-bandwidth multi-file & surgical-symbol bundler (up to 55,000 chars / ~14k tokens).
        Reads and merges 2 to 8 related files OR specific functions/classes in ONE round-trip!
        Each item in `file_paths` can be:
          - A full file path: `"app/models/schemas.py"` (reads full file if <=380 lines, or AST Blueprint if huge)
          - A surgical symbol selector: `"app/services/clustering.py::run_clustering_for_user"` (extracts that
            exact function/class WITH its @decorators AND the file's top imports!)
          - A surgical line range: `"app/services/insights.py:80-210"` (extracts lines 80-210 + top imports!)
        """
        out, err = await _run(
            DocumentConsolidator.consolidate_files,
            graph,
            file_paths,
            include_toc,
            CONSOLIDATE_CHARS
        )
        if err:
            return f"consolidate_files failed: {err}"
        return clip(out, CONSOLIDATE_CHARS)

    return [search_code, explain_symbol, trace_path, read_file, list_files, consolidate_files]
"""
tools.py - Agent tools for codebase reasoning (single source of truth).

Design rules (each one fixes a problem in the previous version):

1. FEW tools with distinct jobs. A small-active-parameter model picks tools far
   more reliably from 5 clear ones than from 8 overlapping ones, and every tool
   schema is re-sent on every LLM call.
2. Every output is BOUNDED. A tool result stays in the context for the rest of
   the turn and is re-processed by every later LLM call, so unbounded results
   are the main source of slowness and context blow-up.
3. `read_file` is SANDBOXED to indexed files (plus an optional, path-checked
   repo_root). The model can be steered by text inside an uploaded repo, so it
   must never be able to read `.env`, `../../etc/passwd`, or your app's source.
4. Tools NEVER raise. Errors come back as short strings the model can act on.
5. `search_code` runs graph traversal and semantic retrieval CONCURRENTLY in one
   call: one LLM round trip instead of two or three.

Engine contract (unchanged from your current code):
    graph.file_contents                     dict[str, str]        (optional)
    graph.get_architectural_summary()       -> str
    graph.query_traversal(question, mode=)  -> str   ('bfs' | 'dfs'; may accept budget=)
    graph.find_path(source, target)         -> str
    graph.explain_symbol(symbol)            -> str
    vector.retrieve(query, top_k=, file_type=) -> list[dict]
        dict keys: filepath, content, score, file_type, page_number
        optional : start_line, end_line  (add these at index time -> real line citations)
    vector.documents                        list of LlamaIndex Documents (optional)
"""
from __future__ import annotations

import asyncio
import inspect
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, tool

# --------------------------------------------------------------------------- #
# Budgets (characters; ~4 chars ~= 1 token)
# --------------------------------------------------------------------------- #
GRAPH_CHARS = 3000          # graph part of search_code
CHUNK_CHARS = 800           # each semantic chunk
SEMANTIC_TOP_K = 5
SYMBOL_CHARS = 3500         # explain_symbol / trace_path
MAX_TOOL_CHARS = 8000       # hard cap for search_code overall
READ_CHARS = 12000          # hard cap for read_file (it is the verification tool)
DEFAULT_READ_LINES = 120
MAX_READ_LINES = 250
LIST_LIMIT = 60
REPO_MAP_MAX_CHARS = 6000   # ~1.5k tokens, goes into the system prompt once

_IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".next", ".tox",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
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
    """Read-only view over everything that is indexed (graph files + vector docs)."""

    def __init__(self, graph: Any, vector: Any, root: Optional[Any] = None):
        self.graph = graph
        self.vector = vector
        self.root = Path(root).resolve() if root else None

    def _graph_files(self) -> Dict[str, str]:
        fc = getattr(self.graph, "file_contents", None)
        return {_norm(k): v for k, v in fc.items()} if isinstance(fc, dict) else {}

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
        known = set(self._graph_files()) | set(self._doc_files())
        if not known:
            known = set(self._walk_root())
        return sorted(p for p in known if p)

    def resolve(self, raw: str) -> Tuple[Optional[str], List[str]]:
        """Map a model-supplied path to an indexed path. -> (path, ambiguous_candidates)."""
        p = _norm(raw)
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
        gf = self._graph_files()
        if path in gf:
            return gf[path]
        docs = self._doc_files().get(path)
        if docs:
            docs = sorted(docs, key=lambda d: (getattr(d, "metadata", None) or {}).get("page_number") or 0)
            return "\n".join(_doc_text(d) for d in docs)
        if self.root:  # sandboxed disk fallback for indexed-but-not-in-memory files
            target = (self.root / path).resolve()
            if target.is_file() and target.is_relative_to(self.root):
                return target.read_text(encoding="utf-8", errors="ignore")
        return None


def _format_chunks(results: Any) -> str:
    out = []
    for i, r in enumerate(list(results)[:SEMANTIC_TOP_K], 1):
        fp = r.get("filepath", "?")
        s, e = r.get("start_line"), r.get("end_line")
        if s:
            loc = f":{s}-{e}" if e else f":{s}"
        elif r.get("file_type") == "pdf" and r.get("page_number"):
            loc = f" (page {r['page_number']})"
        else:
            loc = ""
        score = r.get("score")
        sc = f" score={score:.2f}" if isinstance(score, (int, float)) else ""
        out.append(f"[{i}] {fp}{loc}{sc}\n{clip(r.get('content', ''), CHUNK_CHARS)}")
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# Repo map: injected into the system prompt ONCE (replaces the old
# "you MUST call read_codebase_context first" round trip).
# --------------------------------------------------------------------------- #
def build_repo_map(graph_engine: Any, vector_engine: Any, root: Optional[Any] = None,
                   max_chars: int = REPO_MAP_MAX_CHARS) -> str:
    repo = _Repo(graph_engine, vector_engine, root)
    paths = repo.paths()
    lines: List[str] = []
    if paths:
        exts = Counter(Path(p).suffix.lower() or "(no ext)" for p in paths)
        top = Counter(p.split("/", 1)[0] if "/" in p else "(root)" for p in paths)
        lines.append(f"Indexed files: {len(paths)}")
        lines.append("File types: " + ", ".join(f"{e} x{c}" for e, c in exts.most_common(6)))
        lines.append("Top-level entries: " + ", ".join(f"{d} ({c})" for d, c in top.most_common(10)))

    summary = ""
    fn = getattr(graph_engine, "get_architectural_summary", None)
    if callable(fn):
        try:
            summary = str(fn() or "")
        except Exception:
            summary = ""
    if summary:
        room = max(500, max_chars - sum(len(x) + 1 for x in lines) - 40)
        lines += ["", clip(summary, room, "(use search_code / explain_symbol for detail)")]
    return "\n".join(lines).strip() or "(no repository overview available)"


# --------------------------------------------------------------------------- #
# Tool factory
# --------------------------------------------------------------------------- #
def create_codebase_tools(graph_engine: Any, vector_engine: Any,
                          repo_root: Optional[Any] = None) -> List[BaseTool]:
    """Create the 5 tools bound to one session's engines.

    repo_root: optional directory used ONLY for (a) listing when nothing is indexed and
    (b) reading indexed files whose text is not held in memory. Reads are confined to it.
    """
    graph, vector = graph_engine, vector_engine
    repo = _Repo(graph, vector, repo_root)

    async def _run(fn: Any, *args: Any, **kwargs: Any) -> Tuple[Any, Optional[str]]:
        try:
            return await asyncio.to_thread(fn, *args, **kwargs), None
        except Exception as exc:  # tools must never raise into the agent loop
            return None, f"{type(exc).__name__}: {exc}"

    @tool
    async def search_code(query: str, follow_call_flow: bool = False) -> str:
        """Find relevant code, docs and structure for a topic. START HERE when you do not
        know exact symbol or file names. Runs the structural code graph and semantic
        search together and returns both.

        Args:
            query: What to look for, e.g. "how are API keys validated" or "PaymentService".
            follow_call_flow: False (default) = broad architectural context around the topic.
                True = follow execution/call chains ("what happens when X runs").
        """
        async def graph_part() -> Tuple[Any, Optional[str]]:
            if graph is None:
                return None, None
            kw: Dict[str, Any] = {"mode": "dfs" if follow_call_flow else "bfs"}
            if _accepts(graph.query_traversal, "budget"):
                kw["budget"] = 1500  # Graphify-style token budget, if your wrapper exposes it
            return await _run(graph.query_traversal, query, **kw)

        async def vector_part() -> Tuple[Any, Optional[str]]:
            if vector is None:
                return None, None
            return await _run(vector.retrieve, query, top_k=SEMANTIC_TOP_K, file_type=None)

        (g_out, g_err), (v_out, v_err) = await asyncio.gather(graph_part(), vector_part())

        parts: List[str] = []
        if g_out:
            parts.append("## Graph context\n" + clip(g_out, GRAPH_CHARS,
                         "(use explain_symbol on one node for more)"))
        if v_out:
            parts.append("## Semantic matches\n" + _format_chunks(v_out))
        notes = [f"{name} search failed: {err}" for name, err in (("graph", g_err), ("semantic", v_err)) if err]
        if not parts:
            msg = "No results. Try different keywords, an exact symbol name, or list_files()."
            return msg + (" (" + "; ".join(notes) + ")" if notes else "")
        if notes:
            parts.append("Note: " + "; ".join(notes))
        return clip("\n\n".join(parts), MAX_TOOL_CHARS)

    @tool
    async def explain_symbol(symbol: str) -> str:
        """Structural profile of ONE known class, function or module from the code graph:
        callers, callees, dependencies, community and how central it is. Use the exact
        name; if unsure, call search_code first.
        """
        if graph is None:
            return "No code graph is available for this session."
        out, err = await _run(graph.explain_symbol, symbol)
        if err:
            return f"explain_symbol failed: {err}"
        if not out:
            return f"'{symbol}' was not found in the graph. Use search_code to find the exact name."
        return clip(out, SYMBOL_CHARS, "(ask about a specific caller/callee for more)")

    @tool
    async def trace_path(source_symbol: str, target_symbol: str) -> str:
        """Shortest dependency or call-chain path between two known symbols, classes or
        modules ("how does A reach B?"). Use exact names."""
        if graph is None:
            return "No code graph is available for this session."
        out, err = await _run(graph.find_path, source_symbol, target_symbol)
        if err:
            return f"trace_path failed: {err}"
        if not out:
            return f"No path found between '{source_symbol}' and '{target_symbol}'."
        return clip(out, SYMBOL_CHARS)

    @tool
    async def read_file(filepath: str, start_line: Optional[int] = 1,
                        line_count: Optional[int] = DEFAULT_READ_LINES) -> str:
        """Read numbered lines of an indexed file (source, markdown or document text).
        Use it to VERIFY a claim and to get exact line numbers before citing them.
        Max 250 lines per call; continue with start_line.

        Args:
            filepath: Path as shown by search_code / list_files.
            start_line: 1-indexed first line.
            line_count: Number of lines to read (default 120, max 250).
        """
        def _do() -> str:
            path, cands = repo.resolve(filepath)
            if path is None:
                if cands:
                    return f"'{filepath}' is ambiguous. Candidates:\n" + "\n".join(f"- {c}" for c in cands)
                return f"'{filepath}' is not in the indexed repository. Use list_files() to browse."
            text = repo.text(path)
            if text is None:
                return f"'{path}' is indexed but its text is unavailable."
            lines = text.splitlines()
            total = len(lines)
            s = max(1, int(start_line or 1))
            n = min(max(1, int(line_count or DEFAULT_READ_LINES)), MAX_READ_LINES)
            if s > total:
                return f"'{path}' has only {total} lines."
            e = min(total, s + n - 1)
            body = "\n".join(f"{i}: {ln[:400]}" for i, ln in enumerate(lines[s - 1:e], start=s))
            more = f"\n[more: call again with start_line={e + 1}]" if e < total else ""
            return f"{path} (lines {s}-{e} of {total})\n{body}{more}"

        out, err = await _run(_do)
        if err:
            return f"read_file failed: {err}"
        return clip(out, READ_CHARS, "(reduce line_count)")

    @tool
    async def list_files(path_prefix: str = "") -> str:
        """List ONE directory level of the indexed repository: sub-folders with file
        counts, then files. Start with '' for the root, then drill down (e.g. 'src/api').
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

    return [search_code, explain_symbol, trace_path, read_file, list_files]
    
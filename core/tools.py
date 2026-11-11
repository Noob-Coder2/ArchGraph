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
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from langchain_core.tools import BaseTool, tool
from services.consolidator import DocumentConsolidator, ASTSurgicalSlicer, SMALL_FILE_LINE_THRESHOLD
from services.storage import storage

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
        START HERE when you do not know exact symbol or file names. Runs a hybrid
        fusion (FastEmbed vectors + lexical AST matching + Personalized PageRank
        over the Graphify AST graph via Reciprocal Rank Fusion) and returns ranked
        results with exact file paths and line numbers.
        """
        from core.hybrid_retriever import HybridFusionRetriever

        retriever = HybridFusionRetriever(graph, vector)
        fused, err = await _run(retriever.retrieve, query, top_k=SEMANTIC_TOP_K)
        if err:
            return f"search_code failed: {err}"
        if not fused:
            return "## Hybrid Search Results\nNo results."

        parts: List[str] = []
        results = fused.get("results", [])
        if results:
            lines: List[str] = [
                "## Hybrid Search Results (vector + lexical AST + graph PPR, RRF fused)"
            ]
            for i, r in enumerate(results, start=1):
                channel = r.get("channel", "?")
                rtype = r.get("type", "?")
                fp = r.get("filepath", "unknown")
                sym = r.get("symbol") or ""
                s = r.get("start_line")
                e = r.get("end_line")
                loc = f":{s}-{e}" if s and e else (f":{s}" if s else "")
                rrf = r.get("rrf_score", 0.0)
                content = clip(r.get("content", ""), CHUNK_CHARS)
                head = f"[{i}] `{fp}{loc}`"
                if sym:
                    head += f"  ({rtype}: `{sym}`)"
                head += f"  [channel={channel} rrf={rrf:.4f}]"
                lines.append(f"{head}\n{content}")
            parts.append("\n".join(lines))
        else:
            parts.append("## Hybrid Search Results\nNo results.")

        if graph is None:
            parts.append("Note: graph search unavailable (no GraphifyEngine bound).")
        if vector is None:
            parts.append("Note: semantic search unavailable (no LlamaIndexEngine bound).")
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

    # ------------------------------------------------------------------
    # Tool 7: audit_blast_radius — "If I change X, what breaks?"
    # ------------------------------------------------------------------
    @tool
    async def audit_blast_radius(
        symbol: str,
        max_hops: int = 3,
        filepath: Optional[str] = None,
        top_k_per_hop: int = 10,
    ) -> str:
        """
        Upstream impact analysis: if symbol `symbol` changes, which callers
        (direct and transitive) are affected?

        - Multi-seed resolution: finds ALL matching definitions across the
          codebase (or narrows to one file via `filepath::Symbol`).
        - Upstream ripple traversal: reverses the DiGraph and runs BFS from
          each seed, walking caller -> callee backwards (Hop 1 = direct
          callers, Hop 2 = callers of callers, etc.).
        - Chain reconstruction: for every impacted caller, prints the exact
          breadcrumb trail of relationships back to the changed symbol.
        - Hybrid relevance ranking: callers at each hop are ranked by
          in-degree centrality (how many depend on them) so critical
          business logic surfaces first.
        - Circular dependency warning: flags if the symbol sits inside an
          import/call cycle.
        """
        if graph is None:
            return "No code graph is available for this session."

        g: nx.DiGraph = graph.graph
        if g.number_of_nodes() == 0:
            return "Code graph is empty. Please ingest a codebase first."

        # ------------------------------------------------------------------
        # 1. Multi-seed symbol resolution
        # ------------------------------------------------------------------
        raw = symbol.strip()
        # Split "filepath::Symbol" or "ClassName.method" or just "symbol"
        if "::" in raw:
            filepath_hint, target = raw.split("::", 1)
        else:
            filepath_hint, target = None, raw

        # target may be "ClassName.method" — split into parts
        target_parts = [p for p in target.split(".") if p]
        target_last = target_parts[-1] if target_parts else target
        target_low = target_last.lower()

        # Effective filepath filter: explicit arg overrides :: hint
        eff_filepath = filepath if filepath is not None else filepath_hint

        seeds: list[str] = []
        for n, data in g.nodes(data=True):
            n_fp = str(data.get("filepath") or data.get("source_file", ""))
            n_label = str(data.get("label", n))
            n_id = str(n)

            # Normalize label: strip leading '.', trailing '()', e.g. ".authenticate()" -> "authenticate"
            norm_label = n_label.lstrip(".").rstrip("()").rstrip(")")
            norm_label_low = norm_label.lower()

            # Match if any target part matches the label or the last node-id segment
            matched = False
            if norm_label_low == target_low:
                matched = True
            elif target_parts and len(target_parts) > 1:
                # ClassName.method: label might be just the method name
                if norm_label_low == target_parts[-1].lower():
                    matched = True
            # Node ID uses underscores: services_auth_authservice_authenticate
            id_parts = n_id.split("_")
            if target_last.lower() in [p.lower() for p in id_parts[-3:]]:
                matched = True

            if matched:
                # Filepath filter
                if eff_filepath is not None:
                    if n_fp == eff_filepath or n_fp.endswith(eff_filepath) or eff_filepath in n_fp:
                        seeds.append(n_id)
                else:
                    seeds.append(n_id)

        # Deduplicate while preserving order
        seen: set[str] = set()
        seeds = [s for s in seeds if not (s in seen or seen.add(s))]

        if not seeds:
            hint = f" in file '{filepath}'" if filepath else ""
            return (
                f"Symbol '{symbol}' was not found in the code graph{hint}. "
                "Use `search_code` or `list_files` to find the exact name."
            )

        # ------------------------------------------------------------------
        # 2. Upstream ripple traversal (reversed BFS)
        # ------------------------------------------------------------------
        # g is a DiGraph where edge A -> B means "A calls/depends on B".
        # To find who would break if B changes, walk predecessors:
        # Hop 1 = direct callers of seed, Hop 2 = callers of callers, etc.
        visited: dict[str, tuple[str | None, int | None, str | None]] = {}
        # node -> (parent_node, hop, relation)

        queue: list[tuple[str, int, str | None]] = [(s, 0, None) for s in seeds]
        for s in seeds:
            visited[s] = (None, 0, None)

        head = 0
        while head < len(queue):
            node, hop, rel = queue[head]
            head += 1
            if hop >= max_hops:
                continue
            for parent in g.predecessors(node):
                if parent not in visited:
                    edge_data = g.get_edge_data(parent, node) or {}
                    edge_rel = edge_data.get("relation", "calls")
                    visited[parent] = (node, hop + 1, edge_rel)
                    queue.append((parent, hop + 1, edge_rel))

        # ------------------------------------------------------------------
        # 3. Circular dependency warning
        # ------------------------------------------------------------------
        cycle_warning = ""
        try:
            cycle = nx.find_cycle(g, orientation="original")
            cycle_nodes = [u for u, _, _ in cycle]
            if any(s in cycle_nodes for s in seeds):
                cycle_warning = (
                    "\n\n⚠️  CIRCULAR DEPENDENCY DETECTED: "
                    + " -> ".join(str(n) for n in cycle_nodes + [cycle_nodes[0]])
                    + ". Changes inside this cycle propagate unpredictably."
                )
        except nx.NetworkXNoCycle:
            pass

        # ------------------------------------------------------------------
        # 4. Build chain reconstructions + rank per hop
        # ------------------------------------------------------------------
        def _reconstruct_chain(end_node: str) -> list[tuple[str, str, str, int]]:
            """Returns list of (node, label, relation, hop) from seed to end_node."""
            chain: list[tuple[str, str, str, int]] = []
            cur: str | None = end_node
            while cur is not None and cur in visited:
                parent, hop, rel = visited[cur]
                node_data = g.nodes.get(cur, {})
                label = str(node_data.get("label", cur))
                fp = str(node_data.get("filepath") or node_data.get("source_file", "unknown"))
                chain.append((f"{fp}::{label}", label, rel or "seed", hop or 0))
                cur = parent
            chain.reverse()
            return chain

        # Group impacted callers by hop
        hop_groups: dict[int, list[tuple[str, str, str, int, list]]] = {}
        for node, (parent, hop, rel) in visited.items():
            if hop == 0 or node in seeds:
                continue  # skip seeds themselves
            hop_int = hop if hop is not None else 0
            node_data = g.nodes.get(node, {})
            label = str(node_data.get("label", node))
            fp = str(node_data.get("filepath") or node_data.get("source_file", "unknown"))
            full_id = f"{fp}::{label}"
            chain = _reconstruct_chain(node)
            # Rank by in-degree (how many depend on this caller)
            rank_score = g.in_degree(node)
            hop_groups.setdefault(hop_int, []).append(
                (full_id, label, rel or "calls", rank_score, chain)
            )

        # Sort each hop by rank score (descending)
        for hop in hop_groups:
            hop_groups[hop].sort(key=lambda x: x[3], reverse=True)
            hop_groups[hop] = hop_groups[hop][:top_k_per_hop]

        # ------------------------------------------------------------------
        # 5. Format output
        # ------------------------------------------------------------------
        lines: list[str] = []
        seed_names = ", ".join(str(s) for s in seeds)
        lines.append(f"# Blast Radius: `{symbol}`")
        lines.append(f"Seeds: {seed_names}")
        lines.append(f"Max hops: {max_hops} | Total impacted callers: {len(visited) - len(seeds)}")
        if cycle_warning:
            lines.append(cycle_warning)

        if not hop_groups:
            lines.append("\nNo upstream callers found within the hop limit.")
            return "\n".join(lines)

        for hop in sorted(hop_groups.keys()):
            lines.append(f"\n## Hop {hop} — Direct {'callers' if hop == 1 else 'transitive callers'}")
            for full_id, label, rel, rank_score, chain in hop_groups[hop]:
                chain_str = " <- ".join(reversed([c[0] for c in chain]))
                lines.append(
                    f"  - `{full_id}` (dependents: {rank_score}, link: {rel})"
                )
                lines.append(f"      Chain: {chain_str}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 8: audit_against_blueprint — "Does code match planned architecture?"
    # ------------------------------------------------------------------
    @tool
    async def audit_against_blueprint(
        blueprint: str,
        action: str = "verify",
    ) -> str:
        """
        Two-stage hybrid architectural verification.

        Actions:
        - "discover": Returns macro-flow X-ray (module-to-module link counts,
          communities, circular import cycles) — no rules needed.
        - "store": `blueprint` is a JSON array of rules. Each rule:
            {rule_id, description, rule_type ("forbidden"|"required"),
             source_pattern, target_pattern}
          Rules are persisted in the session SQLite database.
        - "verify": Loads all stored rules and checks them deterministically
          against the AST graph. Forbidden rules flag any crossing edge as a
          violation (drift). Required rules verify at least one crossing edge
          exists. Evidence includes filepath:start_line-end_line citations.
        """
        if graph is None:
            return "No code graph is available for this session."

        g: nx.DiGraph = graph.graph
        session_id = graph.session_id

        if action == "discover":
            return _discover_macro_flows(g)

        if action == "store":
            try:
                rules = json.loads(blueprint)
            except json.JSONDecodeError as exc:
                return f"Invalid JSON blueprint: {exc}"

            if not isinstance(rules, list):
                return "Blueprint must be a JSON array of rule objects."

            stored = 0
            for rule in rules:
                rid = rule.get("rule_id", f"rule_{stored}")
                storage.save_arch_rule(
                    session_id=session_id,
                    rule_id=rid,
                    description=rule.get("description", ""),
                    rule_type=rule.get("rule_type", "forbidden"),
                    source_pattern=rule.get("source_pattern", ""),
                    target_pattern=rule.get("target_pattern", ""),
                )
                stored += 1
            return f"Stored {stored} architectural rule(s) for session {session_id[:8]}."

        if action == "verify":
            rules = storage.list_arch_rules(session_id)
            if not rules:
                return (
                    "No architectural rules stored yet. "
                    "Call with action='store' to persist rules first, "
                    "or action='discover' for a macro-flow X-ray."
                )

            return _verify_rules(g, rules, vector)

        return f"Unknown action '{action}'. Use 'discover', 'store', or 'verify'."

    def _match_pattern(path: str, pattern: str) -> bool:
        """Simple glob-style match: 'api/*' matches 'api/routes.py'."""
        if pattern == "*":
            return True
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            return path == prefix or path.startswith(prefix + "/")
        return path == pattern or path.startswith(pattern + "/") or path == pattern

    def _discover_macro_flows(g: nx.DiGraph) -> str:
        """Stage 1: module-level aggregation + communities + cycles."""
        from collections import Counter

        # Aggregate edges at top-level module level
        module_links: Counter[tuple[str, str]] = Counter()
        for u, v, data in g.edges(data=True):
            u_fp = str(g.nodes[u].get("filepath") or g.nodes[u].get("source_file", ""))
            v_fp = str(g.nodes[v].get("filepath") or g.nodes[v].get("source_file", ""))
            u_mod = u_fp.split("/", 1)[0] if "/" in u_fp else "(root)"
            v_mod = v_fp.split("/", 1)[0] if "/" in v_fp else "(root)"
            if u_mod != v_mod:
                module_links[(u_mod, v_mod)] += 1

        lines: list[str] = []
        lines.append("# Architectural Discovery (Macro Flows)")
        lines.append("## Module-to-Module Dependency Links")
        if module_links:
            for (src, tgt), count in module_links.most_common(20):
                lines.append(f"  {src} -> {tgt}: {count} link(s)")
        else:
            lines.append("  (no cross-module links detected)")

        # Communities
        if graph.communities:
            lines.append(f"\n## Detected Communities: {len(graph.communities)}")
            for cid, members in list(graph.communities.items())[:10]:
                sample = ", ".join(str(m) for m in members[:3])
                lines.append(f"  Community {cid}: {len(members)} node(s) — e.g. {sample}")

        # Cycles
        try:
            cycle = nx.find_cycle(g, orientation="original")
            cycle_nodes = [u for u, _ in cycle]
            lines.append(
                f"\n⚠️  Circular dependency: "
                + " -> ".join(str(n) for n in cycle_nodes + [cycle_nodes[0]])
            )
        except nx.NetworkXNoCycle:
            lines.append("\n✓ No circular dependencies detected.")

        return "\n".join(lines)

    def _verify_rules(
        g: nx.DiGraph,
        rules: list[dict],
        vector_engine: Any,
    ) -> str:
        """Stage 3: deterministic verification + hybrid evidence."""
        from core.hybrid_retriever import HybridFusionRetriever

        retriever = HybridFusionRetriever(graph, vector_engine)

        lines: list[str] = []
        lines.append("# Architectural Compliance Verification")
        total_violations = 0

        for rule in rules:
            rid = rule["rule_id"]
            rtype = rule["rule_type"]
            src_pat = rule["source_pattern"]
            tgt_pat = rule["target_pattern"]
            desc = rule["description"]

            violations: list[dict] = []
            for u, v, data in g.edges(data=True):
                u_fp = str(g.nodes[u].get("filepath") or g.nodes[u].get("source_file", ""))
                v_fp = str(g.nodes[v].get("filepath") or g.nodes[v].get("source_file", ""))
                u_label = str(g.nodes[u].get("label", u))
                v_label = str(g.nodes[v].get("label", v))
                u_full = f"{u_fp}::{u_label}"
                v_full = f"{v_fp}::{v_label}"

                src_match = _match_pattern(u_full, src_pat) or _match_pattern(u_fp, src_pat)
                tgt_match = _match_pattern(v_full, tgt_pat) or _match_pattern(v_fp, tgt_pat)

                if src_match and tgt_match:
                    if rtype == "forbidden":
                        violations.append(
                            {
                                "source": u_full,
                                "target": v_full,
                                "relation": data.get("relation", "calls"),
                                "source_file": u_fp,
                                "target_file": v_fp,
                            }
                        )
                    elif rtype == "required":
                        # Required rule satisfied by at least one edge
                        pass

            if rtype == "forbidden":
                if violations:
                    total_violations += len(violations)
                    lines.append(f"\n❌ VIOLATION (DRIFT) — Rule '{rid}': {desc}")
                    lines.append(f"   Forbidden: {src_pat} -> {tgt_pat}")
                    for v in violations[:10]:
                        # Get line-number evidence via hybrid retriever
                        evidence = retriever.retrieve(
                            f"{v['source']} calls {v['target']}", top_k=2
                        )
                        loc_info = ""
                        for res in evidence.get("results", []):
                            fp = res.get("filepath", "")
                            s = res.get("start_line")
                            e = res.get("end_line")
                            if fp and s and e:
                                loc_info = f" ({fp}:{s}-{e})"
                                break
                        lines.append(
                            f"   - {v['source']} --[{v['relation']}]--> {v['target']}{loc_info}"
                        )
                else:
                    lines.append(f"\n✓ COMPLIANT — Rule '{rid}': {desc}")
            elif rtype == "required":
                # Check if at least one edge satisfies the required link
                satisfied = any(
                    _match_pattern(
                        str(g.nodes[u].get("filepath") or g.nodes[u].get("source_file", "")),
                        src_pat,
                    )
                    and _match_pattern(
                        str(g.nodes[v].get("filepath") or g.nodes[v].get("source_file", "")),
                        tgt_pat,
                    )
                    for u, v in g.edges()
                )
                if satisfied:
                    lines.append(f"\n✓ COMPLIANT — Rule '{rid}': {desc}")
                else:
                    total_violations += 1
                    lines.append(
                        f"\n❌ VIOLATION (MISSING) — Rule '{rid}': {desc}"
                    )
                    lines.append(f"   Required: {src_pat} -> {tgt_pat}")
                    lines.append("   No AST edge found connecting source to target patterns.")

        lines.append(f"\n## Summary: {total_violations} violation(s) across {len(rules)} rule(s)")
        return "\n".join(lines)

    return [
        search_code,
        explain_symbol,
        trace_path,
        read_file,
        list_files,
        consolidate_files,
        audit_blast_radius,
        audit_against_blueprint,
    ]
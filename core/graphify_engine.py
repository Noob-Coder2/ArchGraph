"""
Graphify Code & Knowledge Graph Engine (Hardened & Low-RAM Edition)
Key Production Upgrades:
1. Path Traversal Protection: Confines all temporary file writes to `temp_root` via
   `Path.is_relative_to()` to block Zip-Slip attacks.
2. Zero-RAM Source File & Artifact Offloading: Uses `SessionFileStore` (a dict-compatible
   proxy backed by SQLite) and lazy properties for `html_content` / `report_markdown`.
   Once `build_graphify_pipeline()` finishes, heavy file strings and multi-megabyte HTML
   bundles are spilled to SQLite and freed from Python RAM.
3. Specificity-Ranked Seed Traversal: Filters out query stopwords ("how", "does", "work", etc.)
   and scores candidate AST nodes by token match count, exact/prefix match, and degree centrality.
"""
import os
import re
import tempfile
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
import networkx as nx

logger = logging.getLogger(__name__)

# Official Graphify modules
from graphify.detect import detect
from graphify.extract import extract, collect_files
from graphify.diagnostics import diagnose_extraction
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate as generate_graph_report
from graphify.export import to_json
from graphify.exporters.html import to_html as export_to_html

from services.storage import storage

QUERY_STOPWORDS = {
    "how", "what", "where", "when", "why", "who", "which", "does", "do", "did",
    "is", "are", "was", "were", "can", "could", "should", "would", "will",
    "the", "and", "for", "with", "from", "into", "about", "this", "that", "these",
    "those", "work", "works", "working", "code", "codebase", "repo", "repository",
    "file", "files", "function", "functions", "class", "classes", "method", "methods",
    "module", "modules", "use", "used", "using", "show", "explain", "find", "get",
    "list", "trace", "path", "between", "all", "any", "some", "have", "has"
}


class SessionFileStore(dict):
    """
    Dict-compatible file store that holds files in staging RAM during ingestion,
    then spills raw text to SQLite (`session_files`) after AST build so Python RAM
    drops to near-zero while remaining 100% compatible with `dict` callers.
    """

    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id
        self._spilled_keys: set[str] = set()

    def spill_to_sqlite(self):
        """Writes staged in-memory file contents to SQLite and frees Python RAM."""
        if not super().__len__():
            return
        staged = dict(super().items())
        storage.save_session_files(self.session_id, staged)
        self._spilled_keys.update(staged.keys())
        super().clear()

    def clear(self):
        super().clear()
        self._spilled_keys.clear()

    def __len__(self) -> int:
        return len(set(super().keys()) | self._spilled_keys)

    def __bool__(self) -> bool:
        return bool(super().__len__() or self._spilled_keys)

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key) or (key in self._spilled_keys)

    def keys(self):  # type: ignore[override]
        return sorted(set(super().keys()) | self._spilled_keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __getitem__(self, key: str) -> str:
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self._spilled_keys:
            val = storage.get_session_file(self.session_id, key)
            if val is not None:
                return val
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self) -> List[Tuple[str, str]]:  # type: ignore[override]
        return [(k, self.get(k, "")) for k in self.keys()]

    def values(self) -> List[str]:  # type: ignore[override]
        return [self.get(k, "") for k in self.keys()]


class GraphifyEngine:
    """Production-grade Graphify engine managing structural code graphs and GraphRAG operations."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or "default"
        self.graph: nx.DiGraph = nx.DiGraph()
        self.communities: Dict[int, List[str]] = {}
        self.community_labels: Dict[int, str] = {}
        self.god_nodes_list: List[Dict[str, Any]] = []
        self.surprises_list: List[Dict[str, Any]] = []
        self.suggested_questions_list: List[str] = []
        self.diagnostics_summary: Dict[str, Any] = {}
        self.file_contents: SessionFileStore = SessionFileStore(self.session_id)
        self._report_markdown: Optional[str] = ""
        self._html_content: Optional[str] = ""

    @property
    def report_markdown(self) -> str:
        """Fetches report from RAM if present, otherwise streams from SQLite spill."""
        if self._report_markdown is not None:
            return self._report_markdown
        artifacts = storage.get_artifacts(self.session_id)
        return (artifacts or {}).get("graph_report_md", "")

    @report_markdown.setter
    def report_markdown(self, value: Optional[str]):
        self._report_markdown = value

    @property
    def html_content(self) -> str:
        """Fetches graph.html from RAM if present, otherwise streams from SQLite spill."""
        if self._html_content is not None:
            return self._html_content
        artifacts = storage.get_artifacts(self.session_id)
        return (artifacts or {}).get("graph_html", "")

    @html_content.setter
    def html_content(self, value: Optional[str]):
        self._html_content = value

    def clear(self):
        """Resets all in-memory graph state."""
        self.graph.clear()
        self.communities.clear()
        self.community_labels.clear()
        self.god_nodes_list.clear()
        self.surprises_list.clear()
        self.suggested_questions_list.clear()
        self.diagnostics_summary.clear()
        self.file_contents.clear()
        self._report_markdown = ""
        self._html_content = ""

    def add_file(self, filepath: str, content: str):
        """Registers a file buffer in staging memory with normalized path."""
        clean_path = filepath.replace('\\', '/').lstrip('/')
        self.file_contents[clean_path] = content

    def build_graphify_pipeline(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Executes the complete Graphify pipeline, persists heavy artifacts and source files
        to SQLite, and frees large strings from Python RAM.
        """
        if not self.file_contents:
            return {"status": "empty", "nodes": 0, "edges": 0}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir).resolve()
            target_out_dir = (output_dir or (temp_root / "graphify-out")).resolve()
            target_out_dir.mkdir(parents=True, exist_ok=True)

            # 1. Write files to temp disk with strict Zip-Slip / Path Traversal confinement
            manifest_tracker: Dict[str, Dict[str, Any]] = {}
            safe_written_paths: List[str] = []

            for rel_path, content in self.file_contents.items():
                norm_rel = Path(rel_path.replace("\\", "/").lstrip("/"))
                if ".." in norm_rel.parts:
                    logger.warning("Blocked path traversal filename during ingestion: %s", rel_path)
                    continue
                dest = (temp_root / norm_rel).resolve()
                if not dest.is_relative_to(temp_root):
                    logger.warning("Blocked path escaping temp root: %s", rel_path)
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8", errors="replace")
                safe_written_paths.append(rel_path)

                sha = hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()
                ext = norm_rel.suffix.lower()
                ftype = "code" if ext in {'.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.c', '.cpp', '.cs'} else "document"
                manifest_tracker[rel_path] = {"hash": sha, "type": ftype}

            # 2. Multi-Modal Detection
            try:
                detection_result = detect(temp_root)
            except Exception:
                logger.error("Multi-Modal Detection failed", exc_info=True)
                detection_result = {
                    "total_files": len(safe_written_paths),
                    "files": {"code": safe_written_paths},
                    "total_words": sum(len(self.file_contents.get(p, "").split()) for p in safe_written_paths)
                }

            # 3. Deterministic Tree-Sitter AST Extraction
            code_file_paths = []
            for cat in ("code",):
                for fpath in detection_result.get("files", {}).get(cat, []):
                    p = Path(fpath)
                    if not p.is_absolute():
                        p = temp_root / p
                    if p.is_file():
                        code_file_paths.append(p)

            if not code_file_paths:
                code_file_paths = collect_files(temp_root)

            try:
                ast_extraction = extract(code_file_paths, cache_root=temp_root)
            except Exception:
                logger.error("AST Extraction failed for %d files", len(code_file_paths), exc_info=True)
                ast_extraction = {"nodes": [], "edges": []}

            # 4. Diagnostics & Health Gate
            try:
                self.diagnostics_summary = diagnose_extraction(
                    ast_extraction,
                    directed=True,
                    root=temp_root
                )
            except Exception:
                logger.error("Diagnostics failed", exc_info=True)
                self.diagnostics_summary = {"status": "diagnostics_skipped"}

            # 5. Graph Construction
            self.graph = build_from_json(ast_extraction, root=temp_root, directed=True)

            for rel_path in safe_written_paths:
                if not self.graph.has_node(rel_path):
                    self.graph.add_node(
                        rel_path,
                        label=os.path.basename(rel_path),
                        file_type="file",
                        source_file=rel_path
                    )

            # 6. Community Clustering & Structural Analytics
            if self.graph.number_of_nodes() > 1:
                try:
                    undirected_g = self.graph.to_undirected()
                    self.communities = cluster(undirected_g)
                    cohesion = score_all(undirected_g, self.communities)
                    self.god_nodes_list = god_nodes(undirected_g)
                    self.surprises_list = surprising_connections(undirected_g, self.communities)
                    self.community_labels = {cid: f"Module {cid}" for cid in self.communities}
                    self.suggested_questions_list = suggest_questions(
                        undirected_g, self.communities, self.community_labels
                    )
                except Exception:
                    logger.error("Community clustering or analysis failed", exc_info=True)
                    self.communities = {0: list(self.graph.nodes())}
                    self.community_labels = {0: "Core Module"}
                    cohesion = {}
            else:
                self.communities = {0: list(self.graph.nodes())} if self.graph.number_of_nodes() == 1 else {}
                self.community_labels = {0: "Single Module"} if self.communities else {}
                cohesion = {}

            # 7. Generate GRAPH_REPORT.md & graph.json
            try:
                tokens = {"input": 0, "output": 0}
                generated_report = generate_graph_report(
                    self.graph,
                    self.communities,
                    cohesion,
                    self.community_labels,
                    self.god_nodes_list,
                    self.surprises_list,
                    detection_result,
                    tokens,
                    root=str(temp_root),
                    suggested_questions=self.suggested_questions_list
                )
            except Exception:
                logger.error("Report generation failed", exc_info=True)
                generated_report = self._build_concise_summary()

            graph_json_path = target_out_dir / "graph.json"
            to_json(self.graph, self.communities, str(graph_json_path), community_labels=self.community_labels)

            # 8. Export Official Interactive HTML
            html_path = target_out_dir / "graph.html"
            generated_html = ""
            try:
                export_to_html(
                    self.graph,
                    self.communities,
                    str(html_path),
                    community_labels=self.community_labels,
                    node_limit=5000
                )
                if html_path.exists():
                    generated_html = html_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                logger.error("HTML generation failed", exc_info=True)

            # 9. Persist to Ephemeral SQLite & Free Python RAM
            raw_graph_json = graph_json_path.read_text(encoding="utf-8") if graph_json_path.exists() else "{}"
            analysis_data = {
                "communities": {str(k): v for k, v in self.communities.items()},
                "god_nodes": self.god_nodes_list,
                "surprises": self.surprises_list,
                "suggested_questions": self.suggested_questions_list,
                "diagnostics": self.diagnostics_summary
            }

            storage.create_or_update_session(
                session_id=self.session_id,
                file_count=len(safe_written_paths),
                ast_nodes=self.graph.number_of_nodes(),
                ast_edges=self.graph.number_of_edges(),
                communities_count=len(self.communities),
                god_nodes_count=len(self.god_nodes_list)
            )
            storage.save_artifacts(
                session_id=self.session_id,
                graph_json=raw_graph_json,
                graph_report_md=generated_report,
                graph_html=generated_html,
                analysis_json=json.dumps(analysis_data, indent=2)
            )
            storage.save_manifest(self.session_id, manifest_tracker)

            # Spill raw source files to SQLite and drop multi-MB HTML/report strings from RAM
            self.file_contents.spill_to_sqlite()
            self._report_markdown = None
            self._html_content = None

        return {
            "status": "success",
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": len(self.communities),
            "god_nodes": len(self.god_nodes_list),
            "diagnostics": self.diagnostics_summary
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Topological GraphRAG Traversal Tools (Ranked Seed Selection)
    # ─────────────────────────────────────────────────────────────────────────

    def _rank_seed_nodes(self, question: str, max_seeds: int = 4) -> List[str]:
        """
        Scores AST graph nodes against meaningful query tokens so traversal starts
        from the most specific symbols instead of arbitrary stopword matches.
        """
        raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", question)
        tokens = [
            t.lower() for t in raw_tokens
            if len(t) > 2 and t.lower() not in QUERY_STOPWORDS
        ]

        scored_nodes: List[Tuple[float, str]] = []
        if tokens:
            for n, data in self.graph.nodes(data=True):
                label = str(data.get("label", n)).lower()
                node_id_lower = str(n).lower()
                score = 0.0
                for tok in tokens:
                    if label == tok or node_id_lower.endswith(f"::{tok}"):
                        score += 10.0  # Exact symbol match
                    elif tok in label:
                        score += 4.0 + (len(tok) / max(len(label), 1)) * 2.0
                    elif tok in node_id_lower:
                        score += 2.0
                if score > 0:
                    deg_bonus = min(self.graph.degree(n) * 0.15, 2.5)
                    scored_nodes.append((score + deg_bonus, n))

            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            if scored_nodes:
                return [n for _, n in scored_nodes[:max_seeds]]

        # Fallback to top God Nodes if query is purely conceptual/architectural
        if self.god_nodes_list:
            seeds = [g.get("node", "") for g in self.god_nodes_list[:max_seeds] if g.get("node") in self.graph]
            if seeds:
                return seeds

        degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
        return [degrees[0][0]] if degrees else []

    def query_traversal(self, question: str, mode: str = "bfs", depth: int = 2, budget: int = 1500) -> str:
        """
        Topological graph search: Identifies top-ranked seed nodes matching query tokens,
        traverses callers and callees, and formats community-structured context within character budget.
        """
        if self.graph.number_of_nodes() == 0:
            return "Code graph is empty. Please ingest a codebase first."

        seed_nodes = self._rank_seed_nodes(question, max_seeds=4)
        if not seed_nodes:
            return "No relevant structural graph paths found."

        traversal_results = []
        undirected = self.graph.to_undirected()

        for seed in seed_nodes:
            if seed not in self.graph:
                continue
            seed_data = self.graph.nodes[seed]
            seed_label = seed_data.get("label", seed)
            seed_file = seed_data.get("filepath") or seed_data.get("source_file", "unknown")
            traversal_results.append(f"### Graph Anchor: `{seed_label}` (`{seed_file}`)")

            if mode.lower() == "dfs":
                neighbors = list(nx.dfs_preorder_nodes(undirected, source=seed, depth_limit=depth))
            else:
                neighbors = list(nx.bfs_tree(undirected, source=seed, depth_limit=depth).nodes())

            for nb in neighbors[:12]:
                if nb == seed:
                    continue
                d = self.graph.nodes[nb]
                edge_info = self.graph.edges.get((seed, nb)) or self.graph.edges.get((nb, seed)) or {}
                rel = edge_info.get("relation", "connected_to")
                nb_file = d.get("filepath") or d.get("source_file", "unknown")
                traversal_results.append(
                    f"- `{d.get('label', nb)}` ({d.get('type', d.get('file_type', 'symbol'))}) in `{nb_file}` [{rel}]"
                )

        out = "\n".join(traversal_results)
        if budget and len(out) > budget * 4:
            out = out[: budget * 4] + "\n... [graph traversal truncated to fit token budget]"
        return out

    def find_path(self, source_symbol: str, target_symbol: str) -> str:
        """Computes the shortest dependency or call-chain path between two symbols."""
        if self.graph.number_of_nodes() == 0:
            return "Graph is empty."

        def _match_node(sym: str) -> Optional[str]:
            sym_low = sym.strip().lower()
            # 1. Exact ID or label match
            for n in self.graph.nodes:
                if n == sym or self.graph.nodes[n].get("label") == sym or n.endswith(f"::{sym}"):
                    return n
            # 2. Case-insensitive fallback
            for n in self.graph.nodes:
                lbl = str(self.graph.nodes[n].get("label", n)).lower()
                if lbl == sym_low or str(n).lower().endswith(sym_low):
                    return n
            return None

        src = _match_node(source_symbol)
        tgt = _match_node(target_symbol)

        if not src:
            return f"Source symbol '{source_symbol}' was not found in the code graph."
        if not tgt:
            return f"Target symbol '{target_symbol}' was not found in the code graph."

        try:
            path = nx.shortest_path(self.graph.to_undirected(), source=src, target=tgt)
            chain = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                u_lbl = self.graph.nodes[u].get('label', u)
                v_lbl = self.graph.nodes[v].get('label', v)
                rel = self.graph.edges.get((u, v), self.graph.edges.get((v, u), {})).get("relation", "links_to")
                chain.append(f"`{u_lbl}` --[{rel}]--> `{v_lbl}`")
            return f"**Shortest Path ({len(path) - 1} hops):**\n" + " -> ".join(chain)
        except nx.NetworkXNoPath:
            return f"No path exists in the code graph between `{source_symbol}` and `{target_symbol}`."

    def explain_symbol(self, symbol: str) -> str:
        """Returns deep structural audit context for a symbol: definition, callers, callees, and community."""
        if self.graph.number_of_nodes() == 0:
            return f"Graph is empty. Cannot explain '{symbol}'."

        sym_low = symbol.strip().lower()
        matches = [
            n for n in self.graph.nodes
            if n.endswith(f"::{symbol}") or n == symbol or self.graph.nodes[n].get("label") == symbol
        ]
        if not matches:
            matches = [
                n for n in self.graph.nodes
                if str(self.graph.nodes[n].get("label", "")).lower() == sym_low or str(n).lower().endswith(sym_low)
            ]
        if not matches:
            return f"Symbol '{symbol}' was not found in the code graph."

        nid = matches[0]
        data = self.graph.nodes[nid]
        filepath = data.get("filepath") or data.get("source_file", "unknown")
        lines = [
            f"### Symbol Audit: `{data.get('label', nid)}`",
            f"- **Node ID**: `{nid}`",
            f"- **Type**: {data.get('type', data.get('file_type', 'symbol'))}",
            f"- **File**: `{filepath}`",
            f"- **Degree Centrality**: {self.graph.degree(nid)} connections"
        ]

        preds = list(self.graph.predecessors(nid)) if self.graph.is_directed() else list(self.graph.neighbors(nid))
        if preds:
            callers = [f"`{self.graph.nodes[p].get('label', p)}`" for p in preds[:10]]
            lines.append(f"- **Referenced By (Callers)**: {', '.join(callers)}")

        succs = list(self.graph.successors(nid)) if self.graph.is_directed() else []
        if succs:
            dependencies = [f"`{self.graph.nodes[s].get('label', s)}`" for s in succs[:10]]
            lines.append(f"- **Dependencies (Callees)**: {', '.join(dependencies)}")

        return "\n".join(lines)

    def _build_concise_summary(self) -> str:
        summary = ["### Graphify Architectural Overview\n"]
        summary.append(f"- **Total Entities**: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} relationships")
        summary.append(f"- **Detected Communities**: {len(self.communities)} cohesive functional modules")
        if self.god_nodes_list:
            top_gods = [f"`{g.get('node', g)}`" for g in self.god_nodes_list[:5]]
            summary.append(f"- **Central God Nodes**: {', '.join(top_gods)}")
        return "\n".join(summary)

    def get_architectural_summary(self) -> str:
        """Returns the high-level executive report summary."""
        rep = self.report_markdown
        return rep if rep else self._build_concise_summary()

    def get_symbol_context(self, symbol: str) -> str:
        """Alias for explain_symbol for backwards compatibility."""
        return self.explain_symbol(symbol)

    def to_graph_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Serializes nodes and edges for JSON API responses."""
        if self.graph.number_of_nodes() == 0 and self.file_contents:
            self.build_graphify_pipeline()

        nodes = []
        for n, attrs in self.graph.nodes(data=True):
            nodes.append({
                "id": str(n),
                "label": attrs.get("label", str(n)),
                "type": attrs.get("type", attrs.get("file_type", "symbol")),
                "filepath": attrs.get("filepath", attrs.get("source_file", ""))
            })

        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append({
                "source": str(u),
                "target": str(v),
                "relation": attrs.get("relation", "relates")
            })

        return {"nodes": nodes, "edges": edges}
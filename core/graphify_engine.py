"""
Graphify Code & Knowledge Graph Engine (Complete Full-Stack Implementation)
Integrates all 7 native Graphify capabilities:
1. Multi-Modal Detection (detect.py)
2. Multi-Language Tree-Sitter AST Extraction (extract.py)
3. Graph Diagnostics & Integrity Gate (diagnostics.py)
4. Graph Construction & Louvain Community Clustering (build.py & cluster.py)
5. Architectural Analysis & God Node Detection (analyze.py)
6. Official Interactive HTML & Report Generation (exporters/html.py & report.py)
7. Topological GraphRAG Traversal (Query BFS/DFS, Path, Explain) & SHA Caching
"""
import os
import tempfile
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx

logger = logging.getLogger(__name__)

# Official Graphify modules
from graphify.detect import detect, save_manifest
from graphify.extract import extract, collect_files
from graphify.diagnostics import diagnose_extraction
from graphify.build import build_from_json, edge_data
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate as generate_graph_report
from graphify.export import to_json
from graphify.exporters.html import to_html as export_to_html
from services.storage import storage


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
        self.file_contents: Dict[str, str] = {}
        self.report_markdown: str = ""
        self.html_content: str = ""

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
        self.report_markdown = ""
        self.html_content = ""

    def add_file(self, filepath: str, content: str):
        """Registers a file buffer in memory."""
        clean_path = filepath.replace('\\', '/')
        self.file_contents[clean_path] = content

    def build_graphify_pipeline(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Executes the complete Graphify pipeline:
        1. Multi-modal file detection & classification
        2. Tree-Sitter AST extraction
        3. Extraction diagnostics health check
        4. Directed graph build & Louvain community clustering
        5. God Node & surprising connection analysis
        6. Report & standalone interactive HTML generation
        7. Persistence to embedded SQLite storage
        """
        if not self.file_contents:
            return {"status": "empty", "nodes": 0, "edges": 0}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            target_out_dir = output_dir or (temp_path / "graphify-out")
            target_out_dir.mkdir(parents=True, exist_ok=True)

            # 1. Write files to disk for Tree-Sitter & detector
            manifest_tracker: Dict[str, Dict[str, Any]] = {}
            for rel_path, content in self.file_contents.items():
                dest = temp_path / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8", errors="replace")
                
                # Compute SHA-256 for incremental cache
                sha = hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()
                ext = Path(rel_path).suffix.lower()
                ftype = "code" if ext in ['.py', '.js', '.ts', '.go', '.rs', '.java', '.c', '.cpp', '.cs'] else "document"
                manifest_tracker[rel_path] = {"hash": sha, "type": ftype}

            # 2. Multi-Modal Detection (Capability 1)
            try:
                detection_result = detect(temp_path)
            except Exception as e:
                logger.error("Multi-Modal Detection failed", exc_info=True)
                detection_result = {
                    "total_files": len(self.file_contents),
                    "files": {"code": list(self.file_contents.keys())},
                    "total_words": sum(len(c.split()) for c in self.file_contents.values())
                }

            # 3. Deterministic AST Extraction across all code files (Capability 2)
            code_file_paths = []
            for cat in ("code",):
                for fpath in detection_result.get("files", {}).get(cat, []):
                    p = Path(fpath)
                    if not p.is_absolute():
                        p = temp_path / p
                    if p.is_file():
                        code_file_paths.append(p)

            # Fallback: if detection didn't resolve full paths, collect from temp dir
            if not code_file_paths:
                code_file_paths = collect_files(temp_path)

            try:
                ast_extraction = extract(code_file_paths, cache_root=temp_path)
            except Exception as e:
                logger.error(f"AST Extraction failed for {len(code_file_paths)} files", exc_info=True)
                ast_extraction = {"nodes": [], "edges": []}

            # 4. Diagnostics & Health Gate (Capability 5)
            try:
                self.diagnostics_summary = diagnose_extraction(
                    ast_extraction,
                    directed=True,
                    root=temp_path
                )
            except Exception as e:
                logger.error("Diagnostics failed", exc_info=True)
                self.diagnostics_summary = {"status": "diagnostics_skipped"}

            # 5. Graph Construction (Part C / Step 4)
            self.graph = build_from_json(ast_extraction, root=temp_path, directed=True)

            # Ensure every indexed file has a backing node
            for rel_path in self.file_contents.keys():
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
                    
                    # Generate semantic community labels
                    self.community_labels = {
                        cid: f"Module {cid}" for cid in self.communities
                    }
                    self.suggested_questions_list = suggest_questions(
                        undirected_g, self.communities, self.community_labels
                    )
                except Exception as e:
                    logger.error("Community clustering or analysis failed", exc_info=True)
                    self.communities = {0: list(self.graph.nodes())}
                    self.community_labels = {0: "Core Module"}
                    cohesion = {}
            else:
                self.communities = {0: list(self.graph.nodes())} if self.graph.number_of_nodes() == 1 else {}
                self.community_labels = {0: "Single Module"} if self.communities else {}
                cohesion = {}

            # 7. Generate GRAPH_REPORT.md & graph.json (Step 4 & 5)
            try:
                tokens = {"input": 0, "output": 0}
                self.report_markdown = generate_graph_report(
                    self.graph,
                    self.communities,
                    cohesion,
                    self.community_labels,
                    self.god_nodes_list,
                    self.surprises_list,
                    detection_result,
                    tokens,
                    root=str(temp_path),
                    suggested_questions=self.suggested_questions_list
                )
            except Exception as e:
                logger.error("Report generation failed", exc_info=True)
                self.report_markdown = self.get_architectural_summary()

            graph_json_path = target_out_dir / "graph.json"
            to_json(self.graph, self.communities, str(graph_json_path), community_labels=self.community_labels)

            # 8. Export Official Interactive HTML (Capability 7)
            html_path = target_out_dir / "graph.html"
            try:
                export_to_html(
                    self.graph,
                    self.communities,
                    str(html_path),
                    community_labels=self.community_labels,
                    node_limit=5000
                )
                if html_path.exists():
                    self.html_content = html_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.error("HTML generation failed", exc_info=True)
                self.html_content = ""

            # 9. Persist to Embedded SQLite Database
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
                file_count=len(self.file_contents),
                ast_nodes=self.graph.number_of_nodes(),
                ast_edges=self.graph.number_of_edges(),
                communities_count=len(self.communities),
                god_nodes_count=len(self.god_nodes_list)
            )
            storage.save_artifacts(
                session_id=self.session_id,
                graph_json=raw_graph_json,
                graph_report_md=self.report_markdown,
                graph_html=self.html_content,
                analysis_json=json.dumps(analysis_data, indent=2)
            )
            storage.save_manifest(self.session_id, manifest_tracker)

        return {
            "status": "success",
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": len(self.communities),
            "god_nodes": len(self.god_nodes_list),
            "diagnostics": self.diagnostics_summary
        }

    # ─────────────────────────────────────────────────────────────
    # Topological GraphRAG Traversal Tools (Agent Native APIs)
    # ─────────────────────────────────────────────────────────────

    def query_traversal(self, question: str, mode: str = "bfs", depth: int = 2, budget: int = 1500) -> str:
        """
        Topological graph search: Identifies seed nodes matching query tokens,
        traverses callers and callees, and formats community-structured context.
        """
        if self.graph.number_of_nodes() == 0:
            return "Code graph is empty. Please ingest a codebase first."

        words = [w.lower() for w in question.split() if len(w) > 2]
        seed_nodes = []
        for n, data in self.graph.nodes(data=True):
            label = str(data.get("label", n)).lower()
            if any(w in label for w in words):
                seed_nodes.append(n)

        # Fallback to top God Nodes if no exact word matches
        if not seed_nodes and self.god_nodes_list:
            seed_nodes = [g.get("node", "") for g in self.god_nodes_list[:3] if g.get("node")]

        if not seed_nodes:
            # Pick highest degree node
            degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
            if degrees:
                seed_nodes = [degrees[0][0]]

        visited = set()
        traversal_results = []

        for seed in seed_nodes[:4]:
            if seed not in self.graph:
                continue
            traversal_results.append(f"### 📍 Graph Anchor: `{self.graph.nodes[seed].get('label', seed)}`")
            visited.add(seed)

            # Traverse neighbors
            if mode.lower() == "dfs":
                # Depth-first search (deep execution trace)
                neighbors = list(nx.dfs_preorder_nodes(self.graph.to_undirected(), source=seed, depth_limit=depth))
            else:
                # Breadth-first search (broad architectural context)
                neighbors = list(nx.bfs_tree(self.graph.to_undirected(), source=seed, depth_limit=depth).nodes())

            for nb in neighbors[:10]:
                if nb == seed:
                    continue
                d = self.graph.nodes[nb]
                rel = self.graph.edges.get((seed, nb), {}).get("relation", "connected_to")
                traversal_results.append(f"- `{d.get('label', nb)}` ({d.get('type', 'symbol')}) in `{d.get('filepath', 'unknown')}` [{rel}]")

        return "\n".join(traversal_results) if traversal_results else "No relevant structural graph paths found."

    def find_path(self, source_symbol: str, target_symbol: str) -> str:
        """Computes the shortest dependency or call-chain path between two symbols."""
        if self.graph.number_of_nodes() == 0:
            return "Graph is empty."

        def _match_node(sym: str) -> Optional[str]:
            for n in self.graph.nodes:
                if n == sym or self.graph.nodes[n].get("label") == sym or n.endswith(f"::{sym}"):
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
                u, v = path[i], path[i+1]
                u_lbl = self.graph.nodes[u].get('label', u)
                v_lbl = self.graph.nodes[v].get('label', v)
                rel = self.graph.edges.get((u, v), self.graph.edges.get((v, u), {})).get("relation", "links_to")
                chain.append(f"`{u_lbl}` --[{rel}]--> `{v_lbl}`")
            return f"**Shortest Path ({len(path)-1} hops):**\n" + " -> ".join(chain)
        except nx.NetworkXNoPath:
            return f"No path exists in the code graph between `{source_symbol}` and `{target_symbol}`."

    def explain_symbol(self, symbol: str) -> str:
        """Returns deep structural audit context for a symbol: definition, callers, callees, and community."""
        if self.graph.number_of_nodes() == 0:
            return f"Graph is empty. Cannot explain '{symbol}'."

        matches = [
            n for n in self.graph.nodes
            if n.endswith(f"::{symbol}") or n == symbol or self.graph.nodes[n].get("label") == symbol
        ]
        if not matches:
            return f"Symbol '{symbol}' was not found in the code graph."

        nid = matches[0]
        data = self.graph.nodes[nid]
        lines = [
            f"### 🔍 Symbol Audit: `{data.get('label', nid)}`",
            f"- **Type**: {data.get('type', 'symbol')}",
            f"- **File**: `{data.get('filepath', 'unknown')}`",
            f"- **Degree Centrality**: {self.graph.degree(nid)} connections"
        ]

        # Callers (predecessors)
        preds = list(self.graph.predecessors(nid)) if self.graph.is_directed() else list(self.graph.neighbors(nid))
        if preds:
            callers = [f"`{self.graph.nodes[p].get('label', p)}`" for p in preds[:8]]
            lines.append(f"- **Referenced By (Callers)**: {', '.join(callers)}")

        # Callees (successors)
        succs = list(self.graph.successors(nid)) if self.graph.is_directed() else []
        if succs:
            dependencies = [f"`{self.graph.nodes[s].get('label', s)}`" for s in succs[:8]]
            lines.append(f"- **Dependencies (Callees)**: {', '.join(dependencies)}")

        return "\n".join(lines)

    def get_architectural_summary(self) -> str:
        """Returns the high-level executive report summary."""
        if self.report_markdown:
            return self.report_markdown

        summary = ["### 🏛️ Graphify Architectural Overview\n"]
        summary.append(f"- **Total Entities**: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} relationships")
        summary.append(f"- **Detected Communities**: {len(self.communities)} cohesive functional modules")
        if self.god_nodes_list:
            top_gods = [f"`{g.get('node', g)}`" for g in self.god_nodes_list[:5]]
            summary.append(f"- **Central God Nodes**: {', '.join(top_gods)}")
        return "\n".join(summary)

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
                "type": attrs.get("type", "symbol"),
                "filepath": attrs.get("filepath", "")
            })
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append({
                "source": str(u),
                "target": str(v),
                "relation": attrs.get("relation", "relates")
            })
        return {"nodes": nodes, "edges": edges}

# 🕸️ ArchGraph — Agentic Codebase Intelligence & Architectural Audit Engine

**ArchGraph** is a lightweight, privacy-first **"Second-Opinion" Architectural Auditor & Onboarding Platform**. While IDE coding assistants frequently hallucinate across multi-file dependencies because they rely on naive text chunking, ArchGraph combines **Graphify deterministic AST topology** with **local ONNX semantic vectors** so developers can audit blast radius, circular dependencies, and God-nodes before writing or merging code.

---

## 🧠 Dual-Pillar Architecture

* **Pillar 1: Graphify Structural Code Knowledge Graph (`core/graphify_engine.py`)**
  * Powered by **Graphify**, using **Multi-Language Tree-Sitter AST Extraction** and **NetworkX** to build a directed dependency graph across Python, JS/TS, Go, Rust, Java, C/C++, and C#.
  * Runs **Graphify Louvain Community Clustering** to discover cohesive functional modules, **Degree Centrality Analysis** to flag high-coupling **God Nodes**, and **Surprising Connection Detection**.
* **Pillar 2: Local FastEmbed ONNX Vector Engine (`core/llamaindex_engine.py`)**
  * Powered by **LlamaIndex** + **`BAAI/bge-small-en-v1.5`** running 100% locally on CPU via ONNX Runtime (zero embedding API cost) with exact `start_line` and `end_line` metadata citations.
* **Agentic Orchestration (`core/agent.py` & `core/tools.py`)**
  * Powered by **LangGraph** (`astream_events` v2) with **6 sandboxed, token-bounded tools** (`search_code`, `explain_symbol`, `trace_path`, `read_file`, `list_files`, `consolidate_files`) and real-time Server-Sent Events (SSE) tool timeline telemetry.

---

## 🔒 Zero-Retention Ephemeral Privacy Model

ArchGraph is built for strict intellectual-property safety:
* **No Permanent Source Code Storage:** Source files live strictly in ephemeral session memory during active analysis.
* **Lightweight SQLite Artifact Offloading:** Only rendered graph topology (`graph.json`, `graph.html`, `GRAPH_REPORT.md`) is cached in `data/archgraph.db` (WAL mode) to keep RAM usage flat[cite: 2, 3, 5].
* **Strict 1-Hour Auto-Purge:** Inactive sessions are permanently wiped after 1 hour (`HTTP 410 Gone`).

---

## 🚀 Local Setup & Development

### 1. Install in Editable Mode
```bash
git clone [https://github.com/noob-coder2/archgraph.git](https://github.com/noob-coder2/archgraph.git)
cd archgraph
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
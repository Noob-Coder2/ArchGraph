"""
Unit tests for DocuStream Core: Consolidator & GitHub Services.
"""
import io
import types
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from services.consolidator import DocumentConsolidator
from services.github_service import GitHubService


# ─────────────────────────────────────────────────────────────────────────────
# DocumentConsolidator — read_file_content  (markitdown path)
# ─────────────────────────────────────────────────────────────────────────────

class TestReadFileContent:
    """Tests for DocumentConsolidator.read_file_content."""

    def test_plain_text_file(self):
        f = io.BytesIO(b"Hello, world!")
        f.name = "readme.txt"
        content, was_converted = DocumentConsolidator.read_file_content(f)
        assert content == "Hello, world!"
        assert was_converted is False

    def test_python_code_file(self):
        src = b"def add(a, b):\n    return a + b\n"
        f = io.BytesIO(src)
        f.name = "utils.py"
        content, was_converted = DocumentConsolidator.read_file_content(f)
        assert "def add" in content
        assert was_converted is False

    def test_markdown_file(self):
        src = b"# Title\n\nSome text here."
        f = io.BytesIO(src)
        f.name = "README.md"
        content, was_converted = DocumentConsolidator.read_file_content(f)
        assert "# Title" in content
        assert was_converted is False

    def test_pdf_uses_markitdown_when_available(self):
        fake_result = MagicMock()
        fake_result.text_content = "## Extracted Heading\n\nParagraph text."
        with patch("services.consolidator._MARKITDOWN_AVAILABLE", True):
            with patch("services.consolidator.MarkItDown") as MockMD:
                MockMD.return_value.convert.return_value = fake_result
                f = io.BytesIO(b"%PDF-1.4 fake pdf bytes")
                f.name = "document.pdf"
                content, was_converted = DocumentConsolidator.read_file_content(f)
        assert "Extracted Heading" in content
        assert was_converted is True

    def test_docx_uses_markitdown_when_available(self):
        fake_result = MagicMock()
        fake_result.text_content = "Word document content here."
        with patch("services.consolidator._MARKITDOWN_AVAILABLE", True):
            with patch("services.consolidator.MarkItDown") as MockMD:
                MockMD.return_value.convert.return_value = fake_result
                f = io.BytesIO(b"PK\x03\x04 fake docx bytes")
                f.name = "report.docx"
                content, was_converted = DocumentConsolidator.read_file_content(f)
        assert "Word document content here." in content
        assert was_converted is True

    def test_rich_doc_falls_back_when_markitdown_unavailable(self):
        with patch("services.consolidator._MARKITDOWN_AVAILABLE", False):
            f = io.BytesIO(b"raw bytes that are valid utf-8")
            f.name = "slides.pptx"
            content, was_converted = DocumentConsolidator.read_file_content(f)
        assert "raw bytes" in content
        assert was_converted is False

    def test_markitdown_exception_falls_back_gracefully(self):
        with patch("services.consolidator._MARKITDOWN_AVAILABLE", True):
            with patch("services.consolidator.MarkItDown") as MockMD:
                MockMD.return_value.convert.side_effect = RuntimeError("conversion failed")
                f = io.BytesIO(b"fallback content bytes")
                f.name = "broken.pdf"
                content, was_converted = DocumentConsolidator.read_file_content(f)
        assert isinstance(content, str)
        assert was_converted is False


# ─────────────────────────────────────────────────────────────────────────────
# DocumentConsolidator — consolidate_files  (agent multi-file reader)
# ─────────────────────────────────────────────────────────────────────────────

def _make_engine(file_contents: dict):
    """Minimal mock of a GraphifyEngine exposing file_contents."""
    engine = types.SimpleNamespace()
    engine.file_contents = file_contents
    return engine


class TestConsolidateFiles:
    """
    Tests for DocumentConsolidator.consolidate_files — the agent tool that
    merges multiple indexed files into one Markdown payload in a single call.
    """

    def test_returns_string(self):
        engine = _make_engine({"src/utils.py": "def helper(): pass"})
        result = DocumentConsolidator.consolidate_files(engine, ["src/utils.py"])
        assert isinstance(result, str)

    def test_contains_all_requested_files(self):
        engine = _make_engine({
            "src/a.py": "# file A",
            "src/b.py": "# file B",
        })
        result = DocumentConsolidator.consolidate_files(
            engine, ["src/a.py", "src/b.py"]
        )
        assert "a.py" in result
        assert "b.py" in result
        assert "# file A" in result
        assert "# file B" in result

    def test_code_fenced_with_correct_language(self):
        engine = _make_engine({"module/core.py": "x = 1"})
        result = DocumentConsolidator.consolidate_files(engine, ["module/core.py"])
        assert "```python" in result
        assert "x = 1" in result

    def test_toc_included_by_default(self):
        engine = _make_engine({
            "a.py": "pass",
            "b.py": "pass",
        })
        result = DocumentConsolidator.consolidate_files(engine, ["a.py", "b.py"])
        assert "Table of Contents" in result

    def test_toc_excluded_when_flag_false(self):
        engine = _make_engine({"a.py": "pass"})
        result = DocumentConsolidator.consolidate_files(
            engine, ["a.py"], include_toc=False
        )
        assert "Table of Contents" not in result

    def test_partial_path_match(self):
        """A suffix / partial path like 'utils.py' resolves to 'src/utils.py'."""
        engine = _make_engine({"src/utils.py": "def foo(): ..."})
        result = DocumentConsolidator.consolidate_files(engine, ["utils.py"])
        assert "foo" in result

    def test_missing_path_reported_gracefully(self):
        engine = _make_engine({})
        result = DocumentConsolidator.consolidate_files(
            engine, ["nonexistent/file.py"]
        )
        assert "nonexistent" in result or "resolve" in result.lower()

    def test_empty_file_list_returns_message(self):
        engine = _make_engine({})
        result = DocumentConsolidator.consolidate_files(engine, [])
        assert "No file paths" in result

    def test_mixed_types_python_and_markdown(self):
        engine = _make_engine({
            "README.md": "# Project\nSome docs.",
            "app.py": "import os",
        })
        result = DocumentConsolidator.consolidate_files(
            engine, ["README.md", "app.py"]
        )
        assert "README.md" in result
        assert "app.py" in result
        assert "# Project" in result
        assert "import os" in result

    def test_rich_doc_content_inlined_as_markdown(self):
        """Stored rich-doc content (already Markdown after ingestion) renders inline."""
        engine = _make_engine({
            "report.pdf": "## Section 1\n\nMarkdown from markitdown."
        })
        result = DocumentConsolidator.consolidate_files(engine, ["report.pdf"])
        # Should appear as plain text, not wrapped in a code fence
        assert "## Section 1" in result
        assert "```" not in result.split("report.pdf")[1].split("---")[0]

    def test_path_listed_in_toc(self):
        engine = _make_engine({"core/engine.py": "class Engine: pass"})
        result = DocumentConsolidator.consolidate_files(engine, ["core/engine.py"])
        assert "engine.py" in result
        assert "Engine" in result

    def test_multiple_missing_paths_all_reported(self):
        engine = _make_engine({})
        result = DocumentConsolidator.consolidate_files(
            engine, ["missing1.py", "missing2.py"]
        )
        assert "missing1" in result
        assert "missing2" in result


# ─────────────────────────────────────────────────────────────────────────────
# GitHubService tests (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class TestGitHubService:
    def test_parse_github_url_repo_root(self):
        res = GitHubService.parse_github_url("https://github.com/pallets/flask")
        assert res is not None
        assert res['owner'] == 'pallets'
        assert res['repo'] == 'flask'
        assert res['branch'] == 'main'

    def test_parse_github_url_tree(self):
        res = GitHubService.parse_github_url("https://github.com/pallets/flask/tree/v2.0.0/src/flask")
        assert res is not None
        assert res['owner'] == 'pallets'
        assert res['repo'] == 'flask'
        assert res['branch'] == 'v2.0.0'
        assert res['path'] == 'src/flask'

    def test_parse_github_url_blob(self):
        res = GitHubService.parse_github_url("https://github.com/pallets/flask/blob/main/pyproject.toml")
        assert res is not None
        assert res['owner'] == 'pallets'
        assert res['repo'] == 'flask'
        assert res['branch'] == 'main'
        assert res['path'] == 'pyproject.toml'

    def test_parse_gist_url(self):
        gist_id = GitHubService.parse_gist_url("https://gist.github.com/octocat/6cad326836d38bd3a7ae")
        assert gist_id == "6cad326836d38bd3a7ae"

    def test_tree_structure_conversion(self):
        flat_tree = [
            {'path': 'src/app.py', 'type': 'blob'},
            {'path': 'src/utils.py', 'type': 'blob'},
            {'path': 'README.md', 'type': 'blob'}
        ]
        structured = GitHubService.build_tree_structure(flat_tree)
        assert len(structured) == 2

        tree_select_nodes = GitHubService.get_tree_select_nodes(structured)
        labels = [n['label'] for n in tree_select_nodes]
        assert "📁 src" in labels
        assert "📄 README.md" in labels

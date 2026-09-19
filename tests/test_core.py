"""
Unit tests for DocuStream Core: Consolidator & GitHub Services.
"""
import io
import pytest
from services.consolidator import DocumentConsolidator
from services.github_service import GitHubService


class TestDocumentConsolidator:
    def test_merge_to_markdown_preserves_headings(self):
        f1 = io.BytesIO(b"# Project Title\n\n## Subheading\n\nprint('hello world')")
        f1.name = "script.py"

        f2 = io.BytesIO(b"# Documentation\n\nThis is a readme file.")
        f2.name = "README.md"

        output = DocumentConsolidator.merge_to_markdown([f1, f2], include_toc=True)

        # Ensure headings and TOC are preserved
        assert "# Table of Contents" in output
        assert "1. [script.py]" in output
        assert "2. [README.md]" in output
        assert "# Project Title" in output
        assert "## Subheading" in output
        assert "# Documentation" in output
        assert "```python" in output

    def test_merge_to_text(self):
        f1 = io.BytesIO(b"first file content")
        f1.name = "f1.txt"
        f2 = io.BytesIO(b"second file content")
        f2.name = "f2.txt"

        output = DocumentConsolidator.merge_to_text([f1, f2])
        assert "FILE 1: f1.txt" in output
        assert "first file content" in output
        assert "FILE 2: f2.txt" in output
        assert "second file content" in output

    def test_consolidate_pdf_pure_python(self):
        f1 = io.BytesIO(b"def add(a, b):\n    return a + b\n")
        f1.name = "math_utils.py"

        data, mime = DocumentConsolidator.consolidate([f1], output_format="PDF")
        assert mime == "application/pdf"
        assert len(data) > 0
        assert data.startswith(b"%PDF")

    def test_unsupported_format_raises_error(self):
        f1 = io.BytesIO(b"sample")
        f1.name = "sample.txt"
        with pytest.raises(ValueError, match="Unsupported output format"):
            DocumentConsolidator.consolidate([f1], output_format="DOCX")


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
        assert len(structured) == 2  # 'src' folder and 'README.md' file

        tree_select_nodes = GitHubService.get_tree_select_nodes(structured)
        labels = [n['label'] for n in tree_select_nodes]
        assert "📁 src" in labels
        assert "📄 README.md" in labels

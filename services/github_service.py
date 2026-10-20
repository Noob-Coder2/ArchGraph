"""
GitHub API Service for ArchGraph (Production Decoupled Edition)
Provides:
1. Recursive Git Tree fetching (`fetch_repo_tree`) for instant UI tree preview (0 bytes of code downloaded).
2. Adaptive Tarball Stream (`fetch_repo_archive`) that streams .tar.gz in memory (1 HTTP request),
   filters lockfiles/binaries, extracts selected paths, and enforces max-file boundaries.
3. Concurrent raw file fetching (`fetch_multiple_files`) for small ad-hoc selections.
"""
import asyncio
import io
import os
import re
import tarfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set, Any
import httpx

MAX_REPO_FILES_DEFAULT = int(os.getenv("MAX_REPO_FILES", "1000"))
MAX_SINGLE_FILE_BYTES = int(os.getenv("MAX_SINGLE_FILE_BYTES", "256000"))  # 256 KB guard

IGNORED_DIRS: Set[str] = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", ".idea", ".vscode",
    "dist", "build", "out", "target", "vendor", ".next", ".nuxt", "coverage",
    ".mypy_cache", ".pytest_cache", ".tox", "migrations", "site-packages"
}

IGNORED_FILENAMES: Set[str] = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "uv.lock", "cargo.lock", "go.sum", "composer.lock", "gemfile.lock",
    "pipfile.lock", "bun.lockb"
}

BINARY_OR_NOISE_EXTS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".mp4", ".mp3", ".wav", ".ogg", ".mov", ".avi",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar",
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a",
    ".jar", ".war", ".class", ".lock", ".map", ".db", ".sqlite", ".sqlite3",
    ".parquet", ".feather", ".h5", ".pkl", ".pt", ".pth", ".onnx", ".safetensors"
}


class GitHubService:
    """Handles GitHub API interactions for repository and code exploration."""

    BASE_API_URL = "https://api.github.com"
    RAW_CONTENT_URL = "https://raw.githubusercontent.com"

    @classmethod
    def is_indexable_path(cls, path: str, size_bytes: Optional[int] = None) -> bool:
        """
        Production filter: Excludes lockfiles, minified files, binaries,
        ignored directories, and oversized source dumps.
        """
        norm = path.replace("\\", "/").strip("/")
        if not norm:
            return False

        parts = norm.split("/")
        filename = parts[-1].lower()

        if any(p in IGNORED_DIRS for p in parts[:-1]):
            return False

        if filename in IGNORED_FILENAMES or ".min." in filename or filename.endswith("-lock.json"):
            return False

        ext = Path(filename).suffix.lower()
        if ext in BINARY_OR_NOISE_EXTS:
            return False

        if size_bytes is not None and size_bytes > MAX_SINGLE_FILE_BYTES:
            return False

        return True

    @staticmethod
    def parse_github_url(url: str) -> Optional[Dict[str, str]]:
        """Parses a GitHub URL to extract owner, repo, branch, and sub-path."""
        try:
            url = url.strip().rstrip("/")
            patterns = [
                r"github\.com/([^/]+)/([^/]+)/tree/([^/]+)(?:/(.+))?",
                r"github\.com/([^/]+)/([^/]+)/blob/([^/]+)(?:/(.+))?",
                r"github\.com/([^/]+)/([^/]+)/?$",
            ]
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    groups = match.groups()
                    return {
                        "owner": groups[0],
                        "repo": groups[1].replace(".git", ""),
                        "branch": groups[2] if len(groups) > 2 and groups[2] else "main",
                        "path": groups[3] if len(groups) > 3 and groups[3] else "",
                    }
            return None
        except Exception:
            return None

    @classmethod
    def get_headers(cls, token: Optional[str] = None) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "ArchGraph-Intelligence"}
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"
        return headers

    @classmethod
    async def fetch_default_branch(cls, owner: str, repo: str, token: Optional[str] = None) -> str:
        """Fetches the default branch (main/master) of a repository."""
        url = f"{cls.BASE_API_URL}/repos/{owner}/{repo}"
        try:
            async with httpx.AsyncClient(headers=cls.get_headers(token), timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.json().get("default_branch", "main")
        except Exception:
            pass
        return "main"

    @classmethod
    async def fetch_repo_tree(
        cls, owner: str, repo: str, branch: str = "main", token: Optional[str] = None
    ) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """
        Fetches repository file tree metadata using Git Trees API (recursive).
        Downloads zero file content, returning just structure JSON.
        """
        url = f"{cls.BASE_API_URL}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        try:
            async with httpx.AsyncClient(headers=cls.get_headers(token), timeout=25.0) as client:
                resp = await client.get(url)
                if resp.status_code == 404 and branch == "main":
                    default_br = await cls.fetch_default_branch(owner, repo, token=token)
                    if default_br != branch:
                        resp = await client.get(
                            f"{cls.BASE_API_URL}/repos/{owner}/{repo}/git/trees/{default_br}?recursive=1"
                        )
                if resp.status_code == 404:
                    return None, f"Repository or branch not found: {owner}/{repo} ({branch})"
                if resp.status_code == 401:
                    return None, "Unauthorized access. Provide a valid GitHub Personal Access Token."
                if resp.status_code == 403:
                    return None, "GitHub API rate limit exceeded. Enter a Personal Access Token in Settings."
                resp.raise_for_status()

                data = resp.json()
                raw_tree = data.get("tree", [])
                filtered = [
                    item for item in raw_tree
                    if item.get("type") == "tree" or cls.is_indexable_path(item.get("path", ""), item.get("size"))
                ]
                warning = "Repository tree was truncated by GitHub (>100,000 files)." if data.get("truncated") else None
                return filtered, warning
        except httpx.TimeoutException:
            return None, "GitHub tree fetch timed out. Check network or repository size."
        except Exception as exc:
            return None, f"Failed to fetch tree: {str(exc)}"

    @classmethod
    async def fetch_repo_archive(
        cls,
        owner: str,
        repo: str,
        branch: str = "main",
        token: Optional[str] = None,
        selected_paths: Optional[List[str]] = None,
        max_files: int = MAX_REPO_FILES_DEFAULT,
    ) -> Tuple[List[Dict[str, str]], int, Optional[str]]:
        """
        Downloads the entire repository in 1 compressed tarball stream into RAM and extracts
        only indexable source files up to max_files.
        Returns: (extracted_files, skipped_count, error_message)
        where extracted_files is a list of {'filepath': ..., 'content': ...}.
        """
        url = f"{cls.BASE_API_URL}/repos/{owner}/{repo}/tarball/{branch}"
        selected_set = {p.replace("\\", "/").lstrip("/") for p in selected_paths} if selected_paths else None

        try:
            async with httpx.AsyncClient(
                headers=cls.get_headers(token),
                follow_redirects=True,
                timeout=50.0
            ) as client:
                resp = await client.get(url)
                if resp.status_code == 404 and branch == "main":
                    default_br = await cls.fetch_default_branch(owner, repo, token=token)
                    if default_br != branch:
                        resp = await client.get(f"{cls.BASE_API_URL}/repos/{owner}/{repo}/tarball/{default_br}")
                if resp.status_code == 401:
                    return [], 0, "Unauthorized: GitHub token invalid or lacking repo permissions."
                if resp.status_code == 403:
                    return [], 0, "GitHub rate limit exceeded. Provide a GitHub PAT."
                if resp.status_code == 404:
                    return [], 0, f"Repository not found: {owner}/{repo} ({branch})"
                resp.raise_for_status()
                tar_bytes = resp.content
        except Exception as exc:
            return [], 0, f"Tarball stream failed: {str(exc)}"

        def _extract_tar(raw_data: bytes) -> Tuple[List[Dict[str, str]], int, Optional[str]]:
            extracted: List[Dict[str, str]] = []
            skipped = 0
            CODE_PRIORITY = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".c", ".cpp", ".cs", ".md"}

            try:
                with tarfile.open(fileobj=io.BytesIO(raw_data), mode="r:gz") as tar:
                    members = [m for m in tar.getmembers() if m.isfile()]

                    def _rank(m: tarfile.TarInfo) -> Tuple[int, int]:
                        parts = m.name.split("/", 1)
                        rel = parts[1] if len(parts) > 1 else m.name
                        ext = Path(rel).suffix.lower()
                        return (0 if ext in CODE_PRIORITY else 1, rel.count("/"))

                    members.sort(key=_rank)

                    for member in members:
                        parts = member.name.split("/", 1)
                        if len(parts) < 2:
                            continue
                        rel_path = parts[1].replace("\\", "/").lstrip("/")

                        if selected_set is not None and rel_path not in selected_set:
                            continue
                        if not cls.is_indexable_path(rel_path, member.size):
                            skipped += 1
                            continue

                        f_stream = tar.extractfile(member)
                        if f_stream is None:
                            continue
                        content_bytes = f_stream.read()

                        # Guard against binary files disguised as source code
                        if b"\x00" in content_bytes[:2048]:
                            skipped += 1
                            continue

                        text_content = content_bytes.decode("utf-8", errors="replace")
                        extracted.append({"filepath": rel_path, "content": text_content})

                        if len(extracted) >= max_files:
                            break

                return extracted, skipped, None
            except Exception as ex:
                return [], 0, f"Tarball decompression error: {str(ex)}"

        return await asyncio.to_thread(_extract_tar, tar_bytes)

    @classmethod
    def build_tree_structure(cls, flat_tree: List[Dict]) -> List[Dict]:
        """Converts flat GitHub tree items into nested dictionary nodes for tree selection."""
        root_nodes: Dict[str, Any] = {}
        for item in flat_tree:
            parts = item["path"].split("/")
            curr = root_nodes
            for i, part in enumerate(parts):
                is_last = (i == len(parts) - 1)
                if part not in curr:
                    curr[part] = {
                        "label": part,
                        "value": item["path"] if is_last else "/".join(parts[: i + 1]),
                        "type": item["type"] if is_last else "tree",
                        "children": {} if (not is_last or item["type"] == "tree") else None,
                    }
                if not is_last and curr[part]["children"] is not None:
                    curr = curr[part]["children"]

        def _to_list(node_dict: Dict[str, Any]) -> List[Dict]:
            out = []
            for name, data in sorted(node_dict.items(), key=lambda x: (x[1]["type"] != "tree", x[0])):
                item = {
                    "label": data["label"],
                    "value": data["value"],
                    "type": "folder" if data["type"] == "tree" else "file",
                }
                if data["children"] is not None:
                    item["children"] = _to_list(data["children"])
                out.append(item)
            return out

        return _to_list(root_nodes)

    @classmethod
    def get_tree_select_nodes(cls, tree_structure: List[Dict]) -> List[Dict]:
        """Converts nested dictionary nodes into streamlit-tree-select format."""
        def _convert(n: Dict[str, Any]) -> Dict[str, Any]:
            res: Dict[str, Any] = {
                "label": f"📁 {n['label']}" if n["type"] == "folder" else f"📄 {n['label']}",
                "value": n["value"],
            }
            if n.get("children"):
                res["children"] = [_convert(c) for c in n["children"]]
            return res

        return [_convert(n) for n in tree_structure]
"""
GitHub API Service for DocuStream
Handles fetching repository trees, raw files, and Gists with optional PAT authentication.
Decoupled from Streamlit for reuse across API and Frontend layers.
"""
import io
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict, Any


class GitHubService:
    """Handles GitHub API interactions for repository and code exploration."""

    BASE_API_URL = "https://api.github.com"
    RAW_CONTENT_URL = "https://raw.githubusercontent.com"

    @staticmethod
    def parse_github_url(url: str) -> Optional[Dict[str, str]]:
        """Parse a GitHub URL to extract owner, repo, branch, and path."""
        try:
            url = url.strip().rstrip('/')
            patterns = [
                r'github\.com/([^/]+)/([^/]+)/tree/([^/]+)(?:/(.+))?',
                r'github\.com/([^/]+)/([^/]+)/blob/([^/]+)(?:/(.+))?',
                r'github\.com/([^/]+)/([^/]+)/?$',
            ]
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    groups = match.groups()
                    return {
                        'owner': groups[0],
                        'repo': groups[1].replace('.git', ''),
                        'branch': groups[2] if len(groups) > 2 and groups[2] else 'main',
                        'path': groups[3] if len(groups) > 3 and groups[3] else ''
                    }
            return None
        except Exception:
            return None

    @staticmethod
    def parse_gist_url(url: str) -> Optional[str]:
        """Extract Gist ID from a GitHub Gist URL."""
        match = re.search(r'gist\.github\.com/(?:[^/]+/)?([a-f0-9]+)', url)
        return match.group(1) if match else None

    @classmethod
    def get_headers(cls, token: Optional[str] = None) -> Dict[str, str]:
        """Generate GitHub headers with optional Bearer token."""
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "DocuStream-Pro"}
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"
        return headers

    @classmethod
    def fetch_default_branch(cls, owner: str, repo: str, token: Optional[str] = None) -> str:
        """Fetch the default branch of a repository."""
        url = f"{cls.BASE_API_URL}/repos/{owner}/{repo}"
        try:
            resp = requests.get(url, headers=cls.get_headers(token), timeout=10)
            resp.raise_for_status()
            return resp.json().get('default_branch', 'main')
        except Exception:
            return 'main'

    @classmethod
    def fetch_repo_tree(
        cls,
        owner: str,
        repo: str,
        branch: str = "main",
        token: Optional[str] = None
    ) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """Fetch repository file tree using Git Trees API (recursive)."""
        url = f"{cls.BASE_API_URL}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        try:
            resp = requests.get(url, headers=cls.get_headers(token), timeout=20)
            if resp.status_code == 404:
                return None, f"Repository or branch not found: {owner}/{repo} ({branch})"
            if resp.status_code == 401:
                return None, "Invalid GitHub token or unauthorized access."
            if resp.status_code == 403:
                return None, "GitHub API rate limit exceeded. Provide a GitHub Access Token in settings."
            resp.raise_for_status()
            data = resp.json()
            tree = data.get('tree', [])
            is_truncated = data.get('truncated', False)
            warning = "Repository tree was truncated by GitHub (>100,000 files)." if is_truncated else None
            return tree, warning
        except requests.exceptions.Timeout:
            return None, "GitHub API request timed out."
        except Exception as e:
            return None, f"Failed to fetch repository tree: {str(e)}"

    @classmethod
    def fetch_raw_file(
        cls,
        owner: str,
        repo: str,
        path: str,
        branch: str = "main",
        token: Optional[str] = None
    ) -> Tuple[Optional[io.BytesIO], Optional[str]]:
        """Fetch raw file contents as a BytesIO stream."""
        url = f"{cls.RAW_CONTENT_URL}/{owner}/{repo}/{branch}/{path}"
        headers = {}
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 404:
                return None, f"File not found: {path}"
            resp.raise_for_status()
            file_obj = io.BytesIO(resp.content)
            file_obj.name = path.split('/')[-1]
            return file_obj, None
        except Exception as e:
            return None, f"Error fetching {path}: {str(e)}"

    @classmethod
    def fetch_multiple_files(
        cls,
        owner: str,
        repo: str,
        paths: List[str],
        branch: str = "main",
        token: Optional[str] = None,
        max_workers: int = 5
    ) -> Tuple[List[io.BytesIO], List[str]]:
        """Concurrently fetch multiple files using a thread pool."""
        files = []
        errors = []

        def _fetch(p):
            return p, cls.fetch_raw_file(owner, repo, p, branch, token)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(_fetch, path): path for path in paths}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    _, (file_obj, err) = future.result()
                    if file_obj:
                        files.append(file_obj)
                    else:
                        errors.append(err or f"Failed to fetch {path}")
                except Exception as e:
                    errors.append(f"Error fetching {path}: {str(e)}")

        return files, errors

    @classmethod
    def fetch_gist(cls, gist_id: str, token: Optional[str] = None) -> Tuple[Optional[List[io.BytesIO]], Optional[str]]:
        """Fetch all files from a GitHub Gist."""
        url = f"{cls.BASE_API_URL}/gists/{gist_id}"
        try:
            resp = requests.get(url, headers=cls.get_headers(token), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            files_dict = data.get('files', {})
            result = []
            for fname, finfo in files_dict.items():
                content = finfo.get('content', '')
                fobj = io.BytesIO(content.encode('utf-8'))
                fobj.name = fname
                result.append(fobj)
            return result, None
        except Exception as e:
            return None, f"Error fetching Gist: {str(e)}"

    @classmethod
    def build_tree_structure(cls, flat_tree: List[Dict]) -> List[Dict]:
        """Transforms a flat GitHub tree into a nested dictionary structure."""
        root_nodes = {}
        for item in flat_tree:
            parts = item['path'].split('/')
            current = root_nodes
            for i, part in enumerate(parts):
                is_last = (i == len(parts) - 1)
                if part not in current:
                    current[part] = {
                        'label': part,
                        'value': item['path'] if is_last else '/'.join(parts[:i + 1]),
                        'type': item['type'] if is_last else 'tree',
                        'children': {} if (not is_last or item['type'] == 'tree') else None
                    }
                if not is_last:
                    current = current[part]['children']

        def _dict_to_list(node_dict):
            result = []
            for name, data in sorted(node_dict.items(), key=lambda x: (x[1]['type'] != 'tree', x[0])):
                item = {
                    'label': data['label'],
                    'value': data['value'],
                    'type': 'folder' if data['type'] == 'tree' else 'file'
                }
                if data['children'] is not None:
                    item['children'] = _dict_to_list(data['children'])
                result.append(item)
            return result

        return _dict_to_list(root_nodes)

    @classmethod
    def get_tree_select_nodes(cls, tree_structure: List[Dict]) -> List[Dict]:
        """Formats the tree for streamlit-tree-select."""
        def _convert(node):
            res = {
                'label': f"📁 {node['label']}" if node['type'] == 'folder' else f"📄 {node['label']}",
                'value': node['value']
            }
            if 'children' in node and node['children']:
                res['children'] = [_convert(c) for c in node['children']]
            return res

        return [_convert(n) for n in tree_structure]

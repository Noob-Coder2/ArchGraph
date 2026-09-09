"""
Utility functions for DocuStream WebApp
- Input validation and sanitization
- Rate limiting for API calls
- Common helper functions
"""
import re
import time
from typing import Optional, Tuple
from collections import deque
import hashlib
import requests
import io


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 60, time_window: int = 3600):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in time window
            time_window: Time window in seconds (default 1 hour)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
    
    def can_proceed(self) -> Tuple[bool, Optional[int]]:
        """
        Check if a new call can proceed.
        
        Returns:
            Tuple of (can_proceed, seconds_until_reset)
        """
        now = time.time()
        
        # Remove old calls outside the time window
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()
        
        if len(self.calls) < self.max_calls:
            return True, None
        else:
            # Calculate time until oldest call expires
            seconds_until_reset = int(self.time_window - (now - self.calls[0]))
            return False, seconds_until_reset
    
    def record_call(self):
        """Record a new API call."""
        self.calls.append(time.time())
    
    def get_remaining(self) -> int:
        """Get number of remaining calls in current window."""
        now = time.time()
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()
        return self.max_calls - len(self.calls)


def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate filename for security and compatibility.
    
    Args:
        filename: The filename to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filename:
        return False, "Filename cannot be empty"
    
    # Check length
    if len(filename) > 255:
        return False, "Filename too long (max 255 characters)"
    
    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        return False, "Filename cannot contain path separators or '..' sequences"
    
    # Check for invalid characters (Windows + Unix)
    invalid_chars = r'<>:"|?*\x00-\x1f'
    if any(char in filename for char in invalid_chars):
        return False, f"Filename contains invalid characters"
    
    # Check for reserved names (Windows)
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    name_without_ext = filename.rsplit('.', 1)[0].upper()
    if name_without_ext in reserved_names:
        return False, f"'{filename}' is a reserved system name"
    
    return True, None


def validate_file_size(size_bytes: int, max_size_mb: int = 50) -> Tuple[bool, Optional[str]]:
    """
    Validate file size.
    
    Args:
        size_bytes: Size in bytes
        max_size_mb: Maximum size in megabytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    max_bytes = max_size_mb * 1024 * 1024
    
    if size_bytes <= 0:
        return False, "Invalid file size"
    
    if size_bytes > max_bytes:
        return False, f"File too large (max {max_size_mb}MB)"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Replace invalid characters with underscore
    filename = re.sub(r'[<>:"|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = "unnamed_file"
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_len = 255 - len(ext) - 1 if ext else 255
        filename = name[:max_name_len] + ('.' + ext if ext else '')
    
    return filename


def compute_file_hash(file_obj, algorithm: str = 'sha256') -> str:
    """
    Compute hash of file content for deduplication.
    
    Args:
        file_obj: File-like object
        algorithm: Hash algorithm (default: sha256)
        
    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)
    
    # Save current position
    original_pos = file_obj.tell() if hasattr(file_obj, 'tell') else 0
    
    # Read and hash content
    file_obj.seek(0)
    if hasattr(file_obj, 'getvalue'):
        hasher.update(file_obj.getvalue())
    else:
        chunk_size = 8192
        while chunk := file_obj.read(chunk_size):
            if isinstance(chunk, str):
                chunk = chunk.encode('utf-8')
            hasher.update(chunk)
    
    # Restore position
    file_obj.seek(original_pos)
    
    return hasher.hexdigest()


def format_file_size(bytes_size: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.1f} KB"
    elif bytes_size < 1024 ** 3:
        return f"{bytes_size / (1024 ** 2):.1f} MB"
    else:
        return f"{bytes_size / (1024 ** 3):.1f} GB"


def detect_file_encoding(file_obj) -> str:
    """
    Detect file encoding by reading the beginning of the file.
    
    Args:
        file_obj: File-like object
        
    Returns:
        Detected encoding (default: 'utf-8')
    """
    # Save position
    original_pos = file_obj.tell() if hasattr(file_obj, 'tell') else 0
    
    file_obj.seek(0)
    sample = file_obj.read(min(1024, len(file_obj.getvalue()) if hasattr(file_obj, 'getvalue') else 1024))
    file_obj.seek(original_pos)
    
    # Try UTF-8 first
    try:
        if isinstance(sample, bytes):
            sample.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        pass
    
    # Try common encodings
    for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
        try:
            if isinstance(sample, bytes):
                sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    
    return 'utf-8'  # Default fallback


def fetch_from_github(url):
    """Fetches raw content from a GitHub URL and returns a BytesIO object."""
    try:
        # Basic URL Validation
        if not url.startswith("http"):
            return None, "Invalid URL format."
        
        # Security: Validate that URL is from GitHub domains only (prevent SSRF)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        allowed_domains = ['github.com', 'raw.githubusercontent.com', 'gist.github.com']
        if not any(parsed.netloc == domain or parsed.netloc.endswith('.' + domain) for domain in allowed_domains):
            return None, "Invalid URL. Only GitHub URLs are allowed (github.com, raw.githubusercontent.com, gist.github.com)."
            
        # Convert 'blob' or 'tree' (if user mistakes folder for file, though tree support needs API) to raw
        # Case 1: Standard Blob URL -> Convert to Raw
        # Ex: https://github.com/user/repo/blob/main/file.txt -> https://raw.githubusercontent.com/user/repo/main/file.txt
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            
        # Case 2: User provided a 'tree' URL (Folder) -> Not supported yet, fail gracefully
        if "/tree/" in url:
            return None, "Folder import not supported. Please provide a file URL."

        # Case 3: Raw URL provided directly -> Use as is
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Validation: Check if we got an HTML page (likely an error or wrong URL type for a code file)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type and not url.endswith('.html'):
            return None, "URL returned a webpage, not a file. Please ensure you are using a direct file link."

        filename = url.split("/")[-1]
        file_obj = io.BytesIO(response.content)
        file_obj.name = filename
        return file_obj, None # Success
        
    except requests.exceptions.HTTPError:
        return None, "File not found (404) or private repo."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except Exception as e:
        return None, f"Error: {str(e)}"

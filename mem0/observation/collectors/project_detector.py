"""
Project detection for ECC-style project isolation.

Uses Git remote URL to generate portable project identifiers.
"""

import hashlib
import subprocess
from pathlib import Path
from typing import Optional

from ..models import ProjectInfo


class ProjectDetector:
    """
    Detect project information using Git metadata.

    ECC-style project detection:
    - Hash Git remote URL for portable project_id
    - Extract project name from remote
    - Track current branch
    """

    def __init__(self, working_dir: Optional[str | Path] = None):
        """
        Initialize detector.

        Args:
            working_dir: Git repository directory (default: current dir)
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

    def detect(self) -> ProjectInfo:
        """
        Detect project information.

        Returns:
            ProjectInfo with detected metadata
        """
        remote_url = self._get_git_remote_url()
        branch = self._get_git_branch()

        # Generate portable project_id from remote URL
        if remote_url:
            project_id = self._hash_remote_url(remote_url)
            project_name = self._extract_project_name(remote_url)
        else:
            # Fallback for non-Git projects
            project_id = self._hash_path(self.working_dir)
            project_name = self.working_dir.name

        return ProjectInfo(
            project_id=project_id,
            project_name=project_name,
            git_remote_url=remote_url,
            git_branch=branch,
        )

    def _get_git_remote_url(self) -> Optional[str]:
        """Get Git remote URL."""
        try:
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                # Normalize URL (remove .git suffix, credentials)
                url = result.stdout.strip()
                url = url.removesuffix(".git")

                # Remove credentials for privacy
                if "://" in url:
                    # https://user:pass@github.com/... -> https://github.com/...
                    parts = url.split("://")
                    if "@" in parts[1]:
                        _, clean_host = parts[1].split("@", 1)
                        url = f"{parts[0]}://{clean_host}"

                return url

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def _get_git_branch(self) -> Optional[str]:
        """Get current Git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return result.stdout.strip() or None

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def _hash_remote_url(self, remote_url: str) -> str:
        """
        Generate stable hash from remote URL.

        Uses SHA-256 truncated to 16 characters for readability.
        """
        return hashlib.sha256(remote_url.encode()).hexdigest()[:16]

    def _hash_path(self, path: Path) -> str:
        """Generate hash from filesystem path (fallback)."""
        return hashlib.sha256(str(path.absolute()).encode()).hexdigest()[:16]

    def _extract_project_name(self, remote_url: str) -> str:
        """Extract project name from remote URL."""
        # Handle various Git URL formats
        # https://github.com/user/repo -> repo
        # git@github.com:user/repo -> repo

        # Remove protocol
        for prefix in ["https://", "http://", "git@", "ssh://"]:
            if remote_url.startswith(prefix):
                remote_url = remote_url[len(prefix):]
                break

        # Get last part (repository name)
        parts = remote_url.split("/")
        return parts[-1] if parts else remote_url

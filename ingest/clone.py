import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from git import Repo, GitCommandError


@dataclass
class CloneResult:
    repo_url: str
    local_path: Path
    was_cloned: bool
    did_update: bool


def _parse_github_url(repo_url: str) -> Tuple[str, str]:
    """
    Parse a GitHub repo URL and return (owner, repo).

    Supported:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - git@github.com:owner/repo.git
    """
    repo_url = repo_url.strip()

    # HTTPS
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+?)(\.git)?/?$", repo_url)
    if m:
        owner, repo = m.group(1), m.group(2)
        return owner, repo

    # SSH
    m = re.match(r"^git@github\.com:([^/]+)/([^/]+?)(\.git)?$", repo_url)
    if m:
        owner, repo = m.group(1), m.group(2)
        return owner, repo

    raise ValueError(f"Unsupported GitHub URL format: {repo_url}")


def clone_repo(repo_url: str, base_dir: str = "data/repos", update_if_exists: bool = False) -> CloneResult:
    """
    Clone a GitHub repository into base_dir/owner__repo.

    - If repo already exists:
        - update_if_exists=False => reuse existing clone
        - update_if_exists=True  => git pull
    """
    owner, repo = _parse_github_url(repo_url)
    target_dir = Path(base_dir) / f"{owner}__{repo}"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and (target_dir / ".git").exists():
        # Already cloned
        if update_if_exists:
            try:
                r = Repo(str(target_dir))
                r.remotes.origin.pull()
                return CloneResult(repo_url, target_dir, was_cloned=False, did_update=True)
            except GitCommandError as e:
                raise RuntimeError(f"Failed to pull latest changes: {e}") from e

        return CloneResult(repo_url, target_dir, was_cloned=False, did_update=False)

    # Fresh clone
    try:
        Repo.clone_from(repo_url, str(target_dir))
        return CloneResult(repo_url, target_dir, was_cloned=True, did_update=False)
    except GitCommandError as e:
        raise RuntimeError(f"Failed to clone repository: {e}") from e

from __future__ import annotations

import hashlib
from pathlib import Path


def repo_id_from_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def index_dir_for(url: str) -> Path:
    return Path("data/indexes") / repo_id_from_url(url)

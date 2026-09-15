"""On-disk cache of built graphs."""

import hashlib
from pathlib import Path

import torch

from .config import FEATURE_VERSION, GRAPH_PAYLOAD_VERSION
from .logging_utils import get_logger

log = get_logger("cache")


def _fingerprint(path):
    resolved = Path(path).resolve()
    if resolved.exists():
        return f"{resolved}|{resolved.stat().st_size}"
    return str(resolved)


def cache_key(input_paths, params, extra=""):
    """Deterministic identifier for a graph build."""
    parts = [_fingerprint(path) for path in input_paths]
    parts.append(
        "|".join(
            f"{key}={value}" for key, value in sorted(params.to_dict().items())
        )
    )
    parts.append(f"fv={FEATURE_VERSION}|payload={GRAPH_PAYLOAD_VERSION}")
    if extra:
        parts.append(extra)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


class GraphCache:
    """Load/store graphs under a directory; a directory of ``None`` disables it."""

    def __init__(self, directory):
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self):
        return self.directory is not None

    def path_for(self, key):
        return self.directory / f"graph_{key}.pt" if self.directory else None

    def load(self, key):
        path = self.path_for(key)
        if path is None or not path.exists():
            return None
        try:
            graph = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            log.warning("Ignoring unreadable cache entry %s: %s", path.name, exc)
            return None
        log.info("Graph loaded from cache: %s", path.name)
        return graph

    def store(self, key, graph):
        path = self.path_for(key)
        if path is None:
            return None
        temporary = path.with_suffix(f".tmp{id(graph):x}")
        try:
            torch.save(graph, temporary)
            temporary.replace(path)
        except OSError as exc:
            log.warning("Could not write cache entry %s: %s", path.name, exc)
            temporary.unlink(missing_ok=True)
            return None
        log.debug("Graph cached: %s", path.name)
        return path

"""On-disk cache of built graphs."""

import hashlib
import pickle
from importlib import import_module
from pathlib import Path

import torch

from .config import FEATURE_VERSION, GRAPH_PAYLOAD_VERSION
from .logging_utils import get_logger

log = get_logger("cache")

# A cache directory can be shared between users, or arrive with a dataset, so
# entries are read with weights_only=True and cannot execute pickled code. That
# allowlist covers tensors and primitives only, so the classes a cached PyG
# graph is legitimately built from have to be declared here; anything else in
# an entry still makes the load fail, and the graph is rebuilt instead.
_SAFE_GLOBALS = (
    "torch_geometric.data.data:Data",
    "torch_geometric.data.data:DataEdgeAttr",
    "torch_geometric.data.data:DataTensorAttr",
    "torch_geometric.data.storage:GlobalStorage",
    "numpy.core.multiarray:scalar",
    "numpy._core.multiarray:scalar",
    "numpy:dtype",
)

_safe_globals_registered = False


def _register_safe_globals():
    """Allowlist the cached-graph classes once per process.

    Names that this torch or torch-geometric does not have are skipped: the
    set moved between versions, and a missing one only costs a cache miss.
    """
    global _safe_globals_registered
    if _safe_globals_registered:
        return
    _safe_globals_registered = True

    register = getattr(torch.serialization, "add_safe_globals", None)
    if register is None:  # torch < 2.4
        return

    resolved = []
    for entry in _SAFE_GLOBALS:
        module_name, _, attribute = entry.partition(":")
        try:
            resolved.append(getattr(import_module(module_name), attribute))
        except (ImportError, AttributeError, DeprecationWarning):
            continue
    if resolved:
        register(resolved)


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
        _register_safe_globals()
        try:
            graph = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            log.warning(
                "Ignoring cache entry %s: it contains objects that are not safe to "
                "unpickle. Rebuilding the graph.", path.name,
            )
            return None
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

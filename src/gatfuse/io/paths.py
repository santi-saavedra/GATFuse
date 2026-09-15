"""Input-path validation shared by the parsers."""

from pathlib import Path

from ..errors import InputError


def require_file(path, description):
    """Return *path* as a Path, raising InputError if it is not a readable file."""
    if path is None or str(path) == "":
        raise InputError(f"No {description} was provided.")
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise InputError(f"{description} not found: {resolved}")
    if not resolved.is_file():
        raise InputError(f"{description} is not a regular file: {resolved}")
    if resolved.stat().st_size == 0:
        raise InputError(f"{description} is empty: {resolved}")
    return resolved


def prepare_output(path, description="output file"):
    """Create the parent directory of *path* and return it as a Path."""
    resolved = Path(path).expanduser()
    parent = resolved.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Cannot create directory for {description} ({parent}): {exc}") from exc
    return resolved

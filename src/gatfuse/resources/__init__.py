"""Model and reference data distributed with GATFuse."""

from importlib import resources

from ..errors import InputError

BUNDLED_MODEL = "gatfuse-v1.pt"
BUNDLED_KNOWN_FUSIONS = "mitelman_fusions.tsv"


def _resolve(package, filename, description):
    try:
        path = resources.files(f"gatfuse.resources.{package}").joinpath(filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as exc:
        raise InputError(
            f"The bundled {description} is not available in this installation. "
            "Pass it explicitly on the command line."
        ) from exc


def bundled_model_path():
    """Path to the pretrained model distributed with GATFuse."""
    return _resolve("models", BUNDLED_MODEL, "model")


def bundled_known_fusions_path():
    """Path to the bundled known-fusion catalogue."""
    return _resolve("known_fusions", BUNDLED_KNOWN_FUSIONS, "known-fusion catalogue")

"""STAR per-gene read counts (``ReadsPerGene.out.tab``)."""

import pandas as pd

from ..config import LIBRARY_TYPES
from ..errors import InputError
from ..logging_utils import get_logger
from .paths import require_file

log = get_logger("io.expression")

_COLUMNS = ["gene_id", "unstranded", "stranded_forward", "stranded_reverse"]


def load_expression(path, library_type="unstranded"):
    """Return a (gene_id, count) table for the requested library strandedness."""
    if library_type not in LIBRARY_TYPES:
        raise InputError(
            f"Unknown library type '{library_type}'. Choose from: {', '.join(LIBRARY_TYPES)}"
        )
    path = require_file(path, "STAR gene count file")

    try:
        df = pd.read_csv(path, sep="\t", names=_COLUMNS, header=None,
                         dtype={"gene_id": "string"})
    except Exception as exc:
        raise InputError(f"Could not parse gene count file {path}: {exc}") from exc

    df = df[~df["gene_id"].fillna("").str.startswith("N_")].copy()
    if df.empty:
        raise InputError(
            f"No gene rows in {path}. Was STAR run with --quantMode GeneCounts?"
        )

    counts = df[["gene_id", library_type]].rename(columns={library_type: "count"})
    counts["count"] = pd.to_numeric(counts["count"], errors="coerce").fillna(0)
    counts = counts.reset_index(drop=True)

    expressed = int((counts["count"] > 0).sum())
    log.info("Gene counts: %d genes (%d expressed), library type '%s'",
             len(counts), expressed, library_type)
    if expressed == 0:
        raise InputError(
            f"All gene counts are zero in {path} for library type '{library_type}'. "
            "Check that --library-type matches the protocol used."
        )
    return counts

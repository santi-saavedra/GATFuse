"""STAR chimeric-junction parsing (``Chimeric.out.junction``)."""

import re

import pandas as pd

from ..errors import InputError
from ..logging_utils import get_logger
from .paths import require_file

log = get_logger("io.chimeric")

_COLUMNS = [
    "chr_donor", "brkpt_donor", "strand_donor",
    "chr_acceptor", "brkpt_acceptor", "strand_acceptor",
    "junction_type", "repeat_left_len", "repeat_right_len",
    "read_name", "aln_1_start", "aln_1_cigar", "aln_2_start", "aln_2_cigar",
]

_CIGAR_MATCH = re.compile(r"(\d+)M")


def _cigar_match_length(cigar):
    """Total length of the M operations in a CIGAR string."""
    if not isinstance(cigar, str) or not cigar:
        return 0
    return sum(int(n) for n in _CIGAR_MATCH.findall(cigar))


def load_chimeric_junctions(path, min_split_reads=2):
    """Return one row per chimeric junction supported by >= *min_split_reads* reads.

    Columns: chr_donor, brkpt_donor, strand_donor, chr_acceptor, brkpt_acceptor,
    strand_acceptor, junction_type, num_split_reads, repeat_left_len,
    repeat_right_len, max_balanced_anchor.
    """
    path = require_file(path, "STAR chimeric junction file")

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            names=_COLUMNS,
            comment="#",
            header=None,
            dtype={"chr_donor": "string", "chr_acceptor": "string",
                   "strand_donor": "string", "strand_acceptor": "string"},
            low_memory=False,
        )
    except Exception as exc:
        raise InputError(f"Could not parse chimeric junction file {path}: {exc}") from exc

    # STAR >= 2.7.1 writes a column header as the first line when
    # --chimOutJunctionFormat 1 is used; it is not comment-prefixed.
    df = df[pd.to_numeric(df["brkpt_donor"], errors="coerce").notna()].copy()
    if df.empty:
        log.warning("No chimeric junctions in %s", path.name)
        return _empty_junction_frame()

    for column in ("brkpt_donor", "brkpt_acceptor", "junction_type",
                   "repeat_left_len", "repeat_right_len"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["brkpt_donor", "brkpt_acceptor", "junction_type"])
    df["brkpt_donor"] = df["brkpt_donor"].astype("int64")
    df["brkpt_acceptor"] = df["brkpt_acceptor"].astype("int64")
    df["junction_type"] = df["junction_type"].astype("int64")
    df["repeat_left_len"] = df["repeat_left_len"].fillna(0)
    df["repeat_right_len"] = df["repeat_right_len"].fillna(0)

    left_anchor = df["aln_1_cigar"].map(_cigar_match_length)
    right_anchor = df["aln_2_cigar"].map(_cigar_match_length)
    df["balanced_anchor"] = pd.concat([left_anchor, right_anchor], axis=1).min(axis=1)

    junctions = (
        df.groupby(
            ["chr_donor", "brkpt_donor", "strand_donor",
             "chr_acceptor", "brkpt_acceptor", "strand_acceptor"],
            observed=True,
        )
        .agg(
            num_split_reads=("read_name", "count"),
            junction_type=("junction_type", "first"),
            repeat_left_len=("repeat_left_len", "mean"),
            repeat_right_len=("repeat_right_len", "mean"),
            max_balanced_anchor=("balanced_anchor", "max"),
        )
        .reset_index()
    )
    junctions = (
        junctions[junctions["num_split_reads"] >= min_split_reads]
        .sort_values("num_split_reads", ascending=False)
        .reset_index(drop=True)
    )

    log.info(
        "Chimeric junctions: %d with >= %d split reads (from %d chimeric alignments)",
        len(junctions), min_split_reads, len(df),
    )
    if junctions.empty:
        log.warning(
            "No junction reaches --min-split-reads %d; lower it to recover low-coverage candidates.",
            min_split_reads,
        )
    return junctions


def _empty_junction_frame():
    return pd.DataFrame(
        {
            "chr_donor": pd.Series(dtype="string"),
            "brkpt_donor": pd.Series(dtype="int64"),
            "strand_donor": pd.Series(dtype="string"),
            "chr_acceptor": pd.Series(dtype="string"),
            "brkpt_acceptor": pd.Series(dtype="int64"),
            "strand_acceptor": pd.Series(dtype="string"),
            "num_split_reads": pd.Series(dtype="int64"),
            "junction_type": pd.Series(dtype="int64"),
            "repeat_left_len": pd.Series(dtype="float64"),
            "repeat_right_len": pd.Series(dtype="float64"),
            "max_balanced_anchor": pd.Series(dtype="int64"),
        }
    )

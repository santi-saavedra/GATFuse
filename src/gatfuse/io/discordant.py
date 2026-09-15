"""Inter-chromosomal discordant read pairs from a coordinate-sorted BAM."""

import pandas as pd

from ..errors import InputError
from ..logging_utils import get_logger
from .paths import require_file

log = get_logger("io.discordant")

_PROGRESS_INTERVAL = 20_000_000


def load_discordant_pairs(path, threads=1):
    """Return one row per inter-chromosomal discordant pair.

        Args:
            path: coordinate-sorted BAM.
            threads: BGZF decompression threads. Decompression dominates the scan,
                so 4-8 threads give a near-linear speed-up.

        Columns: chr_A, pos_A, chr_B, pos_B.
    """
    try:
        from pysam import AlignmentFile
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise InputError(
            "pysam is required to read BAM files. Install it with 'pip install pysam'."
        ) from exc

    path = require_file(path, "alignment BAM")

    chr_a, pos_a, chr_b, pos_b = [], [], [], []
    scanned = 0
    try:
        with AlignmentFile(str(path), "rb", threads=max(1, int(threads))) as bam:
            for read in bam:
                scanned += 1
                if scanned % _PROGRESS_INTERVAL == 0:
                    log.debug("Scanned %d million alignments...", scanned // 1_000_000)
                if read.is_unmapped or not read.is_paired or read.mate_is_unmapped:
                    continue
                if not read.is_read1:
                    continue
                reference = read.reference_name
                mate = read.next_reference_name
                if reference and mate and reference != mate:
                    chr_a.append(reference)
                    pos_a.append(read.reference_start)
                    chr_b.append(mate)
                    pos_b.append(read.next_reference_start)
    except ValueError as exc:
        raise InputError(f"Could not read BAM {path}: {exc}") from exc

    pairs = pd.DataFrame(
        {
            "chr_A": pd.array(chr_a, dtype="string"),
            "pos_A": pd.array(pos_a, dtype="int64"),
            "chr_B": pd.array(chr_b, dtype="string"),
            "pos_B": pd.array(pos_b, dtype="int64"),
        }
    )
    log.info("Discordant pairs: %d inter-chromosomal (from %d alignments)", len(pairs), scanned)
    return pairs

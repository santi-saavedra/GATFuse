"""Known-fusion catalogue used for the optional prior-knowledge features."""

import csv

from ..errors import InputError
from ..logging_utils import get_logger
from .paths import require_file

log = get_logger("io.known_fusions")


class KnownFusionCatalogue:
    """Gene-level and pair-level lookups over a catalogue of reported fusions."""

    def __init__(self, pair_counts, partners, source=""):
        self._pair_counts = pair_counts     # frozenset({a, b}) -> case count
        self._partners = partners           # gene -> set of partner genes
        self.source = source

    @property
    def genes(self):
        return self._partners.keys()

    def is_fusion_gene(self, symbol):
        return symbol in self._partners

    def pair_count(self, gene_a, gene_b):
        """Reported case count for the pair, 0 if it is not catalogued."""
        return self._pair_counts.get(frozenset({gene_a, gene_b}), 0)

    def partner_count(self, symbol):
        return len(self._partners.get(symbol, ()))

    def __len__(self):
        return len(self._pair_counts)

    def __repr__(self):
        return (
            f"KnownFusionCatalogue(genes={len(self._partners)}, pairs={len(self._pair_counts)})"
        )


def load_known_fusions(path, col_a="gene_a", col_b="gene_b", count_col="n_cases",
                       use_recurrence=True):
    """Parse a known-fusion table.

    Args:
        path: TSV/CSV file; the delimiter is detected automatically.
        col_a, col_b: partner column names (matched case-insensitively).
        count_col: optional recurrence column. A missing column falls back to
            one case per row, so a bare two-column list is valid input.
        use_recurrence: when False the count column is ignored and every
            catalogued pair counts once.
    """
    path = require_file(path, "known-fusion catalogue")

    with open(path, encoding="utf-8") as handle:
        sample = handle.read(4096)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        delimiter = "\t"

    pair_counts = {}
    partners = {}

    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter=delimiter
        )
        if not reader.fieldnames:
            raise InputError(f"Known-fusion catalogue has no header row: {path}")

        lowered = [name.strip().lower() for name in reader.fieldnames]

        def resolve(name, required=True):
            if name and name.lower() in lowered:
                return reader.fieldnames[lowered.index(name.lower())]
            if required:
                raise InputError(
                    f"Column '{name}' not found in {path}. Columns present: {reader.fieldnames}"
                )
            return None

        key_a = resolve(col_a)
        key_b = resolve(col_b)
        key_count = resolve(count_col, required=False) if use_recurrence else None

        for row in reader:
            gene_a = (row.get(key_a) or "").strip()
            gene_b = (row.get(key_b) or "").strip()
            if not gene_a or not gene_b:
                continue

            count = 1
            if key_count:
                try:
                    count = max(1, int(float(row.get(key_count) or 1)))
                except (TypeError, ValueError):
                    count = 1

            pair_counts[frozenset({gene_a, gene_b})] = (
                pair_counts.get(frozenset({gene_a, gene_b}), 0) + count
            )
            partners.setdefault(gene_a, set()).add(gene_b)
            partners.setdefault(gene_b, set()).add(gene_a)

    if not pair_counts:
        raise InputError(f"No fusion pairs could be read from {path}")

    catalogue = KnownFusionCatalogue(pair_counts, partners, source=str(path))
    log.info("Known-fusion catalogue: %d genes, %d unique pairs",
             len(partners), len(pair_counts))
    return catalogue

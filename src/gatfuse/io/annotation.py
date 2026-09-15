"""Reference annotation (GTF) parsing."""

from dataclasses import dataclass

import pandas as pd

from ..config import BIOTYPE_ATTRIBUTE_MODES
from ..errors import InputError
from ..logging_utils import get_logger
from .paths import require_file

log = get_logger("io.annotation")

# GTF is a 9-column format; only these are needed downstream.
_GTF_COLUMNS = {0: "chr", 2: "feature", 3: "start", 4: "end", 6: "strand", 8: "attribute"}


@dataclass
class Annotation:
    """Gene- and exon-level views of a reference annotation.

    genes: chr, start, end, strand, gene_id, gene_name, gene_biotype, gene_length
    exons: chr, start, end, gene_id
    """

    genes: pd.DataFrame
    exons: pd.DataFrame
    source: str = ""

    @property
    def has_biotype(self):
        return bool(self.genes["gene_biotype"].notna().any())


def _extract(series, key):
    return series.str.extract(rf'{key} "([^"]+)"', expand=False)


def load_annotation(gtf_path, biotype_attributes="ensembl"):
    """Parse a GTF (optionally gzipped) into gene and exon tables.

    Args:
        gtf_path: path to the GTF used for the STAR alignment.
        biotype_attributes: which attribute names carry the gene biotype;
            see ``config.BIOTYPE_ATTRIBUTE_MODES``.
    """
    path = require_file(gtf_path, "annotation GTF")
    attributes = BIOTYPE_ATTRIBUTE_MODES.get(biotype_attributes)
    if attributes is None:
        raise InputError(f"Unknown biotype attribute mode: {biotype_attributes}")

    log.info("Reading annotation: %s", path)
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            header=None,
            usecols=list(_GTF_COLUMNS),
            names=list(_GTF_COLUMNS.values()),
            dtype={"chr": "string", "feature": "string", "strand": "string", "attribute": "string"},
            low_memory=False,
        )
    except Exception as exc:
        raise InputError(f"Could not parse GTF {path}: {exc}") from exc

    if df.empty:
        raise InputError(f"Annotation GTF contains no records: {path}")

    genes = df[df["feature"] == "gene"].copy()
    if genes.empty:
        raise InputError(
            f"No 'gene' records found in {path}. GATFuse needs a GTF with gene features "
            "(the same annotation supplied to STAR)."
        )

    genes["gene_id"] = _extract(genes["attribute"], "gene_id")
    genes["gene_name"] = _extract(genes["attribute"], "gene_name")

    biotype = None
    for attribute in attributes:
        extracted = _extract(genes["attribute"], attribute)
        if extracted.notna().any():
            biotype = extracted
            log.debug("Gene biotype read from attribute '%s'", attribute)
            break
    if biotype is None:
        biotype = pd.Series(pd.NA, index=genes.index, dtype="string")
        alternatives = _detect_alternative_biotype_attribute(genes["attribute"], attributes)
        if alternatives:
            log.warning(
                "No '%s' attribute in %s, but '%s' is present. The protein-coding node feature "
                "will be constant 0 and no coding-gene preference is applied when assigning "
                "breakpoints to genes. Use --biotype-attributes auto to read it (see "
                "docs/reproducibility.md before doing so with the distributed model).",
                "'/'".join(attributes), path.name, alternatives,
            )
        else:
            log.warning(
                "No gene biotype attribute found in %s; the protein-coding feature will be 0.",
                path.name,
            )

    genes["gene_biotype"] = biotype
    genes["gene_length"] = genes["end"] - genes["start"] + 1
    genes = genes[
        ["chr", "start", "end", "strand", "gene_id", "gene_name", "gene_biotype", "gene_length"]
    ].reset_index(drop=True)

    exons = df[df["feature"] == "exon"].copy()
    exons["gene_id"] = _extract(exons["attribute"], "gene_id")
    exons = exons[["chr", "start", "end", "gene_id"]].reset_index(drop=True)

    log.info("Annotation loaded: %d genes, %d exons", len(genes), len(exons))
    if genes["gene_name"].isna().all():
        log.warning(
            "No gene_name attributes in %s; results will report Ensembl gene IDs instead of symbols.",
            path.name,
        )
    return Annotation(genes=genes, exons=exons, source=str(path))


def _detect_alternative_biotype_attribute(attribute_series, searched):
    """Report a biotype-like attribute present in the file but not searched for."""
    sample = attribute_series.head(200).dropna()
    for candidate in ("gene_type", "gene_biotype"):
        if candidate in searched:
            continue
        if sample.str.contains(f"{candidate} ", regex=False).any():
            return candidate
    return None

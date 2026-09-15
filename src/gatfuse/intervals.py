"""Sorted interval indices for mapping genomic positions onto genes and exons."""

import numpy as np


class GeneIntervalIndex:
    """Genes grouped by chromosome and sorted by start coordinate."""

    def __init__(self, genes):
        self._by_chrom = {}
        self._has_biotype = bool(genes["gene_biotype"].notna().any())
        for chrom, group in genes.groupby("chr", sort=False, observed=True):
            starts = group["start"].to_numpy(dtype=np.int64)
            order = np.argsort(starts, kind="stable")
            coding = (
                group["gene_biotype"].to_numpy(dtype=object)[order] == "protein_coding"
                if self._has_biotype
                else np.ones(len(order), dtype=bool)
            )
            self._by_chrom[str(chrom)] = (
                starts[order],
                group["end"].to_numpy(dtype=np.int64)[order],
                group["gene_id"].to_numpy(dtype=object)[order],
                coding,
            )

    @property
    def contigs(self):
        return set(self._by_chrom)

    def overlapping(self, chrom, position, prefer_coding=True):
        """Gene IDs covering (chrom, position)."""
        entry = self._by_chrom.get(chrom)
        if entry is None:
            return []
        starts, ends, gene_ids, coding = entry
        hi = np.searchsorted(starts, position, side="right")
        if hi == 0:
            return []
        overlaps = ends[:hi] >= position
        if not overlaps.any():
            return []
        if prefer_coding:
            coding_hits = overlaps & coding[:hi]
            if coding_hits.any():
                overlaps = coding_hits
        return list(dict.fromkeys(gene_ids[:hi][overlaps].tolist()))


class ExonIntervalIndex:
    """Exons indexed by gene, and (lazily) by chromosome."""

    def __init__(self, exons):
        self._by_gene = {}
        self._by_chrom = None
        self._exons = exons

        if exons is None or exons.empty:
            return
        ordered = exons.sort_values(["gene_id", "start", "end"], kind="stable")
        gene_ids = ordered["gene_id"].to_numpy(dtype=object)
        starts = ordered["start"].to_numpy(dtype=np.int64)
        ends = ordered["end"].to_numpy(dtype=np.int64)
        if len(gene_ids) == 0:
            return
        boundaries = np.flatnonzero(gene_ids[1:] != gene_ids[:-1]) + 1
        for begin, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(gene_ids)], strict=True):
            self._by_gene[gene_ids[begin]] = (starts[begin:stop], ends[begin:stop])

    def covers(self, gene_id, position):
        """True if an exon of *gene_id* contains *position*."""
        entry = self._by_gene.get(gene_id)
        if entry is None:
            return False
        starts, ends = entry
        hi = np.searchsorted(starts, position, side="right")
        return bool(hi and (ends[:hi] >= position).any())

    def _chrom_index(self):
        if self._by_chrom is None:
            self._by_chrom = {}
            if self._exons is not None and not self._exons.empty:
                for chrom, group in self._exons.groupby("chr", sort=False, observed=True):
                    starts = group["start"].to_numpy(dtype=np.int64)
                    order = np.argsort(starts, kind="stable")
                    self._by_chrom[str(chrom)] = (
                        starts[order],
                        group["end"].to_numpy(dtype=np.int64)[order],
                        group["gene_id"].to_numpy(dtype=object)[order],
                    )
        return self._by_chrom

    def genes_with_exon_at(self, chrom, position):
        """Gene IDs having an exon over (chrom, position)."""
        entry = self._chrom_index().get(chrom)
        if entry is None:
            return np.empty(0, dtype=object)
        starts, ends, gene_ids = entry
        hi = np.searchsorted(starts, position, side="right")
        if hi == 0:
            return np.empty(0, dtype=object)
        overlaps = ends[:hi] >= position
        return gene_ids[:hi][overlaps]


def annotate_region(gene_index, exon_index, chrom, position):
    """Classify a breakpoint as 'exon', 'intron' or 'intergenic'."""
    genes = gene_index.overlapping(chrom, position, prefer_coding=False)
    if not genes:
        return "intergenic"
    exonic = exon_index.genes_with_exon_at(chrom, position)
    if len(exonic) and np.isin(exonic, np.asarray(genes, dtype=object)).any():
        return "exon"
    return "intron"

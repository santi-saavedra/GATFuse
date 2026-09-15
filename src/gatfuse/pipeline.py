"""Turning a set of aligner outputs into a graph, with caching."""

from dataclasses import dataclass
from pathlib import Path

from .cache import GraphCache, cache_key
from .config import GraphParams
from .graph import build_graph
from .io import load_annotation, load_chimeric_junctions, load_discordant_pairs, load_expression
from .logging_utils import get_logger

log = get_logger("pipeline")


@dataclass
class SampleInputs:
    """The four files describing one RNA-seq sample."""

    bam: Path
    chimeric_junctions: Path
    gene_counts: Path
    annotation: Path
    sample_id: str = "sample"

    def paths(self):
        return [self.bam, self.chimeric_junctions, self.gene_counts, self.annotation]


class AnnotationLoader:
    """Parses a GTF once and reuses it for every sample that references it."""

    def __init__(self, biotype_attributes="ensembl"):
        self.biotype_attributes = biotype_attributes
        self._key = None
        self._annotation = None

    def get(self, path):
        key = str(Path(path).resolve())
        if key != self._key:
            self._annotation = load_annotation(path, biotype_attributes=self.biotype_attributes)
            self._key = key
        return self._annotation


def build_sample_graph(inputs, params=None, known_fusions=None, annotation_loader=None,
                       bam_threads=1):
    """Parse one sample's files and build its graph (no caching)."""
    params = params or GraphParams()
    loader = annotation_loader or AnnotationLoader(params.biotype_attributes)

    log.info("Building graph for %s", inputs.sample_id)
    annotation = loader.get(inputs.annotation)
    junctions = load_chimeric_junctions(
        inputs.chimeric_junctions, min_split_reads=params.min_split_reads
    )
    discordant = load_discordant_pairs(inputs.bam, threads=bam_threads)
    counts = load_expression(inputs.gene_counts, library_type=params.library_type)

    return build_graph(
        counts, junctions, discordant, annotation,
        params=params, known_fusions=known_fusions,
    )


def build_or_load_graph(inputs, params=None, known_fusions=None, cache=None,
                        annotation_loader=None, bam_threads=1, cache_extra=""):
    """Build a sample graph, reusing a cached copy when one is valid."""
    params = params or GraphParams()
    cache = cache if isinstance(cache, GraphCache) else GraphCache(cache)

    key = None
    if cache.enabled:
        key = cache_key(inputs.paths(), params, extra=cache_extra)
        cached = cache.load(key)
        if cached is not None:
            return cached

    graph = build_sample_graph(
        inputs, params=params, known_fusions=known_fusions,
        annotation_loader=annotation_loader, bam_threads=bam_threads,
    )
    if key is not None:
        cache.store(key, graph)
    return graph

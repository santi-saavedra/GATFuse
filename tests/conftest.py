"""Shared fixtures: a tiny synthetic sample that exercises the full pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_dir():
    return DATA


@pytest.fixture(scope="session")
def annotation_path():
    return DATA / "annotation.gtf"


@pytest.fixture(scope="session")
def chimeric_path():
    return DATA / "sample.Chimeric.out.junction"


@pytest.fixture(scope="session")
def counts_path():
    return DATA / "sample.ReadsPerGene.out.tab"


@pytest.fixture(scope="session")
def bam_path(tmp_path_factory):
    """Convert the checked-in SAM into a sorted, indexed BAM."""
    pysam = pytest.importorskip("pysam")
    workdir = tmp_path_factory.mktemp("bam")
    unsorted = workdir / "unsorted.bam"
    sorted_bam = workdir / "sample.bam"

    with pysam.AlignmentFile(str(DATA / "sample.sam"), "r") as source:
        with pysam.AlignmentFile(str(unsorted), "wb", template=source) as sink:
            for record in source:
                sink.write(record)
    pysam.sort("-o", str(sorted_bam), str(unsorted))
    pysam.index(str(sorted_bam))
    return sorted_bam


@pytest.fixture(scope="session")
def annotation(annotation_path):
    from gatfuse.io import load_annotation

    return load_annotation(annotation_path, biotype_attributes="auto")


@pytest.fixture(scope="session")
def sample_graph(annotation, chimeric_path, counts_path, bam_path):
    from gatfuse.config import GraphParams
    from gatfuse.graph import build_graph
    from gatfuse.io import load_chimeric_junctions, load_discordant_pairs, load_expression

    params = GraphParams(min_split_reads=2, use_known_fusions=False)
    return build_graph(
        load_expression(counts_path),
        load_chimeric_junctions(chimeric_path, min_split_reads=2),
        load_discordant_pairs(bam_path),
        annotation,
        params=params,
    )


@pytest.fixture()
def graph_copy(sample_graph):
    """A throwaway copy for tests that mutate the graph in place."""
    return sample_graph.clone()


@pytest.fixture(scope="session")
def known_fusions():
    from gatfuse.io import load_known_fusions
    from gatfuse.resources import bundled_known_fusions_path

    return load_known_fusions(bundled_known_fusions_path())

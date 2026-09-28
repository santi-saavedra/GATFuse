# GATFuse

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23016797.svg)](https://doi.org/10.5281/zenodo.23016797)

**Gene fusion detection from RNA-seq data using graph attention networks.**

GATFuse reformulates fusion detection as an *edge-classification* problem. Each
RNA-seq sample becomes a directed graph whose nodes are expressed genes and
whose edges are the chimeric junctions reported by STAR. A GATv2 network scores
every edge, separating genuine fusions from the sequencing and alignment
artefacts that dominate chimeric output.

---

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Interpreting results](#interpreting-results)
- [Command reference](#command-reference)
- [Training your own model](#training-your-own-model)
- [Performance](#performance)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Citation](#citation)

---

## How it works

1. **Parse** the STAR outputs: chimeric junctions, the alignment BAM (for
   discordant mate pairs) and per-gene read counts.
2. **Build a graph** per sample. Nodes are genes with positive TPM; each
   chimeric breakpoint becomes a directed donor → acceptor edge. Distinct
   breakpoints between the same gene pair stay distinct edges, so
   breakpoint-level resolution is preserved.
3. **Describe every edge** with 30 features (32 with the optional known-fusion
   catalogue): read support, breakpoint distance and strandedness, exon context,
   depth-normalised evidence (FFPM), anchor balance, per-gene chimeric
   background and partner promiscuity.
4. **Score** each edge with a two-layer GATv2 network whose attention is
   conditioned on the junction embedding, so neighbour importance depends on
   junction quality rather than gene identity alone. An *evidence gate* adds a
   monotone, non-negative path from raw split-read support to the score, so
   strong read support can never lower a candidate's rank.
5. **Calibrate and filter**: Platt scaling turns logits into probabilities, then
   optional rule-based filters remove read-through transcripts, intragenic
   self-joins, artefact-prone gene families and non-canonical homology
   artefacts.

See [docs/method.md](docs/method.md) for the full description.

## Requirements

| Component | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | |
| STAR | ≥ 2.7 | produces the three input files; not required by GATFuse itself |
| RAM | ≥ 8 GB | dominated by parsing the reference annotation |
| GPU | optional | CUDA or Apple MPS; inference is fast on CPU |

Python dependencies (`numpy`, `pandas`, `pysam`, `torch`, `torch-geometric`) are
installed automatically.

## Installation

Every route starts from a clone: the conda environment file and the Dockerfile
are part of the repository, not of the installed package.

```bash
git clone https://github.com/santi-saavedra/GATFuse.git
cd GATFuse
```

Then pip:

```bash
pip install .
gatfuse --version
```

Conda (also installs STAR and samtools):

```bash
conda env create -f environment.yml
conda activate gatfuse
pip install --no-deps -e .
```

Docker:

```bash
docker build -t gatfuse:1.0.0 .
docker run --rm -v "$PWD:/data" gatfuse:1.0.0 detect --help
```

The pretrained model and the known-fusion catalogue are installed with the
package, so no additional download is needed. Full details, including GPU
builds, are in [docs/installation.md](docs/installation.md).

## Quick start

Align with STAR, enabling chimeric detection (`scripts/run_star.sh` wraps this
with the parameters the released model was trained on):

```bash
scripts/run_star.sh -i /path/to/STAR_index -1 reads_1.fastq.gz -2 reads_2.fastq.gz \
                    -o alignments -p sample_ -t 8
```

Then detect fusions:

```bash
gatfuse detect \
    --bam                alignments/sample_Aligned.sortedByCoord.out.bam \
    --chimeric-junctions alignments/sample_Chimeric.out.junction \
    --gene-counts        alignments/sample_ReadsPerGene.out.tab \
    --annotation         reference/annotation.gtf \
    --output             fusions.tsv
```

That is the whole pipeline. Everything else — the model, the threshold, the
feature normalisation, the graph-construction parameters — is read from the
checkpoint, so a run is reproducible from the four input files alone.

## Inputs

| Option | File | Produced by |
|---|---|---|
| `-b, --bam` | `Aligned.sortedByCoord.out.bam` | STAR (`--outSAMtype BAM SortedByCoordinate`) |
| `-c, --chimeric-junctions` | `Chimeric.out.junction` | STAR (`--chimOutType Junctions ...`) |
| `-e, --gene-counts` | `ReadsPerGene.out.tab` | STAR (`--quantMode GeneCounts`) |
| `-g, --annotation` | reference GTF | the annotation used to build the STAR index |

The GTF **must** be the one used for the alignment: contig names and gene IDs
have to match, and GATFuse fails with an explicit message rather than silently
producing an empty result if they do not.
See [docs/input-formats.md](docs/input-formats.md).

## Outputs

A tab-separated table, one row per reported fusion, sorted by descending score:

```
donor_gene  acceptor_gene  score   split_reads  discordant_pairs  chr_donor  brkpt_donor  strand_donor  donor_region  chr_acceptor  brkpt_acceptor  strand_acceptor  acceptor_region  pct_canonical  fusion_type
BCR         ABL1           0.9412  87           31                chr22      23632600     +             exon          chr9          133729450       +                exon             1.0            inter-chromosomal
```

With `--discarded-output` a second table records the candidates the filters
removed and why. Both files always carry the header row, so an empty result is
still machine-readable. Column definitions are in
[docs/output-formats.md](docs/output-formats.md).

## Interpreting results

- **`score`** is a calibrated probability that the junction is a genuine fusion.
  The default cut-off is the threshold that maximised F1 during the model's
  cross-validation (0.154 for the distributed model), not 0.5 — with this class
  imbalance the optimal operating point is well below 0.5.
- **Partner order is not reliable.** GATFuse attempts to report partners
  5' → 3' in transcript orientation, the convention used by Arriba and
  STAR-Fusion, but in benchmarking 7 of 17 recovered fusions came out
  transposed, with correct breakpoints. Read a call as an unordered gene pair
  and resolve the 5' partner from the coordinates and gene strands.
- **Breakpoints are the last transcribed base**, again matching Arriba and
  STAR-Fusion. Junctions inside a microhomology region may still differ by a
  base or two between callers.

- **An empty output file is a valid result**, and the exit status is 0. Use
  `--top-k 10` to inspect the highest-ranked candidates regardless of score.

## Command reference

```
gatfuse detect   score one sample against a trained model
gatfuse train    fit a model on a manifest of labelled samples
```

`gatfuse detect --help` lists every option grouped by purpose. The ones most
often changed:

| Option | Default | Purpose |
|---|---|---|
| `-t, --threshold` | model's CV threshold | minimum score to report |
| `-k, --top-k` | – | report the K best candidates instead; wins over `--threshold` |
| `-d, --discarded-output` | – | write filtered-out candidates with reasons |
| `--annotate-only` | off | keep every candidate, add a `filter_reason` column |
| `--min-split-reads` | model's value | drop junctions with less read support |
| `--device` | `auto` | `cpu`, `cuda` or `mps` |
| `--bam-threads` | 4 | BAM decompression threads |
| `--cache-dir` | off | reuse built graphs across runs |

Exit codes: `0` success (including "no fusions found"), `1` internal error,
`2` command-line misuse, `3` input file problem, `4` incompatible model.
Full reference: [docs/usage.md](docs/usage.md).

## Training your own model

```bash
gatfuse train --manifest cohort.tsv --annotation reference/annotation.gtf \
              --output-model models/my-model.pt
```

The manifest lists one sample per row with its three STAR files and its
confirmed fusions (see [examples/manifest.example.tsv](examples/manifest.example.tsv)).
Training runs 5-fold cross-validation, pools the fold predictions to pick an
operating threshold, refits on the whole cohort and stores everything — weights,
architecture, graph parameters, normalisation statistics, calibration and
threshold — in one checkpoint. `gatfuse detect -m models/my-model.pt` then needs
no further configuration. See [docs/training.md](docs/training.md).

## Performance

Runtime is dominated by input parsing, not by the network. Reading the BAM for
discordant pairs is the single largest cost; raise `--bam-threads` on a
multi-core machine. A full GENCODE annotation needs roughly 4-6 GB of RAM while
parsing. Inference on a typical sample takes a few minutes on CPU, and the GPU
buys little at this graph size. See [docs/performance.md](docs/performance.md).

## Reproducibility

- Every parameter that affects graph construction is recorded in the checkpoint
  and replayed at inference, so a model cannot be applied under conditions
  different from those it was trained under without an explicit override.
- Inference is deterministic: the model runs in evaluation mode with no
  sampling. Training is seeded (`--seed`, default 42).
- Graph caches are keyed on input files, parameters and feature version, so a
  stale cache cannot silently be reused.
- See [docs/reproducibility.md](docs/reproducibility.md), which also documents
  where the current code intentionally differs from the version that produced
  the distributed model.

## Limitations

- Coupled to STAR's chimeric output format; other aligners would need a new
  parser.
- Deliberately conservative. It trades recall for precision, so a fusion with
  weak read support can be missed rather than merely ranked low. Lower
  `--threshold` if exhaustive recall matters more than a short list.
- The distributed model was trained on 101 samples, predominantly breast cancer
  cell lines and glioblastoma; generalisation to other tumour types is not
  established.
- Detects fusions between annotated genes; junctions in unannotated regions are
  not reported.

Full discussion: [docs/limitations.md](docs/limitations.md).

## Citation

Saavedra Rojas M, Serrano E, Rodríguez-González A, Tejera-Nevado P.
*Gene Fusion Detection in RNA-seq Data Using Graph Attention Networks.*

To cite the software itself, use the Zenodo archive: version 1.0.0 is
[10.5281/zenodo.23016798](https://doi.org/10.5281/zenodo.23016798), and
[10.5281/zenodo.23016797](https://doi.org/10.5281/zenodo.23016797) always
resolves to the latest version.

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

If you use the known-fusion features, also cite the Mitelman Database of
Chromosome Aberrations and Gene Fusions in Cancer
(<https://mitelmandatabase.isb-cgc.org>).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

# Installation

## Requirements

| Component | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | |
| STAR | ≥ 2.7 | produces GATFuse's input; not needed to run GATFuse itself |
| RAM | ≥ 8 GB | parsing a full GENCODE GTF is the peak |
| Disk | ~2 GB | the package plus PyTorch |
| GPU | optional | CUDA or Apple MPS |

Python dependencies, installed automatically:

| Package | Minimum | Used for |
|---|---|---|
| `numpy` | 1.24 | feature arrays, interval lookups |
| `pandas` | 2.0 | parsing tabular inputs |
| `pysam` | 0.21 | reading BAM files |
| `torch` | 2.0 | the network |
| `torch-geometric` | 2.4 | GATv2 layers and graph batching |

Only lower bounds are declared, so GATFuse composes with an existing analysis
environment instead of pinning it.

## From source (recommended)

```bash
git clone https://medal.ctb.upm.es/internal/gitlab/Saavedra/gatfuse.git
cd gatfuse
pip install .
gatfuse --version
```

For development, install in editable mode with the test extras:

```bash
pip install -e ".[test]"
pytest
```

The suite runs in a few seconds against a miniature synthetic sample in
`tests/data/` and needs no reference genome.

## Conda

`environment.yml` also brings in STAR and samtools, giving a complete pipeline
environment:

```bash
conda env create -f environment.yml
conda activate gatfuse
pip install --no-deps -e .
```

## Docker

```bash
docker build -t gatfuse:1.0.0 .
docker run --rm -v "$PWD:/data" gatfuse:1.0.0 \
    detect -b /data/sample.bam -c /data/sample.junction \
           -e /data/sample.tab -g /data/annotation.gtf -o /data/fusions.tsv
```

The image ships CPU-only PyTorch. For GPU inference, start from an
`nvidia/cuda` base image and install the matching PyTorch build.

## GPU support

The CPU build is installed by default. To use a GPU, install the matching
PyTorch wheel *before* GATFuse:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install .
```

Then pass `--device cuda`. Apple Silicon works with `--device mps` and the
standard wheel. `--device auto` (the default) prefers CUDA, then MPS, then CPU.
An explicit `--device cuda` fails loudly when no GPU is visible, rather than
quietly falling back — a job submitted to a GPU queue should not run for hours
on CPU without saying so.

## Bundled data

Installing the package also installs:

- `gatfuse/resources/models/gatfuse-v1.pt` — the pretrained model (1.1 MB)
- `gatfuse/resources/known_fusions/mitelman_fusions.tsv` — the known-fusion
  catalogue (0.5 MB)

Both are used by default, so `gatfuse detect` works immediately after
installation. Override them with `--model` and `--known-fusions`.

## Verifying the installation

```bash
gatfuse --version
gatfuse detect --help
pytest            # from a source checkout
```

## Troubleshooting

**`pysam` fails to build.** Install the compression headers first:
`apt-get install zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev`, or
use the conda package (`conda install -c bioconda pysam`).

**`torch-geometric` import errors.** Install `torch` first, then
`torch-geometric`; it compiles against the installed PyTorch version.

**"The bundled model is not available in this installation."** The package data
did not install, which usually means the source tree was added to `PYTHONPATH`
rather than installed. Either `pip install .` or pass `--model` explicitly.

# Changelog

All notable changes to GATFuse are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
semantic versioning.

## [1.0.0] - 2026-09-08

First public release, accompanying the article *Gene Fusion Detection in RNA-seq
Data Using Graph Attention Networks*.

### Added

- `gatfuse` command with `detect` and `train` subcommands, installable via
  `pip install .`.
- Pretrained model and known-fusion catalogue bundled with the package, so
  `gatfuse detect` runs with no additional downloads.
- Edge-level fusion scoring with a GATv2 network over per-sample gene graphs,
  Platt-scaled probabilities and rule-based filtering of read-through
  transcripts, intragenic self-joins and artefact-prone gene families.
- Graph-construction parameters recorded in every checkpoint and replayed at
  inference, so a model cannot be applied under conditions it was not trained
  under without an explicit, logged override.
- `--discarded-output`, writing filtered candidates with the reason for each.
- `discordant_pairs` column in the output, as described in the article.
- Documented exit codes: 0 success, 1 internal error, 2 usage, 3 input, 4 model.
  A sample with no reportable fusion writes an empty table and exits 0; an empty
  result is a biological outcome, not a pipeline failure.

- `--device {auto,cpu,cuda,mps}`, `--threads`, `--seed`, and `--log-level`.
- `--blacklist-pattern` for additional artefact families.
- Named feature schema (`gatfuse/features.py`) from which normalisation and
  documentation derive.
- Gene biotypes read from both Ensembl (`gene_biotype`) and GENCODE
  (`gene_type`) annotations.
- Batch training from a manifest, whose paths resolve relative to the manifest
  file so a manifest travels with its data.
- Structured progress logging on stderr, leaving stdout free.
- Test suite (88 tests) over a synthetic sample, including a regression check
  pinning the distributed model's scores.
- `scripts/run_star.sh`, running STAR with the parameters the released model was
  trained on.
- Documentation set under `docs/`, plus a Dockerfile and conda environment.
- Five declared dependencies with lower bounds only (`numpy`, `pandas`,
  `pysam`, `torch`, `torch-geometric`).

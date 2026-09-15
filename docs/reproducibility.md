# Reproducibility

## What is fixed and what varies

**Inference is deterministic.** The model runs in evaluation mode with dropout
disabled and no sampling anywhere in the path. The same inputs, model and
parameters give the same scores on every run. Results may differ in the last
decimal between CPU and GPU, as with any floating-point pipeline.

**Training is seeded.** `--seed` (default 42) fixes weight initialisation, the
train/validation split, fold assignment, negative subsampling and the Gaussian
feature-noise augmentation. Two runs with the same seed, data and parameters
produce the same model on the same hardware. GPU kernels are not
bit-reproducible across hardware or CUDA versions, so treat cross-machine
equality as approximate.

## Parameters travel with the model

Anything that changes how a graph is built is recorded in the checkpoint and
replayed by `gatfuse detect`:

| Recorded | Why it matters |
|---|---|
| `min_split_reads` | changes which junctions exist at all |
| `library_type` | changes which genes become nodes |
| `biotype_attributes` | changes the protein-coding feature and gene assignment |
| `use_known_fusions` | changes the feature width |
| `known_fusion_recurrence` | changes the recurrence feature's scale |
| `edge_norm_stats` | the exact mean and standard deviation per column |
| `platt_params` | calibration coefficients |
| `best_val_thr` | the operating threshold |

A model therefore cannot be applied under conditions different from those it was
trained under unless you override a setting explicitly, and any override is
logged. This is the main reason `gatfuse detect` needs nothing beyond the four
input files.

## Reproducing a run

The progress log records the model's provenance, the effective parameters and
the candidate counts at each stage, so a pipeline log is sufficient to
reconstruct what happened. To reproduce a result you need:

1. the same input files (BAM, chimeric junctions, gene counts),
2. the same annotation GTF — the *same file*, not merely the same release name,
3. the same checkpoint,
4. the GATFuse version from the log.

Record the annotation's checksum alongside your results. Two GTFs labelled with
the same GENCODE release can differ if one has been backmapped or filtered, and
that changes gene assignment.

## Caching

`--cache-dir` stores built graphs keyed on the input files (resolved path and
size), all graph-construction parameters and the feature-layout version. A stale
cache cannot silently be reused: any change to those inputs produces a different
key. Caching is off by default so no run depends on hidden state.

The key uses file size rather than a content hash, which is fast but would miss
an edit that preserves size exactly. If you rewrite an input in place, clear the
cache.

## The distributed model

`gatfuse-v1.pt` was trained on 101 samples
(predominantly breast cancer cell lines and glioblastoma, plus one synthetic
sample).

| Setting | Value |
|---|---|
| Node / edge features | 5 / 32 (known-fusion features enabled) |
| Hidden dimension | 64 |
| Edge embedding dimension | 32 |
| Attention heads | 2 |
| GATv2 layers | 2 |
| Dropout | 0.6 |
| Learning rate | 0.005 |
| Batch size | 8 |
| Early-stopping patience | 80 |
| Loss | focal, γ = 2.0, α = 0.75 |
| Minimum split reads | 2 |
| Cross-validation | 5-fold |
| Epochs run | 216 |
| Pooled validation AUPRC | 0.604 |
| Operating threshold | 0.154 |
| Calibration | A = 2.209, B = -0.996 |

The test suite pins these scores with a regression check
(`tests/test_detect.py::test_released_model_scores_are_stable`), so no change to
graph construction can silently alter what the distributed model reports for a
given sample.

## Where the distributed model differs from the training defaults

The distributed checkpoint carries the settings it was trained under and
`gatfuse detect` replays them, so none of what follows affects running
`gatfuse-v1.pt`. It matters when you **train a new model**: three settings
default differently today, so retraining on the same cohort does not reproduce
`gatfuse-v1.pt` bit for bit.

### 1. Which features are standardised

GATFuse declares the edge-feature layout by name in `gatfuse/features.py` and
standardises exactly the features marked continuous. The distributed model
carries its own column list in the checkpoint, applied verbatim: it z-scores the
binary `known_pair` indicator, and leaves `log_donor_chimeric_background` and
`log_acceptor_chimeric_background` raw.

### 2. Known-pair recurrence

`log_known_pair_recurrence` is built from the case counts in the known-fusion
catalogue. The distributed model was trained without them, so for it the feature
is `log1p(1)` on every catalogued pair — a duplicate of the binary flag rather
than the recurrence the method describes. The behaviour is recorded per model as
`known_fusion_recurrence`: `false` for the distributed model, `true` for models
trained with this version.

### 3. Gene biotype attribute

`--biotype-attributes auto` reads either the Ensembl attribute `gene_biotype` or
the GENCODE `gene_type`, and is the default for training. `gatfuse detect` uses
the mode recorded in the checkpoint, which is `ensembl` for the distributed
model. Its training cohort was annotated with GENCODE, where that attribute is
absent, so the protein-coding node feature was constant zero and no coding-gene
preference was applied when a breakpoint overlapped several genes. Setting
`--biotype-attributes auto` with the distributed model changes both the node
feature and the breakpoint-to-gene assignment, and is not recommended.

## Retraining on your own cohort

Report, alongside results: the GATFuse version, the seed, the annotation
checksum, the manifest, and the checkpoint's `graph_params` and `train_params`
(both are stored in the file). Those are sufficient for a third party to
reproduce the model, given the same alignments.

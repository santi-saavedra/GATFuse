# Performance

## Where the time goes

Runtime is dominated by parsing, not by the network. For a typical human RNA-seq
sample:

| Stage | Share | Scales with | Lever |
|---|---|---|---|
| BAM scan for discordant pairs | largest | total alignments | `--bam-threads` |
| Annotation parsing | large | GTF size | `--cache-dir`, reuse across samples |
| Graph construction | moderate | chimeric junctions × overlapping genes | `--min-split-reads` |
| Scoring | small | edges | `--device`, `--threads` |

The network itself is ~90 k parameters over a graph with thousands of edges, so
inference is a fraction of a second. Optimising it would not shorten a run
noticeably.

## Tuning

**`--bam-threads` (default 4).** BGZF decompression is the single largest cost.
Raising it to 4-8 gives close to linear speed-up on a multi-core machine; beyond
that the Python loop over reads becomes the limit.

**`--cache-dir`.** Caches built graphs keyed on the input files (path and size),
the construction parameters and the feature version. The second run on the same
sample skips parsing entirely. Use it when scoring one sample with several
thresholds, during a hyperparameter search, or when re-running a cohort after a
software update that does not change the features. Caching is off by default so
that a run never depends on hidden state.

**`--min-split-reads`.** Raising it from 2 to, say, 5 shrinks the candidate set
substantially on deep libraries and speeds up graph construction. It also
removes low-coverage fusions from consideration entirely, so raise it only when
you are willing to trade that recall — and note that it moves the sample away
from the training distribution.

**`--device`.** `auto` prefers CUDA, then Apple MPS, then CPU. At this graph
size the GPU saves little; the main reason to use one is a shared node where CPU
threads are constrained.

**`--threads`.** Caps PyTorch intra-op parallelism. Set it on shared or
job-scheduled machines, where PyTorch's default of "all visible cores" oversubscribes
the allocation.

## Memory

Peak memory is dominated by the annotation. A full GENCODE human GTF needs
roughly 4-6 GB while parsing; only the gene and exon columns are retained
afterwards. The discordant-pair table adds a few hundred MB on samples with many
inter-chromosomal pairs.

Ways to reduce it:

- Use a gene-level GTF if you do not need exon context (the exon features then
  read as zero, which changes results — retrain if you do this).
- Avoid `--workers > 1` during training on large annotations: each worker parses
  its own copy.

Within one process the annotation is parsed once and reused across samples, so a
sequential cohort run pays the cost a single time.

## Scaling to a cohort

Samples are independent, so the natural parallelism is one job per sample:

```bash
# Slurm array
gatfuse detect -b "$SAMPLE/Aligned.sortedByCoord.out.bam" \
               -c "$SAMPLE/Chimeric.out.junction" \
               -e "$SAMPLE/ReadsPerGene.out.tab" \
               -g reference/annotation.gtf \
               -o "results/$NAME.tsv" \
               --bam-threads 8 --threads 4 --device cpu
```

Request memory for the annotation (8 GB is comfortable) and give each job
several cores for the BAM scan. Non-zero exit codes are real failures — a sample
with no fusion exits 0 — so a workflow manager can branch on status directly.

## Training cost

Training is dominated by building the cohort's graphs; the optimisation loop
itself is minutes. Build once with `--cache-dir`, then iterate on
hyperparameters against the cache. Cross-validation multiplies the *training*
cost by the number of folds plus one final refit, but the cached graphs are
reused throughout.

# Usage

```
gatfuse detect   score one sample against a trained model
gatfuse train    fit a model on a manifest of labelled samples
```

Every option is listed by `gatfuse <command> --help`, grouped by purpose.

## `gatfuse detect`

```bash
gatfuse detect -b sample.bam -c sample.Chimeric.out.junction \
               -e sample.ReadsPerGene.out.tab -g annotation.gtf \
               -o fusions.tsv
```

### Required input

| Option | Description |
|---|---|
| `-b, --bam` | coordinate-sorted BAM from STAR |
| `-c, --chimeric-junctions` | `Chimeric.out.junction` |
| `-e, --gene-counts` | `ReadsPerGene.out.tab` |
| `-g, --annotation` | the GTF used to build the STAR index |

### Model

| Option | Default | Description |
|---|---|---|
| `-m, --model` | bundled model | trained checkpoint |
| `--known-fusions` | bundled catalogue | catalogue for the prior-knowledge features |
| `--no-known-fusions` | off | build the graph without those features |

Whether the catalogue is needed follows from the model. Supplying
`--no-known-fusions` to a model trained with them produces a clear error, not a
shape crash.

### Output

| Option | Default | Description |
|---|---|---|
| `-o, --output` | `gatfuse_fusions.tsv` | reported fusions |
| `-d, --discarded-output` | – | filtered candidates with the reason for each |

### Candidate selection

| Option | Default | Description |
|---|---|---|
| `-t, --threshold` | model's CV threshold | minimum score to report |
| `-k, --top-k` | – | report the K best candidates regardless of score |

`--top-k` takes precedence when both are given, and a warning says so. Use
`--top-k` for exploratory review or when comparing samples by rank; use
`--threshold` for production runs where the score has a fixed meaning.

### Graph construction

| Option | Default | Description |
|---|---|---|
| `--min-split-reads` | model's value | minimum split reads per junction |
| `--library-type` | model's value | `unstranded`, `stranded_forward`, `stranded_reverse` |
| `--biotype-attributes` | model's value | `ensembl` or `auto` |

These default to the values recorded in the checkpoint, so inference reproduces
the conditions the model was trained under. Overriding one is legitimate — for
example raising `--min-split-reads` on very deep data — but it moves the sample
away from the training distribution, and the log records the change.

### Post-processing

| Option | Default | Description |
|---|---|---|
| `--no-postprocess` | off | report raw model output, no filters |
| `--annotate-only` | off | keep every candidate, add `filter_reason` |
| `--keep-intragenic` | off | keep same-gene junctions |
| `--keep-readthrough` | off | keep adjacent co-oriented gene pairs |
| `--keep-blacklisted` | off | keep artefact-prone gene families |
| `--keep-noncanonical` | off | keep non-canonical inter-chromosomal exon-exon junctions |
| `--readthrough-max-distance` | 100000 | bp distance treated as read-through |
| `--noncanonical-min-canonical` | 0.1 | minimum canonical-motif fraction to keep |
| `--blacklist-pattern REGEX` | – | extra artefact family; repeatable |

### Performance

| Option | Default | Description |
|---|---|---|
| `--device` | `auto` | `cpu`, `cuda`, `mps` |
| `--threads` | PyTorch default | intra-op threads |
| `--bam-threads` | 4 | BAM decompression threads |
| `--cache-dir` | off | reuse built graphs across runs |

## Examples

**Standard run.**

```bash
gatfuse detect -b sample.bam -c sample.junction -e sample.tab -g ref.gtf -o fusions.tsv
```

**Exploratory review of a negative result.** Nothing passed the threshold; look
at the ten best candidates and why others were filtered.

```bash
gatfuse detect -b sample.bam -c sample.junction -e sample.tab -g ref.gtf \
               -o top10.tsv -d discarded.tsv --top-k 10
```

**Maximum recall.** No filters, no threshold — every scored candidate.

```bash
gatfuse detect -b sample.bam -c sample.junction -e sample.tab -g ref.gtf \
               -o all_candidates.tsv --threshold 0 --no-postprocess
```

**One table with everything annotated.**

```bash
gatfuse detect -b sample.bam -c sample.junction -e sample.tab -g ref.gtf \
               -o annotated.tsv --annotate-only --threshold 0
```

**Keep read-throughs.** Relevant when studying conjoined transcripts, which the
default filter removes.

```bash
gatfuse detect -b sample.bam -c sample.junction -e sample.tab -g ref.gtf \
               -o fusions.tsv --keep-readthrough
```

**A cohort, reusing the annotation parse via the graph cache.**

```bash
for sample in alignments/*/; do
    name=$(basename "$sample")
    gatfuse detect \
        -b "$sample/Aligned.sortedByCoord.out.bam" \
        -c "$sample/Chimeric.out.junction" \
        -e "$sample/ReadsPerGene.out.tab" \
        -g reference/annotation.gtf \
        -o "results/${name}.fusions.tsv" \
        --cache-dir .gatfuse-cache --bam-threads 8
done
```

**On a GPU node.**

```bash
gatfuse detect -b sample.bam -c sample.junction -e sample.tab -g ref.gtf \
               -o fusions.tsv --device cuda --bam-threads 8
```

## Using GATFuse from Python

The CLI is a thin layer over a stable API:

```python
from gatfuse import checkpoint, detect
from gatfuse.config import PostprocessParams
from gatfuse.io import load_annotation, load_known_fusions
from gatfuse.pipeline import SampleInputs, build_or_load_graph
from gatfuse.resources import bundled_known_fusions_path, bundled_model_path

model = checkpoint.load(bundled_model_path())
params = model.graph_params                     # replay the training settings
catalogue = load_known_fusions(
    bundled_known_fusions_path(), use_recurrence=params.known_fusion_recurrence
)
annotation = load_annotation("reference/annotation.gtf", params.biotype_attributes)

graph = build_or_load_graph(
    SampleInputs(
        bam="sample.bam",
        chimeric_junctions="sample.Chimeric.out.junction",
        gene_counts="sample.ReadsPerGene.out.tab",
        annotation="reference/annotation.gtf",
        sample_id="sample",
    ),
    params=params,
    known_fusions=catalogue,
)

result = detect.detect(
    graph, model, annotation,
    threshold=model.threshold,
    postprocess=PostprocessParams(),
)
print(result.reported.head())
```

`result.reported` and `result.discarded` are pandas DataFrames.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success, including "no fusion passed the threshold" |
| 1 | unexpected internal error (re-run with `--log-level debug` for a traceback) |
| 2 | command-line misuse |
| 3 | missing or malformed input |
| 4 | checkpoint missing, unreadable or incompatible |

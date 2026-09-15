# Training a model

GATFuse ships a model trained on 101 samples. Train your own when your data
differ materially from that cohort — another tumour type, another sequencing
protocol, another reference build — or when you have confirmed fusions that the
distributed model was never shown.

```bash
gatfuse train --manifest cohort.tsv --annotation reference/annotation.gtf \
              --output-model models/my-model.pt
```

## Preparing a cohort

Align every sample with the same STAR parameters and the same reference
(`scripts/run_star.sh`), then write a manifest listing each sample's three STAR
files and its confirmed fusions:

```
sample_id	bam	chimeric_junctions	gene_counts	fusions
K562	aln/K562/Aligned.sortedByCoord.out.bam	aln/K562/Chimeric.out.junction	aln/K562/ReadsPerGene.out.tab	BCR:ABL1
MCF-7	aln/MCF-7/Aligned.sortedByCoord.out.bam	aln/MCF-7/Chimeric.out.junction	aln/MCF-7/ReadsPerGene.out.tab	BCAS4:BCAS3;ARFGEF2:SULF2
```

Paths may be relative to the manifest. Full specification:
[input-formats.md](input-formats.md).

### Label quality dominates everything else

The positive class is tiny — often two or three edges among thousands — so each
label carries a great deal of weight.

- **Prefer breakpoint-level labels** (`GENE1:GENE2@chrA:posA-chrB:posB`) when
  coordinates are known. A gene-level label marks *every* junction between the
  pair positive, including artefactual ones, which teaches the model that those
  artefacts are real.
- **Missing labels become false negatives.** A genuine fusion that is present in
  a sample but absent from the manifest trains the model against itself. This is
  the main reason a curated, literature-confirmed label set matters more than
  cohort size.
- **Watch the "not found in the graph" warnings.** They mean a listed fusion had
  no corresponding edge, usually because the gene is not expressed, the symbol
  does not match the annotation, or the junction fell below
  `--min-split-reads`.

### How many samples?

The distributed model used 101 samples yielding a few hundred positive edges.
Below roughly 20 samples, cross-validation folds become too small to give a
stable threshold; `--cv-folds 0 --val-split 0.2` is more honest there. GATFuse
falls back to a single split automatically when there are fewer samples than
folds.

## What training does

1. Builds a graph per sample and labels its edges.
2. Runs stratified K-fold cross-validation (folds balanced by whether a sample
   has any confirmed fusion). Normalisation statistics are computed from each
   fold's training samples only, so validation samples never influence the
   scaling applied to them.
3. Pools validation predictions across folds and picks the threshold maximising
   F1 over the pooled set — an operating point measured on every sample rather
   than on one arbitrary split.
4. Refits on the whole cohort with relaxed patience (no validation signal
   remains).
5. Fits Platt calibration and writes one checkpoint containing weights,
   architecture, graph parameters, normalisation statistics, calibration and
   threshold.

## Options that matter most

| Option | Default | When to change |
|---|---|---|
| `--cv-folds` | 5 | 0 for a single hold-out split on small cohorts |
| `--val-split` | 0.2 | validation fraction when cross-validation is off |
| `--epochs` | 1000 | early stopping usually fires well before this |
| `--patience` | 80 | lower to shorten runs, raise for noisy validation curves |
| `--learning-rate` | 0.005 | lower if the loss oscillates |
| `--batch-size` | 8 | lower if memory is tight |
| `--dropout` | 0.6 | lower with more data, raise if train/val F1 diverge |
| `--select-metric` | `auprc` | `f1` only if you have many positives |
| `--seed` | 42 | change to check that results are not seed artefacts |
| `--keep-empty-graphs` | off | keep samples with no confirmed fusion |

Architecture options (`--hidden-dim`, `--edge-embed-dim`, `--heads`,
`--gnn-layers`, `--no-node-skip`) default to the published configuration. Deeper
stacks oversmooth on cohorts of this size; two layers is a deliberate choice,
not a placeholder.

### `--keep-empty-graphs`

By default, samples with no confirmed fusion are dropped from the *training*
set. Their gradient carries only negative-class signal, which focal loss already
down-weights, so they mainly dilute the few positives. Keep them if your
negative controls are informative and you would rather train the model to be
explicitly quiet on them. If every sample lacks labels, training stops with an
explanatory error rather than producing a meaningless model.

## Monitoring a run

```
INFO  Training on 92 graphs | 312 positive / 486204 negative edges | focal loss (gamma=2.0, alpha=0.75)
INFO  Epoch   10/1000 | loss 0.0042 | train F1 0.412@0.318 | AUPRC 0.387 | lr 0.004988
INFO               val F1 0.5301@0.194 (P=0.48 R=0.59) | val AUPRC 0.4412 | 71/104238 positive
INFO  New best model at epoch 24 | val AUPRC 0.5108 | val F1 0.5612@0.163 | train F1 0.6203
```

Read validation AUPRC, not F1 at 0.5. With this imbalance the optimal threshold
is far below 0.5, and F1@0.5 will look poor even for a good model.

Signs of trouble:

- **Validation AUPRC stays near the positive rate** — the model is not learning.
  Check that labels resolve (look for "not found in the graph" warnings).
- **Train F1 approaches 1.0 while validation stalls** — overfitting. Raise
  `--dropout`, raise `--feature-noise-std`, or add samples.
- **Very few positive edges** — likely gene symbols that do not match the
  annotation.

## After training

The checkpoint is self-contained:

```bash
gatfuse detect -m models/my-model.pt -b sample.bam -c sample.junction \
               -e sample.tab -g reference/annotation.gtf -o fusions.tsv
```

The threshold, normalisation, calibration and graph parameters all come from the
checkpoint. Inspect them with:

```python
from gatfuse import checkpoint
ck = checkpoint.load("models/my-model.pt")
print(ck.describe())
print(ck.graph_params)
print(ck.metadata["train_params"])
```

## Performance notes

Graph construction dominates. Use `--cache-dir` so a hyperparameter search
builds each graph once, and `--bam-threads 8` to speed the BAM scan.
`--workers N` builds N samples in parallel, but each worker parses the
annotation independently, so peak memory scales with N — with a full GENCODE
GTF, 4 workers is already ~20 GB. Sequential building with a warm cache is
usually the better trade.

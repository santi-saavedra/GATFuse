# Output formats

## Reported fusions (`-o, --output`)

A tab-separated table, one row per reported fusion, sorted by descending score.
The header is always written, so an empty result is still machine-readable.

| Column | Type | Description |
|---|---|---|
| `donor_gene` | string | 5' partner (gene symbol, or Ensembl ID when the GTF has no symbol) |
| `acceptor_gene` | string | 3' partner |
| `score` | float 0-1 | calibrated probability that the junction is a genuine fusion |
| `split_reads` | int | reads spanning the junction; summed over merged duplicates |
| `discordant_pairs` | int | inter-chromosomal mate pairs supporting the breakpoints |
| `chr_donor` | string | contig of the 5' breakpoint |
| `brkpt_donor` | int | 5' breakpoint, last transcribed base (1-based) |
| `strand_donor` | `+`/`-` | strand of the 5' partner |
| `donor_region` | enum | `exon`, `intron` or `intergenic` |
| `chr_acceptor` | string | contig of the 3' breakpoint |
| `brkpt_acceptor` | int | 3' breakpoint, last transcribed base (1-based) |
| `strand_acceptor` | `+`/`-` | strand of the 3' partner |
| `acceptor_region` | enum | `exon`, `intron` or `intergenic` |
| `pct_canonical` | float 0-1 | fraction of this gene pair's junctions with a canonical splice motif |
| `fusion_type` | enum | `inter-chromosomal`, `intra-chromosomal` or `intra-genic` |

Example:

```
donor_gene	acceptor_gene	score	split_reads	discordant_pairs	chr_donor	brkpt_donor	strand_donor	donor_region	chr_acceptor	brkpt_acceptor	strand_acceptor	acceptor_region	pct_canonical	fusion_type
BCR	ABL1	0.9412	87	31	chr22	23632600	+	exon	chr9	133729450	+	exon	1.0	inter-chromosomal
EWSR1	FLI1	0.7735	24	9	chr22	29684775	+	exon	chr11	128675261	+	exon	1.0	inter-chromosomal
```

### Conventions

**Partner order is not reliable.** GATFuse attempts to report partners
5' → 3' in transcript orientation, as Arriba and STAR-Fusion do: STAR labels the
side carrying the GT donor motif from the genomic sequence, and for a
minus-strand junction the columns are swapped to recover transcript order. That
recovery often fails. In benchmarking, 7 of the 17 recovered fusions were
reported with the partners transposed, although the breakpoint coordinates were
correct in every case. Read a call as an unordered gene pair with resolved
breakpoints, and determine the 5' partner from those coordinates and the
annotated strand of each gene. See [limitations.md](limitations.md).

**Breakpoint coordinates.** STAR reports the first base of the intron flanking
the junction. GATFuse reports the last transcribed base, one position closer to
the junction, again matching Arriba and STAR-Fusion. Junctions inside a
microhomology or repeat region can still differ by a base or two between
callers, because each anchors differently within the ambiguous stretch.

**Deduplication.** STAR reports the same physical junction twice, once from each
strand's point of view. Those rows are merged after the orientation fix: the
higher-scoring row is kept and read counts are summed, so `split_reads` is the
total evidence for the junction.

## Discarded candidates (`-d, --discarded-output`)

Same columns plus `filter_reason`, listing every rule that matched, separated by
`;`. Written only when the option is given.

| `filter_reason` | Meaning |
|---|---|
| `intragenic` | donor and acceptor are the same gene |
| `readthrough` | adjacent, co-oriented genes on one chromosome; continuous transcription rather than fusion |
| `blacklist` | a partner belongs to an artefact-prone family (ribosomal, mitochondrial, histone, snRNA/snoRNA, tRNA) |
| `noncanonical_artifact` | inter-chromosomal exon-exon junction with no canonical splice motif, typical of paralogous cross-mapping |

Reviewing this file is the fastest way to check that a filter is not discarding
a fusion you expect. To see everything in one table instead, use
`--annotate-only`, which keeps all candidates and adds `filter_reason` to the
main output.

## Progress log

Progress, warnings and errors go to **stderr**, leaving stdout free. Control the
verbosity with `--log-level {debug,info,warning,error}`.

The log records the model's provenance, the parameters actually used and the
candidate counts at each stage, so a pipeline log is enough to reconstruct a
run:

```
[2026-09-08 18:30:35] INFO    Model: gatfuse-v1.pt (format=legacy, train_graphs=92, cv_folds=5, val_auprc=0.6043, threshold=0.1542)
[2026-09-08 18:30:35] INFO    Chimeric junctions: 4 with >= 2 split reads (from 22 chimeric alignments)
[2026-09-08 18:30:36] INFO    Graph edges: 4 chimeric junctions (32 features)
[2026-09-08 18:30:36] INFO    Applied training-time feature normalisation (18 columns)
[2026-09-08 18:30:38] INFO    Applied probability calibration (A=2.2085, B=-0.9958)
[2026-09-08 18:30:38] INFO    Post-processing: removed 2 candidate(s) (readthrough=1, intragenic=1)
```

## Model checkpoints (`gatfuse train -o`)

A PyTorch `.pt` file holding everything needed to reproduce inference:

| Key | Contents |
|---|---|
| `state_dict` | network weights |
| `config` | architecture (feature widths, hidden sizes, heads, layers, dropout) |
| `metadata.graph_params` | graph-construction parameters used for training |
| `metadata.train_params` | optimisation settings |
| `metadata.edge_norm_stats` | per-column normalisation mean and standard deviation |
| `metadata.platt_params` | calibration coefficients `A` and `B` |
| `metadata.best_val_thr` | the operating threshold, from cross-validation |
| `metadata.best_val_auprc` | validation AUPRC |
| `metadata.feature_version` | feature layout identifier |
| `metadata.gatfuse_version` | version that wrote the checkpoint |

Inspect one without running a detection:

```python
from gatfuse import checkpoint
ck = checkpoint.load("models/my-model.pt")
print(ck.describe())
print(ck.graph_params)
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success, including "no fusion passed the threshold" |
| 1 | unexpected internal error |
| 2 | command-line misuse |
| 3 | missing or malformed input |
| 4 | checkpoint missing, unreadable or incompatible with the data |

A sample with no reported fusion is a biological result, not a failure, so it
exits 0 with an empty (header-only) table. Workflow managers can therefore treat
any non-zero status as a genuine problem.

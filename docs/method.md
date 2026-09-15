# Method

GATFuse treats fusion detection as edge classification on a per-sample graph.
This page describes what the tool computes; the article gives the motivation and
the benchmarking results.

## Graph definition

A sample is a directed graph *G = (N, E)*.

**Nodes** are genes with positive TPM in that sample. Restricting to
transcriptionally active genes keeps the graph to loci that could plausibly
produce a fusion transcript.

**Edges** are chimeric junctions. Each breakpoint reported by STAR becomes one
directed edge from the donor gene to the acceptor gene, obtained by mapping the
breakpoint coordinates onto the annotation. Two junctions between the same gene
pair remain two parallel edges, so breakpoint-level resolution survives into the
classifier — a real fusion breakpoint and an artefact between the same two genes
stay distinguishable.

When a breakpoint overlaps several annotated genes, protein-coding genes take
precedence, which prevents a fusion partner from being reported as an
overlapping lncRNA. When the annotation carries no biotype attribute, no
preference is applied.

## Node features

| # | Feature | Description |
|---|---|---|
| 0 | `log_tpm` | log1p(TPM), z-scored within the sample |
| 1 | `log_gene_length` | log1p(gene length in bp), z-scored within the sample |
| 2 | `is_protein_coding` | 1 if the gene biotype is protein_coding |
| 3 | `is_known_fusion_gene` | 1 if the gene appears in the known-fusion catalogue |
| 4 | `log_partner_count` | log1p(number of distinct catalogued fusion partners) |

The last two are present only when a known-fusion catalogue is supplied. TPM
captures transcriptional activity, gene length is a noise proxy (longer genes
accumulate more spurious junctions), and the protein-coding flag encodes a prior
towards biologically plausible partners.

## Edge features

Thirty features per junction, thirty-two with the catalogue. They were developed
in three tiers: direct evidence and genomic architecture; breakpoint context
from the annotation; and background-noise and quality-normalised signals
inspired by heuristics in STAR-Fusion and Arriba.

| # | Feature | Description | z-scored |
|---|---|---|---|
| 0 | `log_split_reads` | log1p(split reads supporting the junction) | yes |
| 1 | `log_discordant_pairs` | log1p(discordant pairs near both breakpoints) | yes |
| 2 | `split_to_discordant_ratio` | split_reads / (discordant_pairs + 1) | yes |
| 3 | `log_breakpoint_distance` | log1p(distance between breakpoints; 1e7 if interchromosomal) | yes |
| 4 | `log_expression_ratio` | log1p(TPM_donor / TPM_acceptor) | yes |
| 5 | `is_interchromosomal` | 1 if donor and acceptor are on different chromosomes | no |
| 6 | `junction_type_0` | STAR junction type 0 (non-canonical motif) | no |
| 7 | `junction_type_1` | STAR junction type 1 (GT/AG) | no |
| 8 | `junction_type_2` | STAR junction type 2 (CT/AC) | no |
| 9 | `junction_type_3` | STAR junction type 3 (reserved; always 0 for current STAR) | no |
| 10 | `strand_plus_minus` | donor + / acceptor - orientation | no |
| 11 | `strand_plus_plus` | donor + / acceptor + orientation | no |
| 12 | `strand_minus_minus` | donor - / acceptor - orientation | no |
| 13 | `log_repeat_left` | log1p(repeat length left of the breakpoint) | yes |
| 14 | `log_repeat_right` | log1p(repeat length right of the breakpoint) | yes |
| 15 | `donor_in_exon` | 1 if the donor breakpoint falls inside an annotated exon | no |
| 16 | `acceptor_in_exon` | 1 if the acceptor breakpoint falls inside an annotated exon | no |
| 17 | `donor_relative_position` | donor breakpoint position within the gene body (0-1) | yes |
| 18 | `acceptor_relative_position` | acceptor breakpoint position within the gene body (0-1) | yes |
| 19 | `log_ffpm` | log1p(1000 x fusion fragments per million mapped reads) | yes |
| 20 | `log_max_balanced_anchor` | log1p(longest balanced anchor over supporting reads) | yes |
| 21 | `log_donor_promiscuity` | log1p(distinct chimeric partners of the donor gene) | yes |
| 22 | `log_acceptor_promiscuity` | log1p(distinct chimeric partners of the acceptor gene) | yes |
| 23 | `is_read_through` | 1 if the genes are adjacent, same-strand and within the read-through distance | no |
| 24 | `known_pair` | 1 if the gene pair appears in the known-fusion catalogue | no |
| 25 | `log_known_pair_recurrence` | log1p(catalogued case count for the pair) | yes |
| 26 | `log_pair_junction_count` | log1p(distinct junctions between this gene pair) | yes |
| 27 | `log_pair_max_split_reads` | log1p(split reads of the strongest junction of the pair) | yes |
| 28 | `pair_canonical_fraction` | fraction of the pair's junctions with a canonical motif | yes |
| 29 | `log_donor_chimeric_background` | log1p(total chimeric reads involving the donor gene) | yes |
| 30 | `log_acceptor_chimeric_background` | log1p(total chimeric reads involving the acceptor gene) | yes |
| 31 | `canonical_fraction_x_both_exonic` | pair_canonical_fraction x donor_in_exon x acceptor_in_exon; the canonical motif is only informative at exon-exon junctions | no |

The layout is declared once, in `gatfuse/features.py`, and every consumer —
graph construction, normalisation, documentation — derives from that
declaration. Because the optional catalogue block sits in the middle, enabling it
shifts the aggregate block by two columns, which is why the layout is addressed
by name rather than by index.

### Why these features

- **Direct evidence** (split reads, discordant pairs) is the primary signal, but
  raw counts are not comparable across samples of different depth, hence FFPM.
- **Anchor balance** distinguishes a read anchored well on both sides of the
  junction from one with a long anchor and a short soft clip; the latter is far
  weaker evidence.
- **Chimeric background and promiscuity** contextualise the raw counts: ten
  split reads mean something different for a gene that produces ten chimeric
  reads in total than for one that produces a thousand.
- **Breakpoint context** encodes that real fusion breakpoints occur
  preferentially at exon boundaries.
- **The interaction term** exists because a canonical splice motif is only
  informative at an exon-exon junction; multiplying by both exon flags gives the
  model a clean signal there and silence elsewhere.

## Feature normalisation

Continuous features are z-scored with mean and standard deviation pooled over
*all* training graphs, not per sample. Per-sample standardisation would erase
the absolute scale of the evidence, which is precisely what separates a junction
with fifty split reads from one with three. Binary indicators and one-hot blocks
are left untouched so their 0/1 semantics reach the model intact.

The statistics are stored in the checkpoint and replayed unchanged at inference,
so a sample is scored on the same scale the model was trained on.

## Architecture

The network combines an edge-local pathway with graph context. A fusion is
mostly a local event — the evidence is in the junction — so context refines the
decision rather than determining it.

1. **Edge encoder.** A small MLP maps the heterogeneous per-edge vector into a
   dense embedding, used both to condition attention and as direct input to the
   classifier.
2. **Node encoder and GATv2 layers.** A small MLP embeds each gene; two GATv2
   message-passing layers refine it, with attention coefficients conditioned on
   the edge embedding, so neighbour importance depends jointly on gene identity
   and junction quality. GATv2 is used rather than the original GAT because its
   dynamic attention lets relevance depend on the donor-acceptor combination
   rather than on the neighbour alone. Depth is limited to two layers to avoid
   oversmoothing on small cohorts.
3. **Edge classifier.** For each edge, the refined donor and acceptor
   representations are concatenated with their pre-message-passing embeddings (a
   node-skip connection guarding against oversmoothing) and the edge embedding,
   then passed through an MLP producing a logit.
4. **Normalisation.** Layer normalisation throughout, not batch normalisation:
   mini-batches merge graphs of very different sizes, so batch statistics shift
   with batch composition and open a train/eval gap that moves the optimal
   threshold. Layer normalisation is per-sample and free of that effect.

### Evidence gate

A monotone path from raw split-read support to the final logit:

```
z' = z + softplus(g) * log(1 + s)
```

where *z* is the model logit, *s* the split-read count and *g* a learnable
scalar. The softplus keeps the weight non-negative, so stronger read support can
only ever raise a candidate's score. The gate consumes the *un-normalised*
evidence — that is why the graph carries `edge_evidence` as a separate tensor —
because standardisation would dilute the very signal the gate exists to protect.

### Classifier bias initialisation

The final layer's bias starts at the empirical log-odds of the positive rate in
the training set. At ratios of 1:2000, starting from the default zero bias (a
50/50 prediction) wastes many epochs on an adjustment that is known in advance.

## Training

- **Focal loss** (γ = 2.0, α = 0.75) by default. With thousands of negatives per
  positive, plain cross-entropy is minimised by a trivial noise-only classifier;
  focal loss down-weights easy negatives so the gradient comes from candidates
  near the decision boundary. `--loss bce` switches to weighted cross-entropy
  with negative subsampling instead.
- **Regularisation**: dropout, L2 weight decay, and Gaussian noise injected into
  the continuous features of positive edges each epoch, so the model learns a
  feature region rather than memorising a few dozen exact points.
- **Optimisation**: Adam with cosine-annealed learning rate, gradient clipping
  at 1.0, mini-batches of graphs merged by PyTorch Geometric into one
  disconnected graph per batch.
- **Model selection**: AUPRC on held-out samples, which is more stable than F1
  when positives are scarce. Early stopping when it fails to improve.
- **Cross-validation**: 5 folds stratified by whether a sample has any confirmed
  fusion. Fold predictions are pooled before choosing the operating threshold,
  so the threshold is set on every sample rather than on one arbitrary split.
  A final model is then refit on the whole cohort.

## Calibration

Raw sigmoid outputs rank candidates well but are not probabilities. Platt
scaling fits *p = σ(A·z + B)*, and the coefficients are stored in the checkpoint
so reported scores approximate the probability that a candidate is genuine.

## Post-processing

Rule-based filters remove artefact classes that established callers also discard:
intragenic self-joins, read-through transcripts between adjacent co-oriented
genes, artefact-prone gene families, and inter-chromosomal exon-exon junctions
with no canonical splice signal (the signature of paralogous cross-mapping).

Filtering runs *before* threshold or top-k selection, so artefacts cannot crowd
genuine candidates out of a top-k list. Every filter can be disabled
individually, and `--annotate-only` keeps all rows while recording what would
have been removed.

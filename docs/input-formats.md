# Input formats

## Required files

GATFuse consumes three STAR outputs plus the reference annotation used for the
alignment.

### 1. Alignment BAM (`-b, --bam`)

A coordinate-sorted BAM, used only to count inter-chromosomal discordant mate
pairs — independent evidence supporting inter-chromosomal candidates. Only the
first mate of each pair is counted, so a pair contributes once. An index is not
required, because the file is read start to end.

Produced by `--outSAMtype BAM SortedByCoordinate`.

### 2. Chimeric junctions (`-c, --chimeric-junctions`)

STAR's `Chimeric.out.junction`, the candidate set GATFuse classifies. Columns
used:

| Column | Meaning |
|---|---|
| 1-3 | donor chromosome, breakpoint, strand |
| 4-6 | acceptor chromosome, breakpoint, strand |
| 7 | junction type (`-1` mate-spanning, `0` non-canonical, `1` GT/AG, `2` CT/AC) |
| 8-9 | repeat length left and right of the junction |
| 10 | read name (counted to obtain split-read support) |
| 12, 14 | CIGAR strings, used for the anchor-balance feature |

Reads sharing a breakpoint pair are collapsed into one junction. The column
header line written by STAR ≥ 2.7.1 and the `#` summary block appended by
`--chimOutJunctionFormat 1` are both skipped.

### 3. Gene counts (`-e, --gene-counts`)

STAR's `ReadsPerGene.out.tab` from `--quantMode GeneCounts`. Determines which
genes are expressed — only genes with positive TPM become nodes — and supplies
the sequencing depth used to normalise evidence into FFPM. The leading `N_*`
summary rows are dropped.

Column 2 is used by default. Set `--library-type stranded_forward` or
`stranded_reverse` to read column 3 or 4 instead. Choosing the wrong column
mainly affects which genes appear as nodes.

### 4. Annotation (`-g, --annotation`)

The GTF that built the STAR index, in plain or gzipped form. It maps breakpoints
onto genes and supplies gene length, biotype and exon structure.

**It must be the same annotation used for the alignment.** GATFuse checks this
and stops with an explanatory error when:

- no contig name is shared between the junction file and the GTF (the classic
  UCSC `chr1` versus Ensembl `1` mismatch), or
- no gene ID in the count file appears in the GTF.

It warns, rather than failing, when fewer than half the counted gene IDs are
annotated, which usually means the two came from different releases.

#### Biotype attribute

Ensembl GTFs name the biotype `gene_biotype`; GENCODE names it `gene_type`.
`--biotype-attributes` selects which is read:

- `ensembl` — read `gene_biotype` only.
- `auto` — read either; the default when training.

The attribute drives the protein-coding node feature and the preference for
protein-coding genes when a breakpoint overlaps several genes. `gatfuse detect`
takes this setting from the checkpoint, so the graph is always built the way the
model expects. See [reproducibility.md](reproducibility.md) for why the
distributed model records `ensembl`.

## Required STAR parameters

Chimeric detection is off by default in STAR. `scripts/run_star.sh` runs it with
the parameters the distributed model was trained on:

| Parameter | Value | Purpose |
|---|---|---|
| `--chimSegmentMin` | 10 | minimum segment length to call a chimeric read |
| `--chimJunctionOverhangMin` | 10 | minimum overhang each side of the junction |
| `--chimScoreMin` | 1 | minimum chimeric alignment score |
| `--chimScoreDropMax` | 30 | maximum allowed score drop |
| `--chimOutType` | `Junctions WithinBAM SoftClip` | emit the junction file |
| `--chimOutJunctionFormat` | 1 | append the summary header |
| `--quantMode` | `GeneCounts` | produce `ReadsPerGene.out.tab` |
| `--outFilterMultimapNmax` | 20 | maximum loci per read |
| `--outSAMmapqUnique` | 60 | MAPQ for unique alignments |

More permissive chimeric settings yield more candidates and a noisier graph;
stricter ones can remove a true junction before GATFuse ever sees it.

## Known-fusion catalogue (`--known-fusions`)

Optional prior knowledge, enabling two node features (is the gene a catalogued
fusion partner; how many distinct partners it has) and two edge features (is the
pair catalogued; its recurrence). A snapshot of the Mitelman Database ships with
the package and is used by default.

A tab- or comma-separated file with a header; the delimiter is detected
automatically and `#` lines are ignored:

```
gene_a	gene_b	n_cases
ABL1	BCR	333
RUNX1	RUNX1T1	103
```

`n_cases` is optional; without it every catalogued pair counts once. Pairs are
unordered, so listing `BCR/ABL1` also covers `ABL1/BCR`.

Whether the catalogue is used at all is a property of the model, not of the run:
a model trained with these features cannot be applied without them, and GATFuse
says so explicitly rather than failing on a shape mismatch. Refresh the
catalogue with `scripts/update_known_fusions.py`, but note that a new snapshot
changes the feature distribution and therefore calls for retraining.

## Training manifest (`-m, --manifest`)

One sample per row, tab- or comma-separated.

| Column | Required | Contents |
|---|---|---|
| `sample_id` | no | label used in logs; defaults to `sample_001`, ... |
| `bam` | yes | coordinate-sorted BAM |
| `chimeric_junctions` | yes | `Chimeric.out.junction` |
| `gene_counts` | yes | `ReadsPerGene.out.tab` |
| `annotation` | no | per-sample GTF; falls back to `--annotation` |
| `fusions` | yes* | semicolon-separated confirmed fusions |
| `fusions_file` | yes* | file with one fusion per line, instead of `fusions` |

\* one of the two is required, and at least one sample must have labels.

Relative paths resolve against the directory containing the manifest, so a
manifest and its alignments move between machines together.

Column names are matched case-insensitively, and each accepts aliases:
`bam_file`, `alignment`, `alignments` for `bam`; `chimeric_file`, `chimeric`,
`junctions` for `chimeric_junctions`; `reads_per_gene_file`, `reads_per_gene`,
`counts` for `gene_counts`; `gtf_file`, `gtf` for `annotation`;
`positive_fusions`, `known_fusions`, `positives` for `fusions`;
`positive_fusions_file` for `fusions_file`; and `sample`, `name`,
`sample_name` for `sample_id`.

### Fusion label syntax

```
DONOR:ACCEPTOR                              gene-level label
DONOR:ACCEPTOR@chrA:posA-chrB:posB          breakpoint-level label
```

Partners may be gene symbols or Ensembl gene IDs. A gene-level label marks every
junction between the pair positive, in either direction. A breakpoint-level
label marks only junctions within 100 bp of both reported coordinates, leaving
other junctions between the same genes as negatives — which is what you want
when a gene pair produces several distinct junctions and only one is real.

Example:

```
sample_id	bam	chimeric_junctions	gene_counts	fusions
K562	aln/K562.bam	aln/K562.junction	aln/K562.tab	BCR:ABL1
VCaP	aln/VCaP.bam	aln/VCaP.junction	aln/VCaP.tab	TMPRSS2:ERG@chr21:42870046-chr21:39956869
```

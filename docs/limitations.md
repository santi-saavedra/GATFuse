# Known limitations

## Method

**Conservative by design.** GATFuse trades recall for precision. The evidence
gate and the calibrated threshold concentrate attention on a few high-confidence
candidates; in benchmarking it returned a mean of 2.6 candidates per sample
against 38.1 for Arriba, 17.4 for STAR-Fusion and 74.1 for FusionCatcher, at a
lower overall detection rate (56.7% of samples versus 96.7%, 90.0% and 86.7%).
A fusion with modest read support can be missed outright rather than merely
ranked low.

If exhaustive recall matters more than a short list, lower `--threshold`, or use
`--top-k` and review the ranking instead of the absolute score.

**Partner order is not resolved.** Both orientations of a confirmed fusion are
labelled positive during training, so the classifier never learns which partner
is 5'. The reported order is assigned after scoring from STAR's strand
annotation, which follows the genomic splice motif rather than the direction of
transcription. In benchmarking, 7 of the 17 recovered fusions were reported
transposed (RUNX1T1-RUNX1 for the described RUNX1-RUNX1T1, for instance) with
correct breakpoints, and the same fusion can be reported in opposite
orientations in different samples. Treat a call as an unordered gene pair.

**Only annotated genes.** Nodes are genes present in the GTF with positive
expression. A junction in an unannotated region, or involving a gene absent from
your annotation, has no edge and cannot be reported. A comprehensive annotation
matters.

**Requires both partners expressed.** Nodes are restricted to genes with
positive TPM. A fusion whose 3' partner is not otherwise transcribed may lack a
node and be missed.

**Discordant-pair evidence is inter-chromosomal only.** The BAM scan collects
pairs whose mates map to different chromosomes. Intra-chromosomal fusions
therefore rely on split reads alone, and their `discordant_pairs` column is 0.

**No fusion-transcript reconstruction.** GATFuse classifies junctions. It does
not assemble the fusion transcript, predict reading frame, or annotate protein
domains. Pair it with a downstream annotator when those matter.

## Training data

The distributed model was trained on 101 samples: predominantly breast cancer
cell lines and glioblastoma, plus one synthetic sample. Fusion patterns
characteristic of other tumour types are not represented, and generalisation to
them is not established. Consider retraining ([training.md](training.md)) if
your setting differs materially.

The ground truth is literature-curated, so it is necessarily incomplete: an
apparent false positive may be a genuine fusion that was never reported, and an
apparent false negative may reflect a missing label rather than a missed event.

## Inputs

**Coupled to STAR.** Graph construction consumes STAR's `Chimeric.out.junction`,
`ReadsPerGene.out.tab` and BAM. Supporting another splice-aware aligner would
require a new parser; the rest of the pipeline is aligner-agnostic.

**Chimeric parameters must match.** More permissive STAR chimeric settings
produce more, noisier candidates than the model saw in training; stricter ones
can remove a true junction before GATFuse sees it. Use `scripts/run_star.sh`.

**The annotation must be the alignment's annotation.** Contig names and gene IDs
have to match. GATFuse detects the two common failures (no shared contig names,
no shared gene IDs) and stops with an explanatory error, but subtler mismatches —
a GTF from a neighbouring release — only produce a warning.

## Practical

**Memory scales with the annotation.** A full GENCODE GTF needs roughly 4-6 GB
while parsing. `--workers > 1` multiplies that, since each worker parses its own
copy.

**Runtime is dominated by the BAM scan**, not by the network, so a GPU helps
little. See [performance.md](performance.md).

**Microhomology breakpoints may differ by a base or two** from Arriba or
STAR-Fusion. Each caller anchors differently within an ambiguous repeat; this is
caller disagreement, not an error.

**Cache keys use file size, not content.** An input edited in place without
changing size would not invalidate its cache entry. Clear `--cache-dir` after
such an edit.

## Interpretation

**Scores are calibrated, not certainties.** Platt scaling makes them approximate
the probability that a candidate is genuine, fitted on the training cohort. On
materially different data that calibration will drift, and the ranking remains
more trustworthy than the absolute value.

**Filtered categories can contain real biology.** Read-through transcripts and
intragenic junctions are removed by default because they are usually artefacts
of continuous transcription, but conjoined-gene transcripts are genuine
phenomena. Use `--discarded-output` or `--annotate-only` when they matter.

**An empty result is not proof of absence.** It means no candidate passed the
threshold given the evidence in this library. Sequencing depth, expression level
and the STAR chimeric settings all bound what can be detected.

# Examples

`manifest.example.tsv` documents the training manifest format:

- Paths may be absolute, or relative to the manifest file itself, so a manifest
  travels with its data.
- `fusions` is a semicolon-separated list of confirmed fusions for that sample.
  Gene-level labels (`BCR:ABL1`) mark every junction between the pair as
  positive; breakpoint-level labels
  (`TMPRSS2:ERG@chr21:42870046-chr21:39956869`) mark only junctions matching
  those coordinates, leaving other junctions between the same genes as
  negatives.
- Column names are case-insensitive and accept aliases (`bam_file`,
  `chimeric_file`, `reads_per_gene_file`, `positive_fusions` and others);
  `docs/input-formats.md` lists them.

See `docs/input-formats.md` for the full specification and
`docs/training.md` for how to train on your own cohort.

A ready-to-run miniature dataset (annotation, chimeric junctions, gene counts
and alignments for five genes) lives in `tests/data/` and is what the test suite
exercises; it is far too small for meaningful predictions but is useful to
confirm an installation works.

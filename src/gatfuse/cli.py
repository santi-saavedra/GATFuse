"""Command-line interface.

    gatfuse detect   score one sample against a trained model
    gatfuse train    fit a model on a manifest of labelled samples
"""

import argparse
import sys

from . import __version__
from .config import (
    BIOTYPE_ATTRIBUTE_MODES,
    LIBRARY_TYPES,
    GraphParams,
    ModelParams,
    PostprocessParams,
    TrainParams,
)
from .errors import EXIT_ERROR, EXIT_SUCCESS, GATFuseError, UsageError
from .logging_utils import LEVELS, configure_logging, get_logger
from .runtime import DEVICE_CHOICES

log = get_logger("cli")

_DETECT_EXAMPLE = """\
examples:
  # Score a sample with the bundled pretrained model
  gatfuse detect -b sample.Aligned.sortedByCoord.out.bam \\
                 -c sample.Chimeric.out.junction \\
                 -e sample.ReadsPerGene.out.tab \\
                 -g annotation.gtf \\
                 -o fusions.tsv

  # Report the 10 highest-scoring candidates regardless of score
  gatfuse detect -b a.bam -c a.junction -e a.tab -g ref.gtf --top-k 10

  # Keep every candidate and record why each would have been filtered
  gatfuse detect -b a.bam -c a.junction -e a.tab -g ref.gtf \\
                 --annotate-only --threshold 0
"""

_TRAIN_EXAMPLE = """\
examples:
  # Five-fold cross-validation, then a final model on all samples
  gatfuse train -m cohort.tsv -g annotation.gtf -o models/my-model.pt

  # Single held-out split instead of cross-validation
  gatfuse train -m cohort.tsv -g annotation.gtf -o models/my-model.pt \\
                --cv-folds 0 --val-split 0.2
"""


def _add_common(parser):
    group = parser.add_argument_group("general")
    group.add_argument("--log-level", choices=LEVELS, default="info",
                       help="verbosity of the progress log written to stderr (default: info)")
    group.add_argument("--version", action="version", version=f"GATFuse {__version__}")
    return group


def _add_performance(parser, with_workers=False):
    group = parser.add_argument_group("performance")
    group.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                       help="compute device; 'auto' prefers CUDA, then Apple MPS (default: auto)")
    group.add_argument("--threads", type=int, default=None,
                       help="PyTorch intra-op threads (default: PyTorch's own choice)")
    group.add_argument("--bam-threads", type=int, default=4,
                       help="BAM decompression threads; the dominant cost of the "
                            "discordant-pair scan (default: 4)")
    if with_workers:
        group.add_argument("--workers", type=int, default=1,
                           help="parallel processes for graph construction, one sample each "
                                "(default: 1)")
    group.add_argument("--cache-dir", default=None,
                       help="directory for cached graphs; reused when inputs and parameters "
                            "are unchanged (default: no caching)")
    return group


def _add_graph_options(parser, defaults_from_model=False):
    group = parser.add_argument_group("graph construction")
    suffix = " (default: the value recorded in the model)" if defaults_from_model else ""
    default_params = GraphParams()
    group.add_argument("--min-split-reads", type=int,
                       default=None if defaults_from_model else default_params.min_split_reads,
                       help=f"discard chimeric junctions with fewer supporting split reads{suffix}")
    group.add_argument("--library-type", choices=LIBRARY_TYPES,
                       default=None if defaults_from_model else default_params.library_type,
                       help=f"strandedness of the RNA-seq library{suffix}")
    group.add_argument("--biotype-attributes", choices=sorted(BIOTYPE_ATTRIBUTE_MODES),
                       default=None if defaults_from_model else "auto",
                       help="GTF attributes read for the gene biotype: 'ensembl' reads "
                            "gene_biotype only, 'auto' also reads GENCODE's gene_type"
                            f"{suffix}")
    return group


def _add_postprocess_options(parser):
    group = parser.add_argument_group("post-processing filters")
    defaults = PostprocessParams()
    group.add_argument("--no-postprocess", action="store_true",
                       help="report raw model output without any artefact filter")
    group.add_argument("--annotate-only", action="store_true",
                       help="keep every candidate but add a filter_reason column")
    group.add_argument("--keep-intragenic", action="store_true",
                       help="keep junctions whose donor and acceptor are the same gene")
    group.add_argument("--keep-readthrough", action="store_true",
                       help="keep adjacent same-strand read-through transcripts")
    group.add_argument("--keep-blacklisted", action="store_true",
                       help="keep junctions involving ribosomal, mitochondrial and similar "
                            "artefact-prone gene families")
    group.add_argument("--keep-noncanonical", action="store_true",
                       help="keep inter-chromosomal exon-exon junctions with no canonical "
                            "splice motif")
    group.add_argument("--readthrough-max-distance", type=int,
                       default=defaults.readthrough_max_distance,
                       help="maximum breakpoint distance treated as read-through, in bp "
                            f"(default: {defaults.readthrough_max_distance})")
    group.add_argument("--noncanonical-min-canonical", type=float,
                       default=defaults.noncanonical_min_canonical_fraction,
                       help="minimum fraction of canonical-motif reads required to keep an "
                            "inter-chromosomal exon-exon junction (default: "
                            f"{defaults.noncanonical_min_canonical_fraction})")
    group.add_argument("--blacklist-pattern", action="append", default=[], metavar="REGEX",
                       help="additional gene-symbol pattern to treat as an artefact family; "
                            "repeatable")
    return group


def _add_known_fusion_options(parser):
    group = parser.add_argument_group("known-fusion catalogue")
    group.add_argument("--known-fusions", default=None, metavar="TSV",
                       help="catalogue enabling the prior-knowledge features "
                            "(default: the catalogue bundled with GATFuse)")
    group.add_argument("--no-known-fusions", action="store_true",
                       help="build the graph without prior-knowledge features")
    return group


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gatfuse",
        description=(
            "GATFuse detects gene fusions in RNA-seq data by classifying the chimeric "
            "junctions reported by STAR as edges of a per-sample gene graph."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"GATFuse {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    detect = subparsers.add_parser(
        "detect",
        help="detect fusions in one sample",
        description="Score the chimeric junctions of one RNA-seq sample and report fusions.",
        epilog=_DETECT_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    required = detect.add_argument_group("required input")
    required.add_argument("-b", "--bam", required=True, metavar="BAM",
                          help="coordinate-sorted alignment BAM from STAR")
    required.add_argument("-c", "--chimeric-junctions", required=True, metavar="FILE",
                          help="STAR Chimeric.out.junction file")
    required.add_argument("-e", "--gene-counts", required=True, metavar="FILE",
                          help="STAR ReadsPerGene.out.tab file")
    required.add_argument("-g", "--annotation", required=True, metavar="GTF",
                          help="the GTF annotation used to build the STAR index")

    model_group = detect.add_argument_group("model")
    model_group.add_argument("-m", "--model", default=None, metavar="PT",
                             help="trained checkpoint (default: the model bundled with GATFuse)")
    _add_known_fusion_options(detect)

    output = detect.add_argument_group("output")
    output.add_argument("-o", "--output", default="gatfuse_fusions.tsv", metavar="TSV",
                        help="reported fusions (default: gatfuse_fusions.tsv)")
    output.add_argument("-d", "--discarded-output", default=None, metavar="TSV",
                        help="candidates removed by the post-processing filters, with the "
                             "reason for each")

    selection = detect.add_argument_group("candidate selection")
    selection.add_argument("-t", "--threshold", type=float, default=None,
                           help="minimum score to report (default: the cross-validation "
                                "threshold stored in the model)")
    selection.add_argument("-k", "--top-k", type=int, default=None,
                           help="report the K highest-scoring candidates instead of applying "
                                "a threshold; takes precedence over --threshold")

    _add_graph_options(detect, defaults_from_model=True)
    _add_postprocess_options(detect)
    _add_performance(detect)
    _add_common(detect)

    train = subparsers.add_parser(
        "train",
        help="train a model on a cohort",
        description="Fit the edge classifier on a manifest of samples with confirmed fusions.",
        epilog=_TRAIN_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_required = train.add_argument_group("required input")
    train_required.add_argument("-m", "--manifest", required=True, metavar="TSV",
                                help="one sample per row; see docs/input-formats.md")
    train_required.add_argument("-g", "--annotation", default=None, metavar="GTF",
                                help="annotation for rows without their own annotation column")
    train_required.add_argument("-o", "--output-model", default="gatfuse-model.pt", metavar="PT",
                                help="where to write the checkpoint (default: gatfuse-model.pt)")
    train.add_argument("--resume-from", default=None, metavar="PT",
                       help="continue training from an existing checkpoint")
    _add_known_fusion_options(train)

    architecture = train.add_argument_group("architecture")
    model_defaults = ModelParams()
    architecture.add_argument("--hidden-dim", type=int, default=model_defaults.hidden_dim,
                              help=f"node embedding width (default: {model_defaults.hidden_dim})")
    architecture.add_argument("--edge-embed-dim", type=int, default=model_defaults.edge_embed_dim,
                              help=f"junction embedding width (default: {model_defaults.edge_embed_dim})")
    architecture.add_argument("--dropout", type=float, default=model_defaults.dropout,
                              help=f"dropout probability (default: {model_defaults.dropout})")
    architecture.add_argument("--heads", type=int, default=model_defaults.heads,
                              help=f"GATv2 attention heads (default: {model_defaults.heads})")
    architecture.add_argument("--gnn-layers", type=int, default=model_defaults.num_gnn_layers,
                              help="message-passing layers; deeper stacks oversmooth on small "
                                   f"cohorts (default: {model_defaults.num_gnn_layers})")
    architecture.add_argument("--no-node-skip", action="store_false", dest="node_skip",
                              help="do not feed pre-message-passing node embeddings to the classifier")
    architecture.set_defaults(node_skip=model_defaults.node_skip)

    optimisation = train.add_argument_group("optimisation")
    train_defaults = TrainParams()
    optimisation.add_argument("--epochs", type=int, default=train_defaults.epochs,
                              help=f"maximum epochs (default: {train_defaults.epochs})")
    optimisation.add_argument("--learning-rate", type=float, default=train_defaults.learning_rate,
                              help=f"Adam learning rate (default: {train_defaults.learning_rate})")
    optimisation.add_argument("--batch-size", type=int, default=train_defaults.batch_size,
                              help=f"graphs per mini-batch (default: {train_defaults.batch_size})")
    optimisation.add_argument("--patience", type=int, default=train_defaults.patience,
                              help=f"early-stopping patience in epochs (default: {train_defaults.patience})")
    optimisation.add_argument("--weight-decay", type=float, default=train_defaults.weight_decay,
                              help=f"L2 weight decay (default: {train_defaults.weight_decay})")
    optimisation.add_argument("--loss", choices=("focal", "bce"), default=train_defaults.loss,
                              help=f"loss function (default: {train_defaults.loss})")
    optimisation.add_argument("--focal-gamma", type=float, default=train_defaults.focal_gamma,
                              help=f"focal loss focusing parameter (default: {train_defaults.focal_gamma})")
    optimisation.add_argument("--focal-alpha", type=float, default=train_defaults.focal_alpha,
                              help="focal loss positive-class weight "
                                   f"(default: {train_defaults.focal_alpha})")
    optimisation.add_argument("--negative-ratio", type=int, default=train_defaults.negative_ratio,
                              help="negatives kept per positive; only used with --loss bce "
                                   f"(default: {train_defaults.negative_ratio})")
    optimisation.add_argument("--feature-noise-std", type=float,
                              default=train_defaults.feature_noise_std,
                              help="standard deviation of the Gaussian noise added to positive "
                                   "edges each epoch; 0 disables it "
                                   f"(default: {train_defaults.feature_noise_std})")

    evaluation = train.add_argument_group("evaluation")
    evaluation.add_argument("--cv-folds", type=int, default=train_defaults.cv_folds,
                            help="cross-validation folds; 0 uses a single --val-split hold-out "
                                 f"(default: {train_defaults.cv_folds})")
    evaluation.add_argument("--val-split", type=float, default=train_defaults.val_split,
                            help="validation fraction when cross-validation is off "
                                 f"(default: {train_defaults.val_split})")
    evaluation.add_argument("--select-metric", choices=("auprc", "f1"),
                            default=train_defaults.select_metric,
                            help="metric driving model selection; AUPRC is more stable with few "
                                 f"positives (default: {train_defaults.select_metric})")
    evaluation.add_argument("--keep-empty-graphs", action="store_false", dest="drop_empty_graphs",
                            help="keep samples that have no confirmed fusion in the training set")
    evaluation.set_defaults(drop_empty_graphs=train_defaults.drop_empty_graphs)
    evaluation.add_argument("--seed", type=int, default=train_defaults.seed,
                            help="random seed for splits and initialisation "
                                 f"(default: {train_defaults.seed})")

    _add_graph_options(train)
    _add_performance(train, with_workers=True)
    _add_common(train)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return EXIT_SUCCESS

    configure_logging(args.log_level)
    try:
        if args.command == "detect":
            from .commands.detect_command import run

            return run(args)
        if args.command == "train":
            from .commands.train_command import run

            return run(args)
        raise UsageError(f"Unknown command: {args.command}")
    except GATFuseError as exc:
        log.error("%s", exc)
        return exc.exit_code
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - reported, then re-raised for the traceback
        log.error("Unexpected error: %s", exc)
        if args.log_level == "debug":
            raise
        log.error("Re-run with --log-level debug for a full traceback.")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

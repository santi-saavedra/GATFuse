"""``gatfuse detect``: score one sample and report its fusions."""

from dataclasses import replace

from .. import checkpoint as checkpoint_io
from .. import detect as detection
from ..config import PostprocessParams
from ..errors import EXIT_SUCCESS, UsageError
from ..io import load_known_fusions
from ..logging_utils import get_logger
from ..pipeline import AnnotationLoader, SampleInputs, build_or_load_graph
from ..resources import bundled_known_fusions_path, bundled_model_path
from ..runtime import resolve_device, set_torch_threads

log = get_logger("detect")


def _graph_params(args, checkpoint):
    """Training-time parameters, overridden only where the user was explicit."""
    params = checkpoint.graph_params
    overrides = {}
    if args.min_split_reads is not None:
        overrides["min_split_reads"] = args.min_split_reads
    if args.library_type is not None:
        overrides["library_type"] = args.library_type
    if args.biotype_attributes is not None:
        overrides["biotype_attributes"] = args.biotype_attributes

    expects_catalogue = checkpoint.uses_known_fusions
    if args.no_known_fusions:
        overrides["use_known_fusions"] = False
    elif expects_catalogue is not None:
        overrides["use_known_fusions"] = expects_catalogue

    updated = replace(params, **overrides).validate()
    for name, value in overrides.items():
        if getattr(params, name) != value:
            log.info("Overriding recorded %s: %s -> %s", name, getattr(params, name), value)
    return updated


def _postprocess_params(args):
    if args.no_postprocess:
        return None
    return PostprocessParams(
        drop_intragenic=not args.keep_intragenic,
        drop_readthrough=not args.keep_readthrough,
        drop_blacklisted=not args.keep_blacklisted,
        drop_noncanonical=not args.keep_noncanonical,
        noncanonical_min_canonical_fraction=args.noncanonical_min_canonical,
        readthrough_max_distance=args.readthrough_max_distance,
        annotate_only=args.annotate_only,
        extra_blacklist=tuple(args.blacklist_pattern),
    ).validate()


def run(args):
    if args.top_k is not None and args.top_k < 1:
        raise UsageError("--top-k must be >= 1")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise UsageError("--threshold must be in [0, 1]")
    if args.top_k is not None and args.threshold is not None:
        log.warning("--top-k and --threshold both given; --top-k takes precedence.")

    set_torch_threads(args.threads)
    device = resolve_device(args.device)

    checkpoint = checkpoint_io.load(args.model or bundled_model_path())
    params = _graph_params(args, checkpoint)

    known_fusions = None
    if params.use_known_fusions:
        known_fusions = load_known_fusions(
            args.known_fusions or bundled_known_fusions_path(),
            use_recurrence=params.known_fusion_recurrence,
        )

    inputs = SampleInputs(
        bam=args.bam,
        chimeric_junctions=args.chimeric_junctions,
        gene_counts=args.gene_counts,
        annotation=args.annotation,
        sample_id=args.output,
    )
    annotation_loader = AnnotationLoader(params.biotype_attributes)
    graph = build_or_load_graph(
        inputs,
        params=params,
        known_fusions=known_fusions,
        cache=args.cache_dir,
        annotation_loader=annotation_loader,
        bam_threads=args.bam_threads,
    )
    annotation = annotation_loader.get(args.annotation)

    threshold = args.threshold
    if threshold is None and args.top_k is None:
        threshold = checkpoint.threshold
        if threshold is None:
            threshold = 0.5
            log.warning("Model carries no validation threshold; falling back to %.2f.", threshold)
        else:
            log.info("Using the model's cross-validation threshold: %.4f", threshold)

    result = detection.detect(
        graph, checkpoint, annotation,
        threshold=threshold, top_k=args.top_k,
        postprocess=_postprocess_params(args), device=device,
    )

    detection.write_table(result.reported, args.output, "reported fusions")
    if args.discarded_output is not None:
        detection.write_table(result.discarded, args.discarded_output, "discarded candidates")

    if result.reported.empty:
        log.warning(
            "No fusion reported. Lower --threshold or use --top-k to inspect the "
            "highest-scoring candidates."
        )
    else:
        top = result.reported.head(5)
        log.info("Top candidates:")
        for row in top.itertuples(index=False):
            log.info(
                "  %-14s %-14s score=%.4f split_reads=%-4d %s",
                row.donor_gene, row.acceptor_gene, row.score, row.split_reads, row.fusion_type,
            )
    log.info(
        "Done: %d fusion(s) reported out of %d candidate junction(s).",
        len(result.reported), result.candidates_evaluated,
    )
    return EXIT_SUCCESS

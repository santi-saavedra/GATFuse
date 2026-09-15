"""``gatfuse train``: fit a model on a cohort of labelled samples."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from .. import checkpoint as checkpoint_io
from ..config import GraphParams, ModelParams, TrainParams
from ..errors import EXIT_SUCCESS, InputError
from ..features import continuous_edge_columns
from ..io import load_known_fusions, load_manifest
from ..io.paths import prepare_output
from ..labels import label_edges
from ..logging_utils import get_logger
from ..model import FusionPredictor, fit_platt_scaling
from ..normalization import apply_edge_norm, compute_edge_norm_stats
from ..pipeline import AnnotationLoader, SampleInputs, build_or_load_graph
from ..resources import bundled_known_fusions_path
from ..runtime import resolve_device, set_seed, set_torch_threads
from ..train import cross_validate, fit, stratified_split

log = get_logger("train")


def _prepare_graph(row, params, known_fusions, cache_dir, annotation_loader, bam_threads):
    """Build one sample's graph and attach its edge labels."""
    inputs = SampleInputs(
        bam=row.bam,
        chimeric_junctions=row.chimeric_junctions,
        gene_counts=row.gene_counts,
        annotation=row.annotation,
        sample_id=row.sample_id,
    )
    fusion_signature = ";".join(":".join(str(part) for part in f) for f in row.fusions)
    graph = build_or_load_graph(
        inputs, params=params, known_fusions=known_fusions, cache=cache_dir,
        annotation_loader=annotation_loader, bam_threads=bam_threads,
        cache_extra=f"labels={fusion_signature}",
    )
    if graph.edge_index.shape[1] == 0:
        log.warning("%s: no chimeric junction mapped to a gene pair; sample skipped.",
                    row.sample_id)
        return None

    if getattr(graph, "y", None) is None:
        graph.y = label_edges(
            graph.edge_index, row.fusions, graph.gene_to_node_id,
            gene_name_to_node_id=graph.gene_name_to_node_id,
            edge_breakpoints=graph.edge_breakpoints,
            sample_id=row.sample_id,
        )
        for attribute in ("gene_to_node_id", "gene_name_to_node_id", "edge_breakpoints"):
            if getattr(graph, attribute, None) is not None:
                delattr(graph, attribute)
    return graph


def _worker(payload):
    row, params, known_fusions, cache_dir, bam_threads = payload
    loader = AnnotationLoader(params.biotype_attributes)
    return _prepare_graph(row, params, known_fusions, cache_dir, loader, bam_threads)


def _build_dataset(rows, params, known_fusions, args):
    dataset = []
    if args.workers and args.workers > 1:
        log.info("Building %d graphs with %d worker processes", len(rows), args.workers)
        log.warning(
            "Each worker parses the annotation independently; peak memory scales with --workers."
        )
        payloads = [
            (row, params, known_fusions, args.cache_dir, args.bam_threads) for row in rows
        ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = [executor.submit(_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                graph = future.result()
                if graph is not None:
                    dataset.append(graph)
    else:
        loader = AnnotationLoader(params.biotype_attributes)
        for index, row in enumerate(rows, start=1):
            log.info("[%d/%d] %s", index, len(rows), row.sample_id)
            graph = _prepare_graph(
                row, params, known_fusions, args.cache_dir, loader, args.bam_threads
            )
            if graph is not None:
                dataset.append(graph)

    if not dataset:
        raise InputError("No sample produced a usable graph; nothing to train on.")

    node_features = dataset[0].x.shape[1]
    edge_features = dataset[0].edge_attr.shape[1]
    for index, graph in enumerate(dataset[1:], start=2):
        if graph.x.shape[1] != node_features or graph.edge_attr.shape[1] != edge_features:
            raise InputError(
                f"Sample {index} has {graph.x.shape[1]}/{graph.edge_attr.shape[1]} node/edge "
                f"features but the first sample has {node_features}/{edge_features}. "
                "Clear the graph cache if it holds graphs from another feature version."
            )
    return dataset


def run(args):
    params = GraphParams(
        min_split_reads=args.min_split_reads,
        library_type=args.library_type,
        biotype_attributes=args.biotype_attributes,
        use_known_fusions=not args.no_known_fusions,
        known_fusion_recurrence=True,
    ).validate()
    model_params = ModelParams(
        hidden_dim=args.hidden_dim,
        edge_embed_dim=args.edge_embed_dim,
        dropout=args.dropout,
        heads=args.heads,
        num_gnn_layers=args.gnn_layers,
        node_skip=args.node_skip,
    ).validate()
    train_params = TrainParams(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience,
        weight_decay=args.weight_decay,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        negative_ratio=args.negative_ratio,
        feature_noise_std=args.feature_noise_std,
        select_metric=args.select_metric,
        cv_folds=args.cv_folds,
        val_split=args.val_split,
        drop_empty_graphs=args.drop_empty_graphs,
        seed=args.seed,
    ).validate()

    set_seed(train_params.seed)
    set_torch_threads(args.threads)
    device = resolve_device(args.device)

    output_path = prepare_output(args.output_model, "output model")
    rows = load_manifest(args.manifest, default_annotation=args.annotation)

    known_fusions = None
    if params.use_known_fusions:
        known_fusions = load_known_fusions(
            args.known_fusions or bundled_known_fusions_path(),
            use_recurrence=params.known_fusion_recurrence,
        )

    dataset = _build_dataset(rows, params, known_fusions, args)
    log.info("Graphs ready: %d of %d samples", len(dataset), len(rows))

    use_known_fusions = params.use_known_fusions
    continuous_cols = continuous_edge_columns(use_known_fusions)
    metadata = {}

    if train_params.cv_folds and train_params.cv_folds > 1 and len(dataset) > train_params.cv_folds:
        metadata.update(
            cross_validate(dataset, model_params, train_params, use_known_fusions, device)
        )
        final_dataset = [graph.clone() for graph in dataset]
        validation = None
    else:
        if train_params.cv_folds and len(dataset) <= train_params.cv_folds:
            log.warning(
                "Only %d samples for %d folds; falling back to a single validation split.",
                len(dataset), train_params.cv_folds,
            )
        train_ids, val_ids = stratified_split(dataset, train_params.val_split, train_params.seed)
        final_dataset = [dataset[i].clone() for i in train_ids]
        validation = [dataset[i].clone() for i in val_ids] if val_ids else None
        log.info("Hold-out split: %d training, %d validation samples",
                 len(final_dataset), len(validation or []))

    if train_params.drop_empty_graphs:
        kept = [graph for graph in final_dataset if graph.y.sum().item() > 0]
        if not kept:
            raise InputError(
                "No training sample has a labelled fusion edge. Use --keep-empty-graphs, or "
                "check that the manifest fusion names match the annotation."
            )
        if len(kept) < len(final_dataset):
            log.info("Dropped %d sample(s) without positive labels", len(final_dataset) - len(kept))
        final_dataset = kept

    norm_stats = compute_edge_norm_stats(final_dataset, use_known_fusions)
    apply_edge_norm(final_dataset, norm_stats)
    if validation:
        apply_edge_norm(validation, norm_stats)

    positives = sum(int(g.y.sum().item()) for g in final_dataset)
    total = sum(int(g.y.numel()) for g in final_dataset)
    model_config = {
        "num_node_features": final_dataset[0].x.shape[1],
        "num_edge_features": final_dataset[0].edge_attr.shape[1],
        **model_params.to_dict(),
        "prior_pos_rate": positives / max(total, 1),
    }

    optimizer_state = None
    if args.resume_from:
        previous = checkpoint_io.load(args.resume_from)
        if previous.model_config["num_edge_features"] != model_config["num_edge_features"]:
            raise InputError(
                f"Cannot resume: the checkpoint expects {previous.model_config['num_edge_features']} "
                f"edge features, this cohort produces {model_config['num_edge_features']}."
            )
        model = previous.model
        model_config = previous.model_config
        optimizer_state = previous.optimizer_state
        log.info("Resuming from %s", args.resume_from)
    else:
        model = FusionPredictor(**model_config)

    log.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))
    log.info("Positive edges: %d of %d (1:%.0f)", positives, total,
             (total - positives) / max(positives, 1))

    patience = train_params.patience if validation else train_params.patience * 2
    summary = fit(model, final_dataset, validation, train_params, device,
                  continuous_cols, patience=patience, optimizer_state=optimizer_state)

    calibration_set = validation or final_dataset
    platt_params = fit_platt_scaling(model, calibration_set, device)

    metadata.setdefault("best_val_thr", summary["best_val_thr"])
    metadata.setdefault("best_val_auprc", summary["best_val_auprc"])
    metadata.setdefault("best_val_f1", summary["best_val_score"])
    metadata.update({
        "epochs_ran": summary["epochs_ran"],
        "best_loss": summary["best_loss"],
        "best_train_f1": summary["best_train_f1"],
        "num_samples": len(dataset),
        "num_train_graphs": len(final_dataset),
        "num_val_graphs": len(validation or []),
        "manifest": str(args.manifest),
        "edge_norm_stats": norm_stats,
        "platt_params": platt_params,
        "graph_params": params.to_dict(),
        "train_params": train_params.to_dict(),
    })

    checkpoint_io.save(
        output_path, model, model_config, metadata=metadata,
        optimizer_state=summary["optimizer_state_dict"],
    )
    log.info("Recommended inference threshold: %.4f", metadata["best_val_thr"])
    log.info("Validation AUPRC: %.4f", metadata["best_val_auprc"])
    return EXIT_SUCCESS

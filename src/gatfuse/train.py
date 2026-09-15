"""Fitting the edge classifier on a cohort of labelled sample graphs."""

import copy
import random

import torch

from .errors import InputError
from .features import continuous_edge_columns
from .logging_utils import get_logger
from .metrics import best_threshold, summarise
from .model import FocalLoss, FusionPredictor, collect_predictions
from .normalization import apply_edge_norm, compute_edge_norm_stats

log = get_logger("train")

LOG_EVERY = 10


def subsample_negatives(logits, labels, ratio):
    """Keep every positive plus a bounded, refreshed sample of negatives."""
    positive_mask = labels == 1.0
    num_positive = int(positive_mask.sum().item())
    if num_positive == 0 or ratio is None:
        return logits, labels

    negative_indices = (~positive_mask).nonzero(as_tuple=True)[0]
    budget = int(num_positive * ratio)
    if len(negative_indices) <= budget:
        return logits, labels

    num_hard = budget // 2
    num_random = budget - num_hard

    order = torch.argsort(logits[negative_indices].detach().squeeze(), descending=True)
    hard = negative_indices[order[:num_hard]]

    remaining_mask = torch.ones(len(negative_indices), dtype=torch.bool, device=negative_indices.device)
    remaining_mask[order[:num_hard]] = False
    remaining = negative_indices[remaining_mask]
    if len(remaining) > num_random:
        remaining = remaining[torch.randperm(len(remaining), device=remaining.device)[:num_random]]

    keep = torch.cat([positive_mask.nonzero(as_tuple=True)[0], hard, remaining])
    return logits[keep], labels[keep]


def augment_positive_edges(edge_attr, labels, noise_std, continuous_cols):
    """Perturb the continuous features of positive edges with Gaussian noise."""
    if noise_std <= 0 or labels.sum() == 0 or not continuous_cols:
        return edge_attr
    noisy = edge_attr.clone()
    positives = labels.nonzero(as_tuple=True)[0]
    columns = torch.tensor(continuous_cols, device=edge_attr.device)
    noise = torch.randn(len(positives), len(columns), device=edge_attr.device) * noise_std
    noisy[positives[:, None], columns[None, :]] += noise
    return noisy


def _train_f1(model, dataset, device):
    """Best-threshold F1 on the training graphs, used as a tie-break."""
    logits, labels = collect_predictions(model, dataset, device)
    return summarise(torch.sigmoid(logits), labels)["f1_best"]


def _make_criterion(params, dataset, device):
    labels = torch.cat([graph.y for graph in dataset])
    num_positive = int(labels.sum().item())
    num_negative = labels.numel() - num_positive
    if num_positive == 0:
        raise InputError(
            "No positive edge in the training set. Check that the fusions listed in the "
            "manifest use gene symbols or Ensembl IDs present in the annotation."
        )

    if params.loss == "focal":
        criterion = FocalLoss(alpha=params.focal_alpha, gamma=params.focal_gamma).to(device)
        log.info(
            "Training on %d graphs | %d positive / %d negative edges | focal loss (gamma=%.1f, alpha=%.2f)",
            len(dataset), num_positive, num_negative, params.focal_gamma, params.focal_alpha,
        )
        return criterion, None

    weight = min(num_negative / max(num_positive, 1), params.max_pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
    log.info(
        "Training on %d graphs | %d positive / %d negative edges | BCE (pos_weight=%.1f)",
        len(dataset), num_positive, num_negative, weight,
    )
    return criterion, params.negative_ratio or None


def fit(model, train_dataset, val_dataset, params, device, continuous_cols, patience=None,
        optimizer_state=None):
    """Run the optimisation loop and restore the best weights found."""
    from torch_geometric.loader import DataLoader

    patience = params.patience if patience is None else patience
    model = model.to(device)
    criterion, negative_ratio = _make_criterion(params, train_dataset, device)

    loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            log.info("Restored optimiser state from the checkpoint")
        except (ValueError, KeyError) as error:
            log.warning("Could not restore optimiser state (%s); starting a fresh optimiser.", error)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.epochs, eta_min=params.learning_rate * 0.01
    )

    best = {
        "score": -1.0, "loss": float("inf"), "train_f1": -1.0,
        "threshold": 0.5, "auprc": 0.0, "state": None, "epoch": 0,
    }
    stalled = 0
    epochs_ran = 0
    model.train()

    for epoch in range(1, params.epochs + 1):
        epochs_ran = epoch
        epoch_loss = 0.0
        batch_logits, batch_labels = [], []

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            edge_attr = augment_positive_edges(
                batch.edge_attr, batch.y, params.feature_noise_std, continuous_cols
            )
            logits = model(
                batch.x, batch.edge_index, edge_attr,
                edge_evidence=getattr(batch, "edge_evidence", None),
            )
            sub_logits, sub_labels = subsample_negatives(logits, batch.y, negative_ratio)
            loss = criterion(sub_logits, sub_labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            batch_logits.append(logits.detach().cpu())
            batch_labels.append(batch.y.detach().cpu())

        scheduler.step()
        average_loss = epoch_loss / max(len(loader), 1)

        validation = None
        if val_dataset:
            logits, labels = collect_predictions(model, val_dataset, device)
            validation = summarise(torch.sigmoid(logits), labels)

        if epoch % LOG_EVERY == 0:
            train_metrics = summarise(
                torch.sigmoid(torch.cat(batch_logits)), torch.cat(batch_labels).float()
            )
            log.info(
                "Epoch %4d/%d | loss %.4f | train F1 %.3f@%.3f | AUPRC %.3f | lr %.6f",
                epoch, params.epochs, average_loss, train_metrics["f1_best"],
                train_metrics["best_threshold"], train_metrics["auprc"],
                optimizer.param_groups[0]["lr"],
            )
            if validation:
                log.info(
                    "             val F1 %.4f@%.3f (P=%.2f R=%.2f) | val AUPRC %.4f | %d/%d positive",
                    validation["f1_best"], validation["best_threshold"], validation["precision"],
                    validation["recall"], validation["auprc"], validation["num_positive"],
                    validation["num_total"],
                )

        if validation:
            score = validation["auprc"] if params.select_metric == "auprc" else validation["f1_best"]
            if score > best["score"]:
                best.update(
                    score=score, train_f1=_train_f1(model, train_dataset, device),
                    threshold=validation["best_threshold"], auprc=validation["auprc"],
                    state={k: v.cpu().clone() for k, v in model.state_dict().items()},
                    epoch=epoch,
                )
                log.info(
                    "New best model at epoch %d | val %s %.4f | val F1 %.4f@%.3f | train F1 %.4f",
                    epoch, params.select_metric.upper(), score, validation["f1_best"],
                    validation["best_threshold"], best["train_f1"],
                )
                stalled = 0
            else:
                if score == best["score"]:
                    train_f1 = _train_f1(model, train_dataset, device)
                    if train_f1 > best["train_f1"]:
                        best.update(
                            train_f1=train_f1, threshold=validation["best_threshold"],
                            auprc=validation["auprc"],
                            state={k: v.cpu().clone() for k, v in model.state_dict().items()},
                            epoch=epoch,
                        )
                        log.debug("Epoch %d ties on val %s; kept for better train F1 %.4f",
                                  epoch, params.select_metric.upper(), train_f1)
                stalled += 1
        elif average_loss < best["loss"]:
            best.update(
                loss=average_loss,
                state={k: v.cpu().clone() for k, v in model.state_dict().items()},
                epoch=epoch,
            )
            stalled = 0
        else:
            stalled += 1

        if stalled >= patience:
            log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        log.info("Restored best weights from epoch %d", best["epoch"])

    return {
        "epochs_ran": epochs_ran,
        "best_loss": best["loss"],
        "best_val_score": best["score"],
        "best_train_f1": best["train_f1"],
        "best_val_thr": best["threshold"],
        "best_val_auprc": best["auprc"],
        "optimizer_state_dict": optimizer.state_dict(),
    }


def stratified_folds(dataset, num_folds, seed):
    """Split graph indices into folds balanced by whether a graph has positives."""
    rng = random.Random(seed)
    with_positives = [i for i, g in enumerate(dataset) if g.y.sum().item() > 0]
    without = [i for i, g in enumerate(dataset) if g.y.sum().item() == 0]
    rng.shuffle(with_positives)
    rng.shuffle(without)

    folds = [[] for _ in range(num_folds)]
    for position, index in enumerate(with_positives):
        folds[position % num_folds].append(index)
    for position, index in enumerate(without):
        folds[position % num_folds].append(index)
    return folds


def stratified_split(dataset, val_split, seed):
    """Hold out a validation set that keeps a comparable positive density."""
    rng = random.Random(seed)
    with_positives = [i for i, g in enumerate(dataset) if g.y.sum().item() > 0]
    without = [i for i, g in enumerate(dataset) if g.y.sum().item() == 0]
    rng.shuffle(with_positives)
    rng.shuffle(without)

    total = max(1, int(len(dataset) * val_split))
    num_positive = max(1, round(len(with_positives) * val_split)) if with_positives else 0
    num_negative = min(max(0, total - num_positive), len(without))

    validation = set(with_positives[:num_positive]) | set(without[:num_negative])
    train = [i for i in range(len(dataset)) if i not in validation]
    return train, sorted(validation)


def _clone(dataset):
    """Independent copies, because normalisation rewrites edge_attr in place."""
    return [copy.deepcopy(graph) for graph in dataset]


def _build_model(dataset, model_params, use_known_fusions):
    positives = sum(int(g.y.sum().item()) for g in dataset)
    total = sum(int(g.y.numel()) for g in dataset)
    prior = positives / max(total, 1)
    config = {
        "num_node_features": dataset[0].x.shape[1],
        "num_edge_features": dataset[0].edge_attr.shape[1],
        **model_params.to_dict(),
        "prior_pos_rate": prior,
    }
    return FusionPredictor(**config), config


def _drop_unlabelled(dataset, enabled):
    """Optionally drop graphs with no positive edge."""
    if not enabled:
        return dataset
    kept = [graph for graph in dataset if graph.y.sum().item() > 0]
    if not kept:
        raise InputError(
            "Every sample was dropped because none has a labelled fusion edge. "
            "Pass --keep-empty-graphs, or check that the manifest fusions match the annotation."
        )
    if len(kept) < len(dataset):
        log.info("Dropped %d graph(s) without positive labels (%d remain)",
                 len(dataset) - len(kept), len(kept))
    return kept


def cross_validate(dataset, model_params, params, use_known_fusions, device):
    """K-fold cross-validation; returns pooled validation metrics."""
    folds = stratified_folds(dataset, params.cv_folds, params.seed)
    log.info("%d-fold cross-validation | fold sizes: %s", params.cv_folds, [len(f) for f in folds])

    continuous_cols = continuous_edge_columns(use_known_fusions)
    pooled_logits, pooled_labels, fold_summaries = [], [], []

    for index, fold in enumerate(folds, start=1):
        log.info("--- Fold %d/%d ---", index, params.cv_folds)
        validation_ids = set(fold)
        train_graphs = _clone([g for i, g in enumerate(dataset) if i not in validation_ids])
        val_graphs = _clone([dataset[i] for i in sorted(validation_ids)])
        train_graphs = _drop_unlabelled(train_graphs, params.drop_empty_graphs)

        stats = compute_edge_norm_stats(train_graphs, use_known_fusions)
        apply_edge_norm(train_graphs, stats)
        apply_edge_norm(val_graphs, stats)

        model, _ = _build_model(train_graphs, model_params, use_known_fusions)
        fold_summaries.append(
            fit(model, train_graphs, val_graphs, params, device, continuous_cols)
        )

        logits, labels = collect_predictions(model, val_graphs, device)
        pooled_logits.append(logits)
        pooled_labels.append(labels)

    probabilities = torch.sigmoid(torch.cat(pooled_logits))
    labels = torch.cat(pooled_labels)
    f1, threshold, precision, recall, auprc = best_threshold(probabilities, labels)
    mean_fold_auprc = sum(s["best_val_auprc"] for s in fold_summaries) / len(fold_summaries)

    log.info(
        "Cross-validation | pooled AUPRC %.4f | mean per-fold AUPRC %.4f | "
        "best F1 %.4f at threshold %.4f (P=%.2f R=%.2f) | %d/%d positive validation edges",
        auprc, mean_fold_auprc, f1, threshold, precision, recall,
        int(labels.sum().item()), labels.numel(),
    )
    return {
        "best_val_thr": threshold,
        "best_val_auprc": auprc,
        "best_val_f1": f1,
        "cv_mean_auprc": mean_fold_auprc,
        "cv_folds": params.cv_folds,
    }

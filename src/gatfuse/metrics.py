"""Ranking metrics for extreme class imbalance."""

import torch


def f1_at_threshold(probabilities, targets, threshold):
    """F1, precision and recall at a fixed threshold."""
    predictions = (probabilities >= threshold).float()
    true_positives = float((predictions * targets).sum().item())
    false_positives = float((predictions * (1.0 - targets)).sum().item())
    false_negatives = float(((1.0 - predictions) * targets).sum().item())
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8), precision, recall


def best_threshold(probabilities, targets):
    """Threshold maximising F1, plus AUPRC.

    Returns:
        (best_f1, threshold, precision, recall, auprc)
    """
    if targets.numel() == 0 or targets.sum().item() == 0:
        return 0.0, 0.5, 0.0, 0.0, 0.0

    order = torch.argsort(probabilities, descending=True)
    sorted_probabilities = probabilities[order]
    sorted_targets = targets[order]

    true_positives = torch.cumsum(sorted_targets, dim=0)
    false_positives = torch.cumsum(1.0 - sorted_targets, dim=0)
    total_positives = float(sorted_targets.sum().item())

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (total_positives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    best = int(torch.argmax(f1).item())
    auprc = float(
        torch.trapezoid(
            torch.cat([precision[:1], precision]), torch.cat([torch.zeros(1), recall])
        ).item()
    )
    return (
        float(f1[best].item()),
        float(sorted_probabilities[best].item()),
        float(precision[best].item()),
        float(recall[best].item()),
        auprc,
    )


def summarise(probabilities, targets):
    """Metric bundle used for logging and model selection."""
    f1_05, precision_05, recall_05 = f1_at_threshold(probabilities, targets, 0.5)
    f1_best, threshold, precision, recall, auprc = best_threshold(probabilities, targets)
    return {
        "f1_at_05": f1_05,
        "precision_at_05": precision_05,
        "recall_at_05": recall_05,
        "f1_best": f1_best,
        "best_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "auprc": auprc,
        "num_positive": int(targets.sum().item()),
        "num_total": int(targets.numel()),
    }

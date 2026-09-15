"""The GATFuse network: a GATv2 edge classifier over the gene/junction graph."""

import math

import torch
import torch.nn.functional as F
from torch.nn import ELU, Dropout, LayerNorm, Linear, ModuleList, Parameter, Sequential
from torch_geometric.nn import GATv2Conv

from .logging_utils import get_logger

log = get_logger("model")


class FocalLoss(torch.nn.Module):
    """Class-balanced focal loss (Lin et al., 2017)."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class FusionPredictor(torch.nn.Module):
    """Scores every edge of a sample graph as fusion or artefact.

    Args:
        num_node_features: width of ``graph.x``.
        num_edge_features: width of ``graph.edge_attr``.
        hidden_dim: node embedding width.
        dropout: dropout probability shared by the encoders, the GAT layers
            and the classifier.
        heads: GATv2 attention heads per layer.
        edge_embed_dim: width of the learned junction embedding.
        prior_pos_rate: positive-edge frequency of the training set; used to
            initialise the classifier bias to the correct base rate.
        num_gnn_layers: depth of the message-passing trunk.
        node_skip: feed pre-message-passing node embeddings to the classifier.

    Forward pass returns raw logits of shape (E,); apply a sigmoid for
    probabilities, or ``calibrate`` first when Platt parameters are available.
    """

    def __init__(self, num_node_features, num_edge_features, hidden_dim=64,
                 dropout=0.6, heads=2, edge_embed_dim=32, prior_pos_rate=None,
                 num_gnn_layers=2, node_skip=True):
        super().__init__()
        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be >= 1")

        self.prior_pos_rate = prior_pos_rate
        self.num_gnn_layers = num_gnn_layers
        self.node_skip = node_skip
        self.hidden_dim = hidden_dim

        self.edge_encoder = Sequential(
            Linear(num_edge_features, edge_embed_dim),
            LayerNorm(edge_embed_dim),
            ELU(),
            Dropout(p=dropout),
            Linear(edge_embed_dim, edge_embed_dim),
            LayerNorm(edge_embed_dim),
            ELU(),
        )

        self.node_encoder = Sequential(
            Linear(num_node_features, hidden_dim),
            LayerNorm(hidden_dim),
            ELU(),
        )

        self.convs = ModuleList()
        self.norms = ModuleList()
        in_dim = hidden_dim
        for layer in range(num_gnn_layers):
            is_last = layer == num_gnn_layers - 1
            self.convs.append(
                GATv2Conv(
                    in_dim, hidden_dim, heads=heads, concat=not is_last,
                    dropout=dropout, edge_dim=edge_embed_dim,
                )
            )
            out_dim = hidden_dim if is_last else hidden_dim * heads
            self.norms.append(LayerNorm(out_dim))
            in_dim = out_dim

        first_out = hidden_dim * heads if num_gnn_layers > 1 else hidden_dim
        self.res_first = Linear(hidden_dim, first_out, bias=False)
        self.dropout = Dropout(p=dropout)

        classifier_in = hidden_dim * 2 + edge_embed_dim
        if node_skip:
            classifier_in += hidden_dim * 2
        self.edge_mlp = Sequential(
            Linear(classifier_in, hidden_dim),
            LayerNorm(hidden_dim),
            ELU(),
            Dropout(p=dropout),
            Linear(hidden_dim, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            ELU(),
            Dropout(p=dropout),
            Linear(hidden_dim // 2, 1),
        )

        if prior_pos_rate is not None and 0.0 < prior_pos_rate < 1.0:
            self.edge_mlp[-1].bias.data.fill_(math.log(prior_pos_rate / (1.0 - prior_pos_rate)))

        # Evidence gate: softplus-constrained path from split-read support to the
        # logit. Initialised so that softplus(g) is about 0.5.
        self.evidence_gate = Parameter(torch.tensor(-0.541, dtype=torch.float32))

    def forward(self, x, edge_index, edge_attr, edge_evidence=None):
        edge_embed = self.edge_encoder(edge_attr)
        node_embed = self.node_encoder(x)

        hidden = node_embed
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            out = F.elu(norm(conv(hidden, edge_index, edge_embed)))
            if layer == 0:
                out = out + self.res_first(node_embed)
            elif out.shape == hidden.shape:
                out = out + hidden
            hidden = self.dropout(out)

        source = hidden[edge_index[0]]
        target = hidden[edge_index[1]]
        if self.node_skip:
            parts = [source, target, node_embed[edge_index[0]], node_embed[edge_index[1]], edge_embed]
        else:
            parts = [source, target, edge_embed]
        logits = self.edge_mlp(torch.cat(parts, dim=1)).squeeze(-1)

        if edge_evidence is not None and edge_evidence.numel() > 0:
            evidence = edge_evidence.view(-1).to(logits.dtype)
            logits = logits + F.softplus(self.evidence_gate) * evidence
        return logits


def calibrate(logits, platt_params):
    """Apply Platt scaling so scores read as probabilities, not just ranks."""
    if not platt_params:
        return logits
    return platt_params["A"] * logits + platt_params["B"]


@torch.no_grad()
def fit_platt_scaling(model, dataset, device=None):
    """Fit sigmoid(A * logit + B) on *dataset*, returning {'A': ..., 'B': ...}."""
    device = device or next(model.parameters()).device
    logits, targets = collect_predictions(model, dataset, device)

    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], lr=0.1, max_iter=100)
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(a * logits + b, targets)
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)

    params = {"A": float(a.item()), "B": float(b.item())}
    log.info("Platt scaling fitted: A=%.4f B=%.4f", params["A"], params["B"])
    return params


@torch.no_grad()
def collect_predictions(model, dataset, device=None):
    """Concatenated logits and labels over a list of graphs."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    all_logits, all_targets = [], []
    for graph in dataset:
        graph = graph.to(device)
        all_logits.append(
            model.forward(
                graph.x, graph.edge_index, graph.edge_attr,
                edge_evidence=getattr(graph, "edge_evidence", None),
            ).cpu()
        )
        all_targets.append(graph.y.cpu())
    if was_training:
        model.train()
    return torch.cat(all_logits), torch.cat(all_targets).float()

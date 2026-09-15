"""Default parameters for graph construction, model architecture and training."""

from dataclasses import asdict, dataclass, field, fields

# Identifier for the node/edge feature layout.
FEATURE_VERSION = "v8"

# Version of the non-feature payload stored alongside cached graphs.
GRAPH_PAYLOAD_VERSION = "2"

LIBRARY_TYPES = ("unstranded", "stranded_forward", "stranded_reverse")

# GTF attributes searched, in order, for the gene biotype.
#   "ensembl": Ensembl/`gene_biotype` only.
#   "auto":    also accept GENCODE's `gene_type`.
BIOTYPE_ATTRIBUTE_MODES = {
    "ensembl": ("gene_biotype",),
    "auto": ("gene_biotype", "gene_type"),
}


@dataclass(frozen=True)
class GraphParams:
    """Parameters that determine the content of a sample graph."""

    min_split_reads: int = 2
    library_type: str = "unstranded"
    biotype_attributes: str = "ensembl"
    use_known_fusions: bool = True
    # Weight the known-pair feature by the catalogued case count.
    known_fusion_recurrence: bool = False

    # Window around each breakpoint within which a discordant pair counts as
    # supporting evidence.
    discordant_window: int = 10_000
    # Maximum gene-body distance for the read-through edge feature.
    readthrough_max_distance: int = 100_000

    def validate(self):
        from .errors import UsageError

        if self.min_split_reads < 1:
            raise UsageError("--min-split-reads must be >= 1")
        if self.library_type not in LIBRARY_TYPES:
            raise UsageError(
                f"Unknown library type '{self.library_type}'. Choose from: {', '.join(LIBRARY_TYPES)}"
            )
        if self.biotype_attributes not in BIOTYPE_ATTRIBUTE_MODES:
            raise UsageError(
                f"Unknown biotype attribute mode '{self.biotype_attributes}'. "
                f"Choose from: {', '.join(BIOTYPE_ATTRIBUTE_MODES)}"
            )
        if self.discordant_window < 0:
            raise UsageError("--discordant-window must be >= 0")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Rebuild from checkpoint metadata, ignoring keys from other versions."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass(frozen=True)
class ModelParams:
    """GATv2 architecture hyperparameters."""

    hidden_dim: int = 64
    edge_embed_dim: int = 32
    dropout: float = 0.6
    heads: int = 2
    num_gnn_layers: int = 2
    node_skip: bool = True

    def validate(self):
        from .errors import UsageError

        if self.hidden_dim < 2 or self.hidden_dim % 2:
            raise UsageError("--hidden-dim must be an even integer >= 2")
        if self.edge_embed_dim < 1:
            raise UsageError("--edge-embed-dim must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise UsageError("--dropout must be in [0, 1)")
        if self.heads < 1:
            raise UsageError("--heads must be >= 1")
        if self.num_gnn_layers < 1:
            raise UsageError("--gnn-layers must be >= 1")
        return self

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainParams:
    """Optimisation and model-selection settings."""

    epochs: int = 1000
    learning_rate: float = 0.005
    batch_size: int = 8
    patience: int = 80
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    loss: str = "focal"            # "focal" or "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    # Only used with loss="bce": maximum negatives kept per positive.
    negative_ratio: int = 50
    # Ceiling on the BCE positive-class weight.
    max_pos_weight: float = 100.0

    feature_noise_std: float = 0.01
    select_metric: str = "auprc"   # "auprc" or "f1"
    cv_folds: int = 5
    val_split: float = 0.2
    drop_empty_graphs: bool = True
    seed: int = 42

    def validate(self):
        from .errors import UsageError

        if self.epochs < 1:
            raise UsageError("--epochs must be >= 1")
        if self.learning_rate <= 0:
            raise UsageError("--learning-rate must be > 0")
        if self.batch_size < 1:
            raise UsageError("--batch-size must be >= 1")
        if self.loss not in ("focal", "bce"):
            raise UsageError("--loss must be 'focal' or 'bce'")
        if self.select_metric not in ("auprc", "f1"):
            raise UsageError("--select-metric must be 'auprc' or 'f1'")
        if not 0.0 <= self.val_split < 1.0:
            raise UsageError("--val-split must be in [0, 1)")
        if self.cv_folds and self.cv_folds < 2:
            raise UsageError("--cv-folds must be 0 (disabled) or >= 2")
        if self.feature_noise_std < 0:
            raise UsageError("--feature-noise-std must be >= 0")
        return self

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PostprocessParams:
    """Rule-based filters applied to scored candidates."""

    drop_intragenic: bool = True
    drop_readthrough: bool = True
    drop_blacklisted: bool = True
    drop_noncanonical: bool = True
    noncanonical_min_canonical_fraction: float = 0.1
    readthrough_max_distance: int = 100_000
    annotate_only: bool = False
    # Extra gene symbols or regular expressions to treat as artefact families.
    extra_blacklist: tuple = field(default_factory=tuple)

    def validate(self):
        from .errors import UsageError

        if not 0.0 <= self.noncanonical_min_canonical_fraction <= 1.0:
            raise UsageError("--noncanonical-min-canonical must be in [0, 1]")
        if self.readthrough_max_distance < 0:
            raise UsageError("--readthrough-max-distance must be >= 0")
        return self

    def to_dict(self):
        return asdict(self)


# Tolerance, in base pairs, when matching a manifest breakpoint to a graph edge.
BREAKPOINT_LABEL_TOLERANCE = 100

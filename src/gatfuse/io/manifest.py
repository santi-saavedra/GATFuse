"""Training manifest: one RNA-seq sample per row."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import InputError
from ..logging_utils import get_logger

log = get_logger("io.manifest")

# Canonical column -> accepted aliases (case-insensitive).
_COLUMN_ALIASES = {
    "sample_id": ("sample", "name", "sample_name"),
    "bam": ("bam_file", "alignment", "alignments"),
    "chimeric_junctions": ("chimeric_file", "chimeric", "junctions"),
    "gene_counts": ("reads_per_gene_file", "reads_per_gene", "counts"),
    "annotation": ("gtf_file", "gtf"),
    "fusions": ("positive_fusions", "known_fusions", "positives"),
    "fusions_file": ("positive_fusions_file",),
}

REQUIRED_COLUMNS = ("bam", "chimeric_junctions", "gene_counts")


@dataclass
class ManifestRow:
    """One sample and the fusions confirmed in it."""

    sample_id: str
    bam: Path
    chimeric_junctions: Path
    gene_counts: Path
    annotation: Path = None
    fusions: list = field(default_factory=list)
    row_number: int = 0

    def input_paths(self):
        paths = [self.bam, self.chimeric_junctions, self.gene_counts]
        if self.annotation:
            paths.append(self.annotation)
        return paths


def _normalise_header(fieldnames):
    """Map the file's column names onto canonical names."""
    lookup = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        lookup[canonical] = canonical
        for alias in aliases:
            lookup[alias] = canonical

    mapping = {}
    for name in fieldnames or []:
        key = (name or "").strip().lower()
        if key in lookup:
            mapping[name] = lookup[key]
    return mapping


def parse_fusion(text):
    """Parse one confirmed fusion.

    Accepted forms:
        GENE1:GENE2                                  gene-level label
        GENE1:GENE2@chrA:posA-chrB:posB              breakpoint-level label

    Gene-level labels mark every edge between the pair as positive; breakpoint
    labels mark only edges whose coordinates match (see config.BREAKPOINT_LABEL_TOLERANCE).
    """
    raw = text.strip()
    if not raw:
        raise InputError("Empty fusion entry.")

    gene_part, _, breakpoint_part = raw.partition("@")

    separator = next((sep for sep in (":", ",", "\t", "-") if sep in gene_part), None)
    if separator is None:
        raise InputError(f"Cannot parse fusion '{raw}'. Expected DONOR:ACCEPTOR.")
    donor, acceptor = (part.strip() for part in gene_part.split(separator, 1))
    if not donor or not acceptor:
        raise InputError(f"Cannot parse fusion '{raw}'. Expected DONOR:ACCEPTOR.")

    if not breakpoint_part:
        return (donor, acceptor)

    try:
        left, right = breakpoint_part.split("-", 1)
        chr_a, pos_a = left.rsplit(":", 1)
        chr_b, pos_b = right.rsplit(":", 1)
        return (donor, acceptor, chr_a.strip(), int(pos_a), chr_b.strip(), int(pos_b))
    except (ValueError, TypeError):
        log.warning(
            "Could not parse breakpoints in '%s'; falling back to a gene-level label.", raw
        )
        return (donor, acceptor)


def _read_fusions_file(path):
    fusions = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    fusions.append(parse_fusion(stripped))
                except InputError as exc:
                    raise InputError(f"{exc} ({path}, line {line_number})") from exc
    except OSError as exc:
        raise InputError(f"Could not read fusion list {path}: {exc}") from exc
    return fusions


def load_manifest(path, default_annotation=None):
    """Parse a manifest into ManifestRow objects, validating every input path."""
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise InputError(f"Manifest not found: {manifest_path}")
    base = manifest_path.parent

    with open(manifest_path, encoding="utf-8") as handle:
        sample = handle.read(4096)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        delimiter = "\t"

    def resolve(value):
        if not value:
            return None
        candidate = Path(value.strip().strip('"')).expanduser()
        return candidate if candidate.is_absolute() else (base / candidate)

    rows = []
    with open(manifest_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        mapping = _normalise_header(reader.fieldnames)
        missing = [c for c in REQUIRED_COLUMNS if c not in mapping.values()]
        if missing:
            raise InputError(
                f"Manifest {manifest_path} is missing required column(s): {', '.join(missing)}.\n"
                f"Columns found: {reader.fieldnames}\n"
                "See docs/input-formats.md for the manifest specification."
            )

        for index, raw_row in enumerate(reader, start=1):
            row = {}
            for name, value in raw_row.items():
                canonical = mapping.get(name)
                if canonical:
                    row[canonical] = (value or "").strip().strip('"')
            if not any(row.values()):
                continue

            annotation = resolve(row.get("annotation")) or default_annotation
            if annotation is None:
                raise InputError(
                    f"Manifest row {index}: no annotation column and no --annotation given."
                )

            fusions = []
            if row.get("fusions"):
                for item in row["fusions"].split(";"):
                    if item.strip():
                        try:
                            fusions.append(parse_fusion(item))
                        except InputError as exc:
                            raise InputError(f"{exc} (manifest row {index})") from exc
            elif row.get("fusions_file"):
                fusions = _read_fusions_file(resolve(row["fusions_file"]))

            entry = ManifestRow(
                sample_id=row.get("sample_id") or f"sample_{index:03d}",
                bam=resolve(row.get("bam")),
                chimeric_junctions=resolve(row.get("chimeric_junctions")),
                gene_counts=resolve(row.get("gene_counts")),
                annotation=Path(annotation),
                fusions=fusions,
                row_number=index,
            )
            for label, file_path in (
                ("bam", entry.bam),
                ("chimeric_junctions", entry.chimeric_junctions),
                ("gene_counts", entry.gene_counts),
                ("annotation", entry.annotation),
            ):
                if file_path is None or not file_path.is_file():
                    raise InputError(
                        f"Manifest row {index} ({entry.sample_id}): {label} not found: {file_path}"
                    )
            rows.append(entry)

    if not rows:
        raise InputError(f"Manifest contains no sample rows: {manifest_path}")

    labelled = sum(1 for row in rows if row.fusions)
    log.info("Manifest: %d samples (%d with confirmed fusions)", len(rows), labelled)
    if labelled == 0:
        raise InputError(
            "No sample in the manifest lists a confirmed fusion. Training needs positive "
            "labels in the 'fusions' column or via 'fusions_file'."
        )
    return rows

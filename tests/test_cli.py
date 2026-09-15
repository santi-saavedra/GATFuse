"""Command-line contract: exit codes, defaults and end-to-end runs."""

import pandas as pd
import pytest

from gatfuse.cli import build_parser, main
from gatfuse.errors import EXIT_INPUT, EXIT_MODEL, EXIT_SUCCESS, EXIT_USAGE


def test_help_and_version_exit_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])
    assert exit_info.value.code == EXIT_SUCCESS


def test_no_command_prints_help(capsys):
    assert main([]) == EXIT_SUCCESS
    assert "detect" in capsys.readouterr().out


def test_missing_required_argument_is_a_usage_error():
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["detect", "-b", "a.bam"])
    assert exit_info.value.code == EXIT_USAGE


def test_graph_options_default_to_the_model_on_detect():
    """Unset options must stay None so the checkpoint's values win."""
    args = build_parser().parse_args(
        ["detect", "-b", "a", "-c", "b", "-e", "c", "-g", "d"]
    )
    assert args.min_split_reads is None
    assert args.library_type is None
    assert args.biotype_attributes is None


def test_train_uses_explicit_defaults():
    args = build_parser().parse_args(["train", "-m", "manifest.tsv"])
    assert args.min_split_reads == 2
    assert args.biotype_attributes == "auto"
    assert args.cv_folds == 5
    assert args.drop_empty_graphs is True


def test_missing_input_file_returns_the_input_exit_code(tmp_path):
    code = main([
        "detect", "-b", str(tmp_path / "nope.bam"), "-c", str(tmp_path / "nope.junction"),
        "-e", str(tmp_path / "nope.tab"), "-g", str(tmp_path / "nope.gtf"),
        "-o", str(tmp_path / "out.tsv"), "--device", "cpu", "--log-level", "error",
    ])
    assert code == EXIT_INPUT


def test_unreadable_model_returns_the_model_exit_code(tmp_path, bam_path, chimeric_path,
                                                     counts_path, annotation_path):
    broken = tmp_path / "broken.pt"
    broken.write_text("not a checkpoint")
    code = main([
        "detect", "-b", str(bam_path), "-c", str(chimeric_path), "-e", str(counts_path),
        "-g", str(annotation_path), "-m", str(broken), "-o", str(tmp_path / "out.tsv"),
        "--device", "cpu", "--log-level", "error",
    ])
    assert code == EXIT_MODEL


def test_detect_run_writes_both_tables(tmp_path, bam_path, chimeric_path, counts_path,
                                       annotation_path):
    output = tmp_path / "fusions.tsv"
    discarded = tmp_path / "discarded.tsv"
    code = main([
        "detect", "-b", str(bam_path), "-c", str(chimeric_path), "-e", str(counts_path),
        "-g", str(annotation_path), "-o", str(output), "-d", str(discarded),
        "--top-k", "5", "--device", "cpu", "--log-level", "error",
    ])
    assert code == EXIT_SUCCESS

    reported = pd.read_csv(output, sep="\t")
    assert {"donor_gene", "acceptor_gene", "score", "discordant_pairs"} <= set(reported.columns)
    assert ("BCR", "ABL1") in set(zip(reported["donor_gene"], reported["acceptor_gene"], strict=True))
    assert "filter_reason" in pd.read_csv(discarded, sep="\t").columns


def test_no_fusion_found_is_still_a_successful_run(tmp_path, bam_path, chimeric_path,
                                                   counts_path, annotation_path):
    """An empty result is a biological outcome, not a pipeline failure."""
    output = tmp_path / "none.tsv"
    code = main([
        "detect", "-b", str(bam_path), "-c", str(chimeric_path), "-e", str(counts_path),
        "-g", str(annotation_path), "-o", str(output), "--threshold", "1.0",
        "--device", "cpu", "--log-level", "error",
    ])
    assert code == EXIT_SUCCESS
    assert pd.read_csv(output, sep="\t").empty


def test_train_then_detect_round_trip(tmp_path, bam_path, chimeric_path, counts_path,
                                      annotation_path):
    manifest = tmp_path / "cohort.tsv"
    rows = ["sample_id\tbam\tchimeric_junctions\tgene_counts\tfusions"]
    for index in range(3):
        rows.append(
            f"S{index}\t{bam_path}\t{chimeric_path}\t{counts_path}\tBCR:ABL1;EWSR1:FLI1"
        )
    manifest.write_text("\n".join(rows) + "\n")

    model = tmp_path / "model.pt"
    assert main([
        "train", "-m", str(manifest), "-g", str(annotation_path), "-o", str(model),
        "--cv-folds", "0", "--val-split", "0.34", "--epochs", "3", "--patience", "2",
        "--batch-size", "1", "--device", "cpu", "--log-level", "error",
    ]) == EXIT_SUCCESS
    assert model.exists()

    output = tmp_path / "round-trip.tsv"
    assert main([
        "detect", "-m", str(model), "-b", str(bam_path), "-c", str(chimeric_path),
        "-e", str(counts_path), "-g", str(annotation_path), "-o", str(output),
        "--top-k", "3", "--device", "cpu", "--log-level", "error",
    ]) == EXIT_SUCCESS
    assert not pd.read_csv(output, sep="\t").empty


def test_trained_checkpoint_records_its_parameters(tmp_path, bam_path, chimeric_path,
                                                   counts_path, annotation_path):
    from gatfuse import checkpoint as checkpoint_io

    manifest = tmp_path / "cohort.tsv"
    manifest.write_text(
        "sample_id\tbam\tchimeric_junctions\tgene_counts\tfusions\n"
        + "\n".join(
            f"S{i}\t{bam_path}\t{chimeric_path}\t{counts_path}\tBCR:ABL1" for i in range(2)
        )
        + "\n"
    )
    model = tmp_path / "recorded.pt"
    main([
        "train", "-m", str(manifest), "-g", str(annotation_path), "-o", str(model),
        "--cv-folds", "0", "--val-split", "0.5", "--epochs", "2", "--patience", "1",
        "--batch-size", "1", "--min-split-reads", "3", "--device", "cpu",
        "--log-level", "error",
    ])
    checkpoint = checkpoint_io.load(model)
    assert checkpoint.graph_params.min_split_reads == 3
    assert checkpoint.graph_params.biotype_attributes == "auto"
    assert checkpoint.graph_params.known_fusion_recurrence is True
    assert checkpoint.metadata["feature_version"] == "v8"


def test_resume_restores_weights_and_optimiser_state(tmp_path, bam_path, chimeric_path,
                                                     counts_path, annotation_path):
    """--resume-from must continue an optimiser, not silently restart one."""
    from gatfuse import checkpoint as checkpoint_io

    manifest = tmp_path / "cohort.tsv"
    manifest.write_text(
        "sample_id\tbam\tchimeric_junctions\tgene_counts\tfusions\n"
        + "\n".join(
            f"S{i}\t{bam_path}\t{chimeric_path}\t{counts_path}\tBCR:ABL1" for i in range(2)
        )
        + "\n"
    )
    common = [
        "train", "-m", str(manifest), "-g", str(annotation_path),
        "--cv-folds", "0", "--val-split", "0.5", "--epochs", "2", "--patience", "1",
        "--batch-size", "1", "--device", "cpu", "--log-level", "error",
    ]
    first = tmp_path / "first.pt"
    assert main([*common, "-o", str(first)]) == EXIT_SUCCESS
    assert checkpoint_io.load(first).optimizer_state is not None

    second = tmp_path / "second.pt"
    assert main([*common, "-o", str(second), "--resume-from", str(first)]) == EXIT_SUCCESS
    assert checkpoint_io.load(second).optimizer_state is not None

"""Error types and the exit codes they map to."""

EXIT_SUCCESS = 0
EXIT_ERROR = 1        # unexpected internal failure
EXIT_USAGE = 2        # invalid command line (also used by argparse)
EXIT_INPUT = 3        # missing or malformed input file / incompatible data
EXIT_MODEL = 4        # checkpoint missing, unreadable or incompatible


class GATFuseError(Exception):
    """Base class for errors that are reported to the user without a traceback."""

    exit_code = EXIT_ERROR


class UsageError(GATFuseError):
    """Invalid combination of command-line options."""

    exit_code = EXIT_USAGE


class InputError(GATFuseError):
    """An input file is missing, unreadable, or does not have the expected content."""

    exit_code = EXIT_INPUT


class ModelError(GATFuseError):
    """The checkpoint cannot be loaded or does not match the data it is applied to."""

    exit_code = EXIT_MODEL

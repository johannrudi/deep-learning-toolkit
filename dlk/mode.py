import enum


class Mode(enum.Flag):
    """Define bit flags used to select one or more execution modes.

    Examples:
        if Mode.TRAIN in mode:
            run_training()

        if Mode.PROFILE in mode:
            run_profiling()

        if not mode.any(Mode.PREDICT | Mode.EVAL):
            exit()

        if Mode.EVAL not in mode:
            exit()
    """

    TRAIN = enum.auto()
    PROFILE = enum.auto()
    VALIDATE = enum.auto()
    PREDICT = enum.auto()
    EVAL = enum.auto()

    def any(self, modes: "Mode") -> bool:
        """Check whether the current mode includes any requested mode.

        Args:
            modes: One or more modes combined with bitwise OR.

        Returns:
            bool: `True` when at least one flag in `modes` is set in `self`.
        """
        return bool(self & modes)


def get_mode_from_name(name: str) -> Mode | None:
    """Parse an underscore-delimited mode name into a combined `Mode`.

    Args:
        name: Mode name such as `"train"` or `"train_profile"`.

    Returns:
        Mode | None: Parsed mode combination. Returns `None` when `name` is
        empty.

    Raises:
        ValueError: If any token in `name` does not map to a known mode.
    """
    mode: Mode | None = None
    if not name:
        return mode
    try:
        for token in name.split("_"):
            parsed_mode = Mode[token.upper()]
            mode = parsed_mode if mode is None else mode | parsed_mode
    except KeyError as exc:
        raise ValueError(f"unknown mode name: {name!r}") from exc
    return mode

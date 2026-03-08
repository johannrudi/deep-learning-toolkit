import pytest

from dlk.mode import Mode, get_mode_from_name


def test_mode_any_returns_true_when_requested_flags_overlap() -> None:
    """Return True when at least one requested mode is present."""
    mode = Mode.TRAIN | Mode.PROFILE

    assert mode.any(Mode.PROFILE | Mode.EVAL)


def test_mode_any_returns_false_when_requested_flags_do_not_overlap() -> None:
    """Return False when none of the requested modes are present."""
    mode = Mode.TRAIN | Mode.PROFILE

    assert not mode.any(Mode.PREDICT | Mode.EVAL)


def test_mode_membership_operator_detects_included_and_excluded_flags() -> None:
    """Support inclusion checks using `in` with combined flag values."""
    mode = Mode.TRAIN | Mode.PROFILE | Mode.VALIDATE

    assert Mode.TRAIN in mode
    assert Mode.VALIDATE in mode
    assert Mode.PREDICT not in mode


def test_get_mode_from_name_returns_none_for_empty_name() -> None:
    """Return None when the provided mode string is empty."""
    assert get_mode_from_name("") is None


@pytest.mark.parametrize(
    ("name", "expected_mode"),
    [
        ("train", Mode.TRAIN),
        ("TRAIN", Mode.TRAIN),
        ("profile", Mode.PROFILE),
        ("train_profile_eval", Mode.TRAIN | Mode.PROFILE | Mode.EVAL),
    ],
)
def test_get_mode_from_name_parses_valid_mode_names(
    name: str,
    expected_mode: Mode,
) -> None:
    """Parse single and underscore-delimited mode names."""
    assert get_mode_from_name(name) == expected_mode


def test_get_mode_from_name_raises_for_unknown_mode_token() -> None:
    """Raise ValueError when any token does not match a known mode."""
    with pytest.raises(ValueError, match=r"unknown mode name: 'train_unknown'"):
        get_mode_from_name("train_unknown")

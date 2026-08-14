"""Shared validation utilities keep exact configuration contracts consistent across LatentSeq.

The public component factories deliberately avoid hidden fallback configuration.  This module
contains only small structural checks reused by those factories; it owns no component semantics.
"""

from collections.abc import Mapping


# main


def require_exact_keys(mapping: Mapping, expected: set[str], name: str) -> None:
    """Require a mapping to contain exactly the documented keys.

    Args:
        mapping: Mapping whose schema is being checked.
        expected: Complete allowed-and-required key set.
        name: Human-readable object name for failures.

    Returns:
        None. Raises `ValueError` on missing or unknown keys.
    """
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping")
    missing = expected - set(mapping)
    unknown = set(mapping) - expected
    if missing or unknown:
        raise ValueError(
            f"{name} keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def require_positive_int(value: object, name: str) -> int:
    """Return `value` as an int after enforcing a positive non-bool integer contract."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

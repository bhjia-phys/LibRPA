#!/usr/bin/env python3
"""Expose current-v6 contracts to the historical trace-closure observer."""

from __future__ import annotations

from compare_qsgw_component_traces_v4 import *  # noqa: F403
from compare_qsgw_component_traces_v4 import _matrix_groups as _matrix_groups
from compare_qsgw_component_traces_v4 import parse_contract as _parse_legacy
from cmp_qsgw_v6 import _parse_contract as _parse_current


def _declared_version(text: str, label: str) -> int:
    versions: list[int] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.strip().split()
        if len(fields) == 3 and fields[:2] == ["#", "qsgw_contract_version"]:
            try:
                versions.append(int(fields[2]))
            except ValueError as error:
                raise ValueError(
                    f"{label}:{line_number}: invalid QSGW contract version"
                ) from error
    if len(versions) != 1:
        raise ValueError(f"{label}: expected exactly one QSGW contract version")
    return versions[0]


def parse_contract(
    text: str,
    label: str,
    *,
    require_current: bool = False,
) -> dict[str, object]:
    """Parse v6 with the regression comparator and retain v4/v5 support."""
    version = _declared_version(text, label)
    if version != 6:
        return _parse_legacy(text, label, require_current=require_current)

    contract = _parse_current(text, label)
    contract["task"] = (
        "qsgw" if contract["band"] == "disabled_stage1" else "qsgw_band"
    )
    contract["qsgw_update_hartree"] = (
        "1" if contract["hartree"] == "delta_density" else "0"
    )
    return contract

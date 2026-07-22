from pathlib import Path


PATCH = Path(__file__).with_name("exact847_legacy_wr_diagnostic_v1.patch")


def test_patch_is_limited_to_upstream_epsilon_observer_source():
    text = PATCH.read_text(encoding="utf-8")
    assert text.count("diff --git ") == 1
    assert "diff --git a/src/core/epsilon.cpp b/src/core/epsilon.cpp" in text
    assert "src/core/gw.cpp" not in text
    assert "src/core/exx.cpp" not in text
    assert "driver/tasks/qsgw.cpp" not in text


def test_patch_restores_the_exact847_wr_route():
    text = PATCH.read_text(encoding="utf-8")
    assert "SymmetryIrreducibleWRPlan" in text
    assert "allocate_symmetry_irreducible_wr_storage" in text
    assert "restore_symmetry_abf_rspace_dense_blocks" in text
    assert "plan.local_irreducible_sector" in text
    assert "DIAGNOSTIC: GW symmetry accumulates irreducible-sector" in text


def test_patch_does_not_add_a_runtime_product_switch():
    text = PATCH.read_text(encoding="utf-8")
    assert "getenv" not in text
    assert "LIBRPA_" not in text
    assert "qsgw_" not in text.lower()

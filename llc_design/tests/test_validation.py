import json

import pytest

from llc_design.validation import (
    EvidenceStatus,
    ReferenceDataNotReleaseReady,
    assert_hardware_release_ready,
    run_builtin_baseline_validation,
    validate_bundled_data,
    validation_summary,
)


def test_bundled_reference_data_has_complete_provenance():
    report = validate_bundled_data()
    assert report.valid
    assert {item.dataset for item in report.datasets} == {
        "devices.json",
        "cores.json",
        "materials.json",
        "transformer_core_presets.json",
    }
    assert all(item.record_count > 0 for item in report.datasets)
    assert not report.hardware_release_ready


def test_reference_data_cannot_be_approved_for_hardware_release():
    with pytest.raises(ReferenceDataNotReleaseReady, match="hardware release blocked"):
        assert_hardware_release_ready()


def test_provenance_validation_rejects_unverified_release_override(tmp_path):
    source = validate_bundled_data().datasets
    assert source
    data_dir = tmp_path
    for dataset in source:
        payload = {
            "metadata": {
                "source": "test",
                "provenance": "test",
                "status": "UNKNOWN",
                "verified_for_hardware": True,
            }
        }
        collection = {
            "devices.json": "primary_mosfets",
            "cores.json": "cores",
            "materials.json": "materials",
            "transformer_core_presets.json": "presets",
        }[dataset.dataset]
        payload[collection] = [{"part_number": "test"}]
        if dataset.dataset == "devices.json":
            payload["sr_mosfets"] = []
        (data_dir / dataset.dataset).write_text(json.dumps(payload), encoding="utf-8")
    report = validate_bundled_data(data_dir)
    assert not report.valid
    assert any("requires status VERIFIED" in issue.message for issue in report.issues)


def test_builtin_baseline_separates_software_from_external_evidence():
    matrix = run_builtin_baseline_validation()
    assert matrix.software_regression_passed
    assert not matrix.hardware_verified
    assert matrix.status is EvidenceStatus.UNKNOWN
    unknown = {item.check_id for item in matrix.checks if item.status is EvidenceStatus.UNKNOWN}
    assert unknown == {
        "plecs_or_ltspice_correlation",
        "bench_measurement_correlation",
    }


def test_validation_summary_is_json_safe():
    payload = validation_summary()
    encoded = json.dumps(payload)
    assert encoded
    assert payload["data_provenance"]["hardware_release_ready"] is False
    assert payload["baseline_matrix"]["hardware_verified"] is False

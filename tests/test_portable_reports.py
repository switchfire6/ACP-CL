"""Public report copies preserve scientific values and original artifact identity."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_portable_reports.py"
SPEC = importlib.util.spec_from_file_location("portable_reports_exporter", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Different formatting/newlines make accidental original rewriting visible.
    path.write_bytes(json.dumps(document, ensure_ascii=False, indent=3).replace("\n", "\r\n").encode())


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    return root


def test_export_changes_only_declared_paths_preserves_values_order_and_byte_identity(workspace):
    root = workspace
    source = root / "reports" / "first.json"
    second = root / "reports" / "other.json"
    raw = root / "runs" / "study" / "seed42"
    document = {
        "source_directory": str(raw),
        "runs": [{"source_files": {"metrics/~": {"path": str(raw / "result.json")}},
                  "checkpoint_path": str(raw / "checkpoint.pt")}],
        "result_path": str(second),
        "metrics": [0.8, -0.0, 1e-200, 2**70, True, None, "日本語"],
        "description": f"Previously recorded at {raw}; this prose is preserved verbatim.",
        "url": "https://example.org/reference",
    }
    other = {"result_path": "runs/study/seed42/result.json", "ordering": [3, 2, 1]}
    write_json(source, document)
    write_json(second, other)
    original = {path: path.read_bytes() for path in (source, second)}
    manifest = exporter.export_reports(root, Path("runs/public_export"),
                                       ["reports/first.json", "reports/other.json"],
                                       proof=Path("reports/proof.json"))
    output = root / "runs" / "public_export"
    expected = copy.deepcopy(document)
    expected["source_directory"] = "runs/study/seed42"
    expected["runs"][0]["source_files"]["metrics/~"]["path"] = "runs/study/seed42/result.json"
    expected["runs"][0]["checkpoint_path"] = "runs/study/seed42/checkpoint.pt"
    expected["result_path"] = "reports/other.json"
    exported = json.loads((output / "reports/first.json").read_bytes())
    assert exported == expected
    assert list(exported) == list(document)
    assert exported["metrics"][1].hex() == document["metrics"][1].hex()
    assert json.loads((output / "reports/other.json").read_bytes()) == other
    assert manifest["report_count"] == 2
    assert manifest["converted_path_count"] == 4
    assert all(manifest["verification"].values())
    assert manifest["raw_artifacts"]["checkpoint_files_included"] == 0
    mappings = manifest["files"][0]["path_mappings"]
    assert mappings[1]["pointer"] == "/runs/0/source_files/metrics~1~0/path"
    assert all(row["reference_kind"] == "ignored_raw_run_reference" for row in mappings[:3])
    assert not any(row["included_in_export"] for row in mappings[:3])
    assert mappings[3]["reference_kind"] == "included_report_copy"
    assert mappings[3]["included_in_export"] is True
    assert mappings[3]["export_path"] == "reports/other.json"
    for path, payload in original.items():
        assert path.read_bytes() == payload
    for row in manifest["files"]:
        assert row["original_sha256"] == hashlib.sha256((root / row["source"]).read_bytes()).hexdigest()
        assert row["original_lf_sha256"] == hashlib.sha256(
            (root / row["source"]).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert row["export_sha256"] == hashlib.sha256((output / row["export_path"]).read_bytes()).hexdigest()
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()) == [
        "export_manifest.json", "reports/first.json", "reports/other.json",
    ]
    assert not list(output.rglob("*.pt"))
    public_manifest = (output / "export_manifest.json").read_text(encoding="utf-8")
    proof_payload = (root / "reports/proof.json").read_bytes()
    assert str(root) not in public_manifest
    assert json.dumps(str(root))[1:-1].encode() not in proof_payload
    proof = json.loads(proof_payload)
    assert proof["full_mapping_manifest_sha256"] == hashlib.sha256(public_manifest.encode()).hexdigest()
    assert proof["files"][0]["converted_path_count"] == 4
    assert "path_mappings" not in proof["files"][0]
    assert proof["export_directory"] == "runs/public_export"
    # An identical repeat neither rewrites originals nor changes exported hashes.
    timestamps = {path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
    assert exporter.export_reports(root, Path("runs/public_export"),
                                   ["reports/first.json", "reports/other.json"],
                                   proof=Path("reports/proof.json")) == manifest
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in timestamps.items())


def test_crlf_source_keeps_raw_hash_and_adds_git_style_lf_identity(workspace):
    source = workspace / "reports/audit.json"
    source.parent.mkdir()
    # An escaped carriage return inside a JSON string is content, and must not
    # be confused with a physical CRLF line ending when canonicalizing bytes.
    canonical_lf = b'{\n  "metric": 0.875,\n  "text": "preserve \\r and \\n escapes"\n}\n'
    original_crlf = canonical_lf.replace(b"\n", b"\r\n")
    source.write_bytes(original_crlf)
    manifest = exporter.export_reports(workspace, Path("runs/export"), ["reports/audit.json"])
    row = manifest["files"][0]
    assert row["original_sha256"] == hashlib.sha256(original_crlf).hexdigest()
    assert row["original_lf_sha256"] == hashlib.sha256(canonical_lf).hexdigest()
    assert row["original_sha256"] != row["original_lf_sha256"]
    assert "Git text checkout normalized to LF" in manifest["hash_conventions"]["original_lf_sha256"]
    assert source.read_bytes() == original_crlf
    exported = (workspace / "runs/export/reports/audit.json").read_bytes()
    assert b"\r\n" not in exported
    assert json.loads(exported) == json.loads(canonical_lf)


@pytest.mark.parametrize("kind", ["external", "prefix_sibling", "absolute_escape", "relative_escape",
                                 "drive_relative", "file_uri", "foreign_absolute", "relative_backslash"])
def test_external_or_ambiguous_paths_fail_before_export_creation(workspace, kind):
    paths = {
        "external": str(workspace.parent / "outside" / "result.json"),
        "prefix_sibling": str(workspace.parent / (workspace.name + "_other") / "result.json"),
        "absolute_escape": str(workspace / ".." / "outside.json"),
        "relative_escape": "../outside.json",
        "drive_relative": "Q:result.json",
        "file_uri": "file:///unprovided/root/result.json",
        "foreign_absolute": "/outside/result.json" if os.name == "nt" else r"Q:\outside\result.json",
        "relative_backslash": r"runs\result.json",
    }
    source = workspace / "reports" / "audit.json"
    write_json(source, {"path": paths[kind], "accuracy": 0.9})
    before = source.read_bytes()
    with pytest.raises(exporter.ExportError):
        exporter.export_reports(workspace, Path("runs/export"), ["reports/audit.json"])
    assert source.read_bytes() == before
    assert not (workspace / "runs/export").exists()


def test_undeclared_absolute_field_is_not_silently_rewritten_or_left_behind(workspace):
    source = workspace / "reports" / "audit.json"
    write_json(source, {"unreviewed_field": str(workspace / "runs/result.json")})
    with pytest.raises(exporter.ExportError, match="undeclared path field"):
        exporter.export_reports(workspace, Path("runs/export"), ["reports/audit.json"])
    assert not (workspace / "runs/export").exists()


def test_missing_references_are_portable_locators_not_claimed_included_files(workspace):
    source = workspace / "reports" / "audit.json"
    write_json(source, {"path": str(workspace / "docs/not_present.md"),
                        "source_directory": str(workspace)})
    manifest = exporter.export_reports(workspace, Path("runs/export"), ["reports/audit.json"])
    mappings = manifest["files"][0]["path_mappings"]
    assert mappings[0]["portable_value"] == "docs/not_present.md"
    assert mappings[1]["portable_value"] == "."
    assert all(row["reference_kind"] == "repository_reference_not_exported" for row in mappings)
    assert not any(row["included_in_export"] for row in mappings)


def test_source_escape_and_raw_run_source_are_rejected(workspace):
    outside = workspace.parent / "outside.json"
    write_json(outside, {"metric": 1})
    raw = workspace / "runs" / "result.json"
    write_json(raw, {"metric": 1})
    for source in ("../outside.json", "runs/result.json"):
        with pytest.raises(exporter.ExportError):
            exporter.export_reports(workspace, Path("runs/export"), [source])
    assert not (workspace / "runs/export").exists()


def test_original_destination_and_conflicting_existing_exports_are_never_overwritten(workspace):
    source = workspace / "reports/audit.json"
    write_json(source, {"path": str(workspace / "runs/study"), "metric": 0.7})
    before = source.read_bytes()
    with pytest.raises(exporter.ExportError, match="original report"):
        exporter.export_reports(workspace, workspace, ["reports/audit.json"])
    assert source.read_bytes() == before
    conflicting = workspace / "runs/export/reports/audit.json"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"unrelated existing artifact")
    with pytest.raises(exporter.ExportError, match="different data"):
        exporter.export_reports(workspace, Path("runs/export"), ["reports/audit.json"])
    assert conflicting.read_bytes() == b"unrelated existing artifact"
    assert source.read_bytes() == before
    assert not (workspace / "runs/export/export_manifest.json").exists()
    with pytest.raises(exporter.ExportError, match="original report"):
        exporter.export_reports(workspace, Path("runs/fresh"), ["reports/audit.json"],
                                proof=Path("reports/audit.json"))
    assert not (workspace / "runs/fresh").exists()


def test_symlink_reference_cannot_escape_checkout(workspace):
    outside = workspace.parent / "external_data"
    outside.mkdir()
    link = workspace / "linked_data"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit test directory symlinks")
    write_json(workspace / "reports/audit.json", {"path": str(link / "result.json")})
    with pytest.raises(exporter.ExportError, match="outside the workspace"):
        exporter.export_reports(workspace, Path("runs/export"), ["reports/audit.json"])
    assert not (workspace / "runs/export").exists()


def test_symlink_destination_cannot_escape_export_directory(workspace):
    outside = workspace.parent / "external_destination"
    outside.mkdir()
    destination = workspace / "runs/export/reports"
    destination.parent.mkdir(parents=True)
    try:
        destination.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit test directory symlinks")
    write_json(workspace / "reports/audit.json", {"metric": 0.8})
    with pytest.raises(exporter.ExportError, match="symbolic link"):
        exporter.export_reports(workspace, Path("runs/export"), ["reports/audit.json"])
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("payload", [b'{"x": 1, "x": 2}', b'{"metric": NaN}', b'{"metric": 1e1000}'])
def test_json_that_cannot_be_preserved_fails_without_writing(workspace, payload):
    source = workspace / "reports/audit.json"
    source.parent.mkdir()
    source.write_bytes(payload)
    with pytest.raises(exporter.ExportError):
        exporter.export_reports(workspace, Path("runs/export"), ["reports/audit.json"])
    assert source.read_bytes() == payload
    assert not (workspace / "runs/export").exists()


def test_roundtrip_verification_catches_changed_metrics_array_order_and_types(workspace):
    original = {"path": str(workspace / "runs/result.json"), "scores": [0.5, 0.8], "count": 1}
    converted, mappings, inverses = exporter.convert_document(original, workspace, set())
    for alteration in (
        {**converted, "scores": [0.5, 0.9]},
        {**converted, "scores": [0.8, 0.5]},
        {**converted, "count": True},
        {**converted, "path": "runs/different.json"},
    ):
        with pytest.raises(exporter.ExportError):
            exporter.verify_roundtrip(original, alteration, mappings, inverses)


def test_cli_uses_requested_destination_and_emits_compact_public_proof(workspace, capsys):
    write_json(workspace / "reports/audit.json", {"path": str(workspace / "runs/study")})
    assert exporter.main(["--workspace", str(workspace), "--output", "runs/chosen_export",
                          "--proof", "reports/proof.json", "reports/audit.json"]) == 0
    assert "converted 1 paths" in capsys.readouterr().out
    assert (workspace / "runs/chosen_export/reports/audit.json").is_file()
    assert (workspace / "reports/proof.json").is_file()

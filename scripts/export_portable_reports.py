"""Export archived report JSON with verified, repository-relative path locators.

Only the selected JSON reports are copied; references to ignored runs and their
checkpoints remain references, not included artifacts. Originals are never
rewritten. Known path fields are handled explicitly, and unexpected absolute
strings or paths outside this checkout fail before any export is written.

Run from this historical checkout (its absolute paths must resolve here)::

    python scripts/export_portable_reports.py --proof reports/public_export_manifest.json

The default destination is ignored ``runs/public_report_export``. The full
export_manifest.json contains JSON-Pointer mappings; the optional compact proof
contains their manifest hash and per-file hashes/counts without local usernames.
This is a presentation export, not an input directory for ``acp-cl analyze``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


DEFAULT_SOURCES = (
    "reports/v2/diagnostics/v2_gaussian/diagnostics.json",
    "reports/v2/diagnostics/v2_shapes_development/diagnostics.json",
    "reports/v2/diagnostics/v2_shapes_long/diagnostics.json",
    "reports/v2/diagnostics/v2_shapes_noise_only/diagnostics.json",
    "reports/v2/diagnostics/v2_shapes_stationary/diagnostics.json",
    "reports/v2/reset_damage/endpoint_audit.json",
    "reports/v2/reset_damage/instrumented_audit.json",
    "reports/v2/shortcuts_development/shortcut_audit.json",
    "reports/v2/shortcuts_long/shortcut_audit.json",
)
PATH_FIELDS = frozenset({
    "path", "source_directory", "suite", "original_run", "run_directory",
    "checkpoint_path", "result_path",
})
MANIFEST_NAME = "export_manifest.json"


class ExportError(ValueError):
    """An export cannot preserve or safely interpret the supplied report."""


@dataclass
class PreparedReport:
    source: Path
    relative: str
    original: bytes
    original_document: Any
    payload: bytes
    mappings: list[dict]
    original_paths: dict[str, str]


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(document: Any) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _load_json(payload: bytes) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExportError("duplicate JSON object key would lose information")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ExportError(f"non-standard JSON numeric constant: {value}")

    try:
        return json.loads(payload.decode("utf-8-sig"), object_pairs_hook=unique_object,
                          parse_constant=invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError("source must be valid UTF-8 JSON") from error


def _pointer(parent: str, key: str | int) -> str:
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _absolute_candidate(value: str) -> bool:
    # Drive-relative Windows strings and file URIs are also rejected instead of
    # guessing their base. Embedded prose and HTTP links are not path values.
    return value.startswith(("/", "\\", "file:")) or bool(re.match(r"^[A-Za-z]:", value))


def _within(path: Path, root: Path, *, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ExportError(f"{description} escapes or lies outside the workspace") from error
    return resolved


def _portable(value: str) -> bool:
    path = PurePosixPath(value)
    return (bool(value) and "\\" not in value and not _absolute_candidate(value)
            and ".." not in path.parts and path.as_posix() == value)


def _relative_locator(value: str, root: Path, pointer: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ExportError(f"unsupported or incomplete absolute path at {pointer}; no base is inferred")
    resolved = _within(path, root, description=f"path at {pointer}")
    portable = resolved.relative_to(root).as_posix()
    if not _portable(portable):
        raise ExportError(f"path at {pointer} has no unambiguous portable locator")
    return portable


def _reference(portable: str, included: set[str]) -> dict:
    if portable in included:
        return {"reference_kind": "included_report_copy", "included_in_export": True,
                "export_path": portable}
    if PurePosixPath(portable).parts[:1] == ("runs",):
        return {"reference_kind": "ignored_raw_run_reference", "included_in_export": False}
    return {"reference_kind": "repository_reference_not_exported", "included_in_export": False}


def convert_document(document: Any, root: Path, included: set[str]) -> tuple[Any, list[dict], dict]:
    """Return a copy and portable mapping records, retaining inverses in memory only."""
    mappings, originals = [], {}

    def visit(value, pointer="", field=None):
        if isinstance(value, dict):
            return {key: visit(child, _pointer(pointer, key), key) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, _pointer(pointer, index)) for index, child in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            raise ExportError(f"non-finite JSON value at {pointer}")
        if not isinstance(value, str):
            return value
        absolute = _absolute_candidate(value)
        if absolute and field not in PATH_FIELDS:
            raise ExportError(f"unexpected absolute string in an undeclared path field at {pointer}")
        if absolute:
            portable = _relative_locator(value, root, pointer)
            originals[pointer] = value
            mappings.append({"pointer": pointer, "portable_value": portable,
                             **_reference(portable, included)})
            return portable
        if field in PATH_FIELDS:
            if not _portable(value):
                raise ExportError(f"nonportable relative path at {pointer}")
            _within(root / value, root, description=f"relative path at {pointer}")
        return value

    return visit(document), mappings, originals


def verify_roundtrip(original: Any, exported: Any, mappings: list[dict], originals: dict) -> None:
    """Verify structure/order/types and every unchanged value after serialization.

    Each changed pointer has an exact inverse kept only for this verification.
    Neither the public mapping manifest nor compact proof includes that inverse.
    Float hex comparison also preserves the sign of zero.
    """
    changed = {row["pointer"]: row["portable_value"] for row in mappings}
    visited = set()

    def check(a, b, pointer=""):
        if pointer in changed:
            if type(a) is not str or b != changed[pointer] or a != originals[pointer] or not _portable(b):
                raise ExportError(f"path roundtrip failed at {pointer}")
            visited.add(pointer)
            return
        if type(a) is not type(b):
            raise ExportError(f"JSON value type changed at {pointer}")
        if isinstance(a, dict):
            if list(a) != list(b):
                raise ExportError(f"JSON object keys/order changed at {pointer}")
            for key in a:
                check(a[key], b[key], _pointer(pointer, key))
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ExportError(f"JSON array length changed at {pointer}")
            for index, (x, y) in enumerate(zip(a, b)):
                check(x, y, _pointer(pointer, index))
        elif isinstance(a, float):
            if a.hex() != b.hex():
                raise ExportError(f"JSON number changed at {pointer}")
        elif a != b:
            raise ExportError(f"nonpath JSON value changed at {pointer}")

    check(original, exported)
    if visited != set(changed) or visited != set(originals):
        raise ExportError("path mapping pointers do not exactly match the changed values")


def _destination(path: Path, root: Path, payload: bytes, protected: set[Path]) -> Path:
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ExportError("an export destination escapes through a symbolic link") from error
    if target in protected:
        raise ExportError("export must not replace an original report")
    if target.exists() and (not target.is_file() or target.read_bytes() != payload):
        raise ExportError("export destination already contains different data; choose a new directory")
    return target


def _write_new_or_identical(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ExportError("export destination changed after validation")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never overwrites a report or a concurrent writer.
    with path.open("xb") as handle:
        handle.write(payload)


def export_reports(workspace: Path, output: Path, sources=DEFAULT_SOURCES,
                   *, proof: Path | None = None) -> dict:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ExportError("workspace must be a directory")
    output = (root / output).resolve()
    selected = []
    for source in sources:
        path = _within(root / source, root, description="source report")
        if (path.suffix.lower() != ".json" or not path.is_file()
                or path.relative_to(root).parts[:1] != ("reports",)):
            raise ExportError("every selected source must be an existing JSON report under reports/")
        selected.append(path)
    if not selected or len(set(selected)) != len(selected):
        raise ExportError("select at least one source, without duplicate reports")
    included = {path.relative_to(root).as_posix() for path in selected}
    prepared = []
    for source in selected:
        original = source.read_bytes()
        document = _load_json(original)
        converted, mappings, originals = convert_document(document, root, included)
        payload = _json_bytes(converted)
        verify_roundtrip(document, _load_json(payload), mappings, originals)
        prepared.append(PreparedReport(source, source.relative_to(root).as_posix(), original,
                                       document, payload, mappings, originals))

    manifest = {
        "schema_version": 1,
        "purpose": "Portable presentation copies of archived report JSON; original files retained.",
        "locator_base": "repository_root",
        "export_path_base": "export_bundle_root",
        "hash_conventions": {
            "original_sha256": "SHA-256 of untouched source bytes from the exporter's working copy.",
            "original_lf_sha256": "SHA-256 after replacing CRLF byte pairs with LF; every other byte is retained. Use this to verify a Git text checkout normalized to LF.",
            "export_sha256": "SHA-256 of the exported UTF-8 JSON bytes, serialized with LF line endings.",
        },
        "raw_artifacts": {
            "policy": "Referenced runs and checkpoints are omitted, not copied or uploaded.",
            "raw_run_files_included": 0, "checkpoint_files_included": 0,
            "ignored_run_locator_prefix": "runs/",
            "reproduction": "Use the source revision/configuration recorded in each report; these are not download links.",
        },
        "report_count": len(prepared),
        "converted_path_count": sum(len(report.mappings) for report in prepared),
        "verification": {
            "original_bytes_untouched": True, "all_nonpath_values_unchanged": True,
            "object_and_array_order_preserved": True, "targeted_values_portable": True,
            "inverse_path_roundtrip_verified": True,
        },
        "files": [{
            "source": report.relative, "original_sha256": sha256(report.original),
            "original_lf_sha256": sha256(report.original.replace(b"\r\n", b"\n")),
            "export_path": report.relative, "export_sha256": sha256(report.payload),
            "converted_path_count": len(report.mappings), "path_mappings": report.mappings,
        } for report in prepared],
    }
    manifest_payload = _json_bytes(manifest)
    protected = set(selected)
    destinations = [_destination(output / report.relative, output, report.payload, protected)
                    for report in prepared]
    manifest_path = _destination(output / MANIFEST_NAME, output, manifest_payload, protected)
    if manifest_path in destinations:
        raise ExportError("selected report collides with the export manifest")
    proof_path, proof_payload = None, None
    if proof is not None:
        proof_path = (root / proof).resolve()
        try:
            export_locator = output.relative_to(root).as_posix()
        except ValueError:
            export_locator = None
        compact = {**manifest, "export_directory": export_locator,
                   "full_mapping_manifest": MANIFEST_NAME,
                   "full_mapping_manifest_sha256": sha256(manifest_payload),
                   "files": [{key: value for key, value in row.items() if key != "path_mappings"}
                             for row in manifest["files"]]}
        proof_payload = _json_bytes(compact)
        proof_path = _destination(proof_path, root, proof_payload, protected)
        if proof_path in destinations or proof_path == manifest_path:
            raise ExportError("proof destination collides with the export bundle")

    # All sources, paths, JSON values, and output collisions are validated before
    # creating any destination. Recheck the original bytes before and afterward.
    def check_originals():
        if any(report.source.read_bytes() != report.original for report in prepared):
            raise ExportError("an original report changed during export")

    check_originals()
    for report, destination in zip(prepared, destinations):
        _write_new_or_identical(destination, report.payload)
        verify_roundtrip(report.original_document, _load_json(destination.read_bytes()),
                         report.mappings, report.original_paths)
    check_originals()
    _write_new_or_identical(manifest_path, manifest_payload)
    if proof_path is not None:
        _write_new_or_identical(proof_path, proof_payload)
    check_originals()
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", help="Repository-relative report JSON paths; defaults to the nine inventoried v2 reports")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("runs/public_report_export"))
    parser.add_argument("--proof", type=Path, help="Optional compact proof JSON inside the workspace")
    args = parser.parse_args(argv)
    try:
        manifest = export_reports(args.workspace, args.output, args.sources or DEFAULT_SOURCES,
                                  proof=args.proof)
    except (ExportError, OSError) as error:
        parser.exit(2, f"Export failed: {error}\n")
    print(f"Exported {manifest['report_count']} report copies; converted "
          f"{manifest['converted_path_count']} paths. Original bytes and every other JSON value verified unchanged.")
    print("Raw run files and checkpoints included: 0. Full mappings: export_manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

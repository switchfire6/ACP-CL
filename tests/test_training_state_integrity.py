"""Regression cases for damaged files left by the observed hard restart."""

import importlib.util
from pathlib import Path
import zipfile

import pytest


path = Path(__file__).resolve().parents[1]/"scripts/check_training_state_files.py"
spec = importlib.util.spec_from_file_location("training_state_integrity", path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_nonempty_zeroed_json_and_checkpoint_are_rejected_without_mutation(tmp_path):
    paths = [tmp_path/"prefix.json", tmp_path/"prefix.pt"]
    for p in paths:
        p.write_bytes(b"\x00"*256)
    result = checker.inspect_files(tmp_path)
    assert not result["passed"] and len(result["files"]) == 2
    assert all(not r["valid"] for r in result["files"])
    assert all(p.read_bytes() == b"\x00"*256 for p in paths)


def test_zip_payload_corruption_is_detected_even_with_valid_directory(tmp_path):
    p = tmp_path/"checkpoint.pt"
    payload = b"known_unique_tensor_payload"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as z:
        z.writestr("tensor", payload)
    raw = bytearray(p.read_bytes())
    raw[raw.index(payload)] ^= 1
    p.write_bytes(raw)
    result = checker.inspect_files(tmp_path)
    assert not result["passed"] and "CRC" in result["files"][0]["error"]


def test_valid_files_pass_and_uncommitted_temporary_files_are_ignored(tmp_path):
    (tmp_path/"prefix.json").write_text('{"valid":true}')
    (tmp_path/"checkpoint.pt.tmp").write_bytes(b"unfinished write")
    with zipfile.ZipFile(tmp_path/"checkpoint.pt", "w") as z:
        z.writestr("saved_state", b"complete")
    result = checker.inspect_files(tmp_path)
    assert result["passed"] and len(result["files"]) == 2


def test_corrupted_compressed_stream_is_recorded_without_aborting(tmp_path):
    p = tmp_path/"training_source.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source.py", b"scientific source\n"*100)
    raw = bytearray(p.read_bytes())
    name_length = int.from_bytes(raw[26:28], "little")
    extra_length = int.from_bytes(raw[28:30], "little")
    # Set the first deflate block type to the reserved value, preserving the
    # ZIP directory so validation has to inspect the compressed payload.
    raw[30+name_length+extra_length] |= 0b110
    p.write_bytes(raw)
    result = checker.inspect_files(tmp_path)
    assert not result["passed"] and not result["files"][0]["valid"]
    assert "invalid block type" in result["files"][0]["error"]
    assert p.read_bytes() == bytes(raw)


def test_empty_directory_is_not_reported_as_success(tmp_path):
    with pytest.raises(ValueError, match="no stable"):
        checker.inspect_files(tmp_path)

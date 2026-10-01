"""Read-only restart preflight; run only after the training process has stopped.

Check stable JSON, source archives and checkpoint ZIP CRCs. Never remove files
or choose retries automatically. Full learner-state/probe audits remain separate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import zlib


def inspect_files(directory):
    directory = Path(directory)
    rows = []
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix in (".json", ".pt", ".zip"))
    for path in paths:
        raw = path.read_bytes()
        row = dict(path=path.relative_to(directory).as_posix(), bytes=len(raw),
                   sha256=hashlib.sha256(raw).hexdigest())
        try:
            if path.suffix == ".json":
                json.loads(raw)
            else:
                with zipfile.ZipFile(path) as archive:
                    member = archive.testzip()
                    if member is not None:
                        raise ValueError(f"CRC mismatch in {member}")
            row["valid"] = True
        except (ValueError, UnicodeError, zipfile.BadZipFile, EOFError, zlib.error) as exc:
            row.update(valid=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    if not rows:
        raise ValueError("no stable study artifacts found")
    return dict(passed=all(r["valid"] for r in rows), files=rows,
        scope="Read-only JSON/ZIP integrity preflight for stopped jobs; not a model or scientific audit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inspect_files(args.input)
    output = Path(args.output)
    if output.resolve().is_relative_to(Path(args.input).resolve()):
        raise ValueError("write preflight reports outside the inspected run")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"], checked=len(result["files"]),
        invalid=[r["path"] for r in result["files"] if not r["valid"]]), indent=2))
    raise SystemExit(0 if result["passed"] else 1)

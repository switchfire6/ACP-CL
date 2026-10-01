"""Package development evidence, analysis source, and hashes after reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from acp_cl.persistence.study import write_json


def archive(output, development):
    root = Path(__file__).resolve().parents[1]
    output, development = Path(output).resolve(), Path(development).resolve()
    destination = output / "development_v1"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(development.rglob("*")):
        if source.is_file() and (source.suffix == ".json" or source.name == "training_source.zip"):
            target = destination / source.relative_to(development)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    files = [*sorted((root / "scripts").glob("*conditional*.py")),
             *sorted((root / "tests").glob("test_conditional*.py")),
             *sorted((root / "configs").glob("conditional*.json")),
             root / "docs/conditional_relationships.md", root / "reports/conditional_results.md",
             root / "README.md", root / ".github/workflows/ci.yml"]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output / "analysis_source.zip", "w", zipfile.ZIP_DEFLATED) as handle:
        for source in files:
            handle.write(source, source.relative_to(root).as_posix())
    write_json(output / "analysis_source_manifest.json", hashes)
    artifacts = {p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"}
    write_json(output / "artifact_manifest.json", artifacts)
    print(json.dumps(dict(artifacts=len(artifacts), bytes=sum(a["bytes"] for a in artifacts.values())), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development", required=True)
    args = parser.parse_args()
    archive(args.output, args.development)

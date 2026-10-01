"""Preserve every declared comparator development attempt, including failures."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil

from summarize_contextual import adequacy_checks, aggregate_seeds, audit_records, metrics
from acp_cl.persistence.study import write_json


def summarize(inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    studies = []
    for directory in map(Path, inputs):
        manifest, _, records, artifacts = audit_records(directory, require_lock=False)
        config = manifest["config"]
        raw = {key: metrics(record) for key, record in records.items()}
        values, aggregate = aggregate_seeds(raw, config)
        adequacy = adequacy_checks(aggregate["return"]["recurrent"], aggregate["return"]["oracle"])
        study = dict(study=config["study"], identity=manifest["identity"],
            recipe="interactions" if config.get("interaction_features", False) else "plain",
            updates=config["updates_per_batch"], aggregate=aggregate, adequacy=adequacy,
            qualifies=all(adequacy.values()), prefixes=len(config["seeds"])*len(config["gaps"])*len(config["methods"]),
            branches=len(records), result_hashes=artifacts,
            seeds=[dict(seed=s, method=m, branch=b, **row) for (s, m, b), row in values.items()])
        studies.append(study)
        target = output/"development"/directory.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "completion.json", "training_source.zip"):
            shutil.copy2(directory/name, target/name)
        write_json(target/"config.json", config)
        write_json(target/"summary.json", study)
        # Preserve all outcomes, including branches not used for selecting settings.
        with gzip.GzipFile(filename=str(target/"records.json.gz"), mode="wb", mtime=0) as handle:
            handle.write(json.dumps([records[key] for key in sorted(records)],
                                    separators=(",", ":"), allow_nan=False).encode())
    result = dict(studies=studies, prefixes=sum(s["prefixes"] for s in studies),
                  branches=sum(s["branches"] for s in studies),
                  qualifying_studies=[s["study"] for s in studies if s["qualifies"]],
                  inference="Development qualification only; not a test of candidate superiority.")
    write_json(output/"development_summary.json", result)
    lines = ["# Recurrent comparator development", "",
        f"{result['prefixes']} prefix fits and {result['branches']} challenge branches. Two development seeds per setting.", "",
        "Thresholds: oracle and recurrent initial acquisition >=5 pp over no transfer;",
        "recurrent return >=5 pp over no transfer; correct-minus-opposite history benefit >=2 pp.", "",
        "| Recipe | Updates | Oracle acquisition (pp) | Recurrent acquisition (pp) | Return gain (pp) | History benefit (pp) | Qualifies |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for study in studies:
        r, o = (study["aggregate"]["return"][m] for m in ("recurrent", "oracle"))
        row = [o["initial_gain"], r["initial_gain"], r["before"]-r["before_no_transfer"], r["before_history_effect"]]
        lines.append(f"| {study['recipe']} | {study['updates']} | "+" | ".join(f"{100*v:+.3f}" for v in row)
                     +f" | {'yes' if study['qualifies'] else 'no'} |")
    lines += ["", "All seeds, branches, configurations and exact training sources are preserved in `development/`.",
        "These repeated development outcomes do not provide independent confirmatory evidence.",
        "Do not pool the six settings as twelve independent seeds or interpret a failed comparator as a failed candidate.", ""]
    (output/"development_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "studies"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)

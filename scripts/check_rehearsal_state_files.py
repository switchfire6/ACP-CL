"""Read-only integrity check for stopped rehearsal-state jobs, including NPZ."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from check_training_state_files import inspect_files


def inspect(directory):
    directory = Path(directory)
    result = inspect_files(directory)
    for path in sorted(directory.rglob('*.npz')):
        row = dict(path=path.relative_to(directory).as_posix(), bytes=path.stat().st_size,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError('NPZ CRC mismatch')
            with np.load(path, allow_pickle=False) as archive:
                if len(archive.files) != len(set(archive.files)):
                    raise ValueError('duplicate NPZ array names')
                for name in archive.files:
                    if not np.isfinite(archive[name]).all():
                        raise ValueError('nonfinite numerical array')
            row['valid'] = True
        except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
            row.update(valid=False, error=f'{type(exc).__name__}: {exc}')
        result['files'].append(row)
    result['files'].sort(key=lambda row:row['path'])
    result['passed'] = all(row['valid'] for row in result['files'])
    result['scope'] = 'Read-only stopped-job JSON/ZIP/checkpoint and finite NPZ integrity; not model reconstruction or scientific inference.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.resolve().is_relative_to(Path(args.input).resolve()):
        raise ValueError('integrity output must be outside the run')
    result = inspect(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'], checked=len(result['files']),
                         invalid=[r['path'] for r in result['files'] if not r['valid']]), indent=2))
    raise SystemExit(0 if result['passed'] else 1)

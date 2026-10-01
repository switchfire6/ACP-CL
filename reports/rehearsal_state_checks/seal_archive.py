"""Package a completed cohort; never changes scientific records or metrics."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def seal(cohort):
    folder = ROOT / 'reports' / 'rehearsal_state' / cohort
    for name in ('summary.json', 'audit.json', 'file_integrity.json'):
        if not (folder/name).is_file():
            raise ValueError(f'Missing completed evidence: {name}')
    for name in ('audit.json', 'file_integrity.json'):
        if json.loads((folder/name).read_text(encoding='utf-8'))['passed'] is not True:
            raise ValueError(f'Incomplete check: {name}')
    if (folder/'verifier_lock.json').exists():
        raise ValueError('Cohort already sealed; preserve it unchanged')
    names = ('scripts/verify_rehearsal_state_archive.py',
             'tests/test_rehearsal_state_archive.py')
    files = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    shutil.copyfile(ROOT/names[0], folder/'verify_archive.py')
    with zipfile.ZipFile(folder/'verifier_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT/name, name)
    lock = dict(sealed_utc=datetime.now(timezone.utc).isoformat(), files=files,
                scope='Independent packaging verifier and tests, excluded prospectively from the scientific lock.')
    (folder/'verifier_lock.json').write_text(json.dumps(lock, indent=2)+'\n', encoding='utf-8')
    artifacts = {path.relative_to(folder).as_posix():
                 dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in sorted(folder.rglob('*'))
                 if path.is_file() and path != folder/'artifact_manifest.json'}
    (folder/'artifact_manifest.json').write_text(json.dumps(artifacts, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(cohort=cohort, sealed=True, files=len(artifacts))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=('smoke', 'diagnostic'), required=True)
    seal(parser.parse_args().cohort)

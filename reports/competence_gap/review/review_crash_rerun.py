"""Reviewer D addendum: recompute the two seed-19102 batch-1 endpoint oracle tasks written around the
IndexError crash (exact at 05:36:33, gauge at 05:47:00) into a temp dir and compare bit-for-bit."""
import os
import sys
import json
import tempfile
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import competence_gap as CG

manifest = json.loads((ROOT / 'runs/competence_gap/manifest.json').read_text())
config = manifest['config']
out = {}
for name, gauge, channel in (('exact', False, CG.MC_ENDPOINT_EXACT), ('gauge', True, CG.MC_ENDPOINT_GAUGE)):
    path = Path(tempfile.gettempdir()) / f'cg_review_{name}_19102_01.npz'
    if path.exists():
        path.unlink()
    CG.endpoint_oracle_task(dict(config=config, seed=19102, gauge=gauge, channel=channel, batch=1, path=str(path)))
    with np.load(path) as a, np.load(ROOT / f'runs/competence_gap/oracle/endpoint/{name}_19102_01.npz') as b:
        out[name] = {k: bool(np.array_equal(a[k], b[k])) for k in ('query_counts', 'support_counts', 'kb', 'case_simulations')}
(ROOT / 'reports/competence_gap/review/crash_rerun.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out))

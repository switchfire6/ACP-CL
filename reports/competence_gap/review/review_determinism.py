"""Reviewer check D: re-run one locked fit from scratch in a temp dir and compare bit-for-bit."""
import os, sys, json, hashlib, tempfile, shutil, time
sys.dont_write_bytecode = True
for n in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(n, '1')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import competence_gap as CG

arm, seed = sys.argv[1], int(sys.argv[2])
run = ROOT / 'runs' / 'competence_gap'
manifest = json.loads((run / 'manifest.json').read_text())
config, identity = manifest['config'], manifest['identity']
tmp = Path(tempfile.gettempdir()) / f'cg_review_{arm}_{seed}'
if tmp.exists():
    shutil.rmtree(tmp)
t0 = time.perf_counter()
CG.run_fit(config, identity, seed, arm, str(tmp))
secs = time.perf_counter() - t0
new_dir, old_dir = CG.fit_dir(tmp, seed, arm), CG.fit_dir(run, seed, arm)
new, old = (json.loads((d / 'result.json').read_text()) for d in (new_dir, old_dir))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = dict(arm=arm, seed=seed, rerun_seconds=secs, tmp=str(tmp))
out['scores_probes_equal'] = [a['probes'] == b['probes'] for a, b in zip(new['scores'], old['scores'])]
out['scores_signature_equal'] = [a['signature'] == b['signature'] for a, b in zip(new['scores'], old['scores'])]
out['scores_model_sha_equal'] = [a['model_sha256'] == b['model_sha256'] for a, b in zip(new['scores'], old['scores'])]
for k in ('end_signature', 'final_model_sha256', 'initial_model_sha256', 'initial_signature', 'fingerprints', 'marginals', 'optimizer_steps'):
    out[k + '_equal'] = new[k] == old[k]
for name in ('evaluations.npz', 'evaluations.json', 'training.npz'):
    out[name + '_sha_equal'] = sha(new_dir / name) == sha(old_dir / name)
with np.load(new_dir / 'training.npz') as a, np.load(old_dir / 'training.npz') as b:
    out['training_arrays_equal'] = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
with np.load(new_dir / 'evaluations.npz') as a, np.load(old_dir / 'evaluations.npz') as b:
    out['evaluation_arrays_equal'] = bool(set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files))
d_new = {k: v for k, v in new['diagnostics'].items() if k != 'cost'}
d_old = {k: v for k, v in old['diagnostics'].items() if k != 'cost'}
out['diagnostics_equal_excl_cost'] = d_new == d_old
out['cost_equal_excl_seconds'] = ({k: v for k, v in new['diagnostics']['cost'].items() if 'seconds' not in k}
                                  == {k: v for k, v in old['diagnostics']['cost'].items() if 'seconds' not in k})
flat = [v for k, v in out.items() if k.endswith('_equal')]
flat = [all(v) if isinstance(v, list) else (all(v.values()) if isinstance(v, dict) else v) for v in flat]
out['all_identical'] = bool(all(flat) and out['training_arrays_equal'] and out['evaluation_arrays_equal'])
(ROOT / 'reports/competence_gap/review' / f'determinism_{arm}_{seed}.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

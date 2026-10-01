"""Reviewer E: independent recompute of online_excess.csv from training.npz and oracle online counts."""
import sys
import csv
import json
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
import numpy as np

RUN = ROOT / 'runs' / 'competence_gap'
SEEDS = list(range(19101, 19107))
ARMS = [f'U{u}_R{r}_W{w}' for u in (12, 3) for r in (16, 256) for w in (64, 256)] + ['exposure', 'all_action']
saved = {}
with open(ROOT / 'reports/competence_gap/online_excess.csv', newline='') as h:
    for row in csv.DictReader(h):
        saved[int(row['seed']), row['arm'], int(row['chunk'])] = float(row['excess'])
maxdiff, viol, rows, o_online = 0., 0, 0, {}
for s in SEEDS:
    with np.load(RUN / 'oracle' / f'seed_{s}.npz') as z:
        po = z['online_counts'] / int(z['online_counts'].max() and 1024)
        ot = z['online_truth'].astype(np.float64)
    ob = ((np.minimum.accumulate(po, axis=-1) - ot) ** 2).mean(axis=(1, 2))
    o_online[str(s)] = float(ob[:8192].mean())
    for arm in ARMS:
        with np.load(RUN / 'fits' / f'{arm}_{s}' / 'training.npz') as t:
            p, tr = t['probabilities'].astype(np.float64), t['truth'].astype(np.float64)
        assert np.array_equal(tr, ot[:len(tr)])
        viol += int((np.diff(p, axis=-1) > 0).sum())
        lb = ((p - tr) ** 2).mean(axis=(1, 2))
        ex = lb.reshape(-1, 1024).mean(1) - ob[:len(lb)].reshape(-1, 1024).mean(1)
        for i, v in enumerate(ex):
            maxdiff = max(maxdiff, abs(v - saved[s, arm, i]))
            rows += 1
out = dict(rows_compared=rows, csv_rows=len(saved), max_abs_diff=maxdiff, learner_online_monotonicity_violations=viol,
           O_online_brier_first_8192=o_online)
(ROOT / 'reports/competence_gap/review/review_online.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

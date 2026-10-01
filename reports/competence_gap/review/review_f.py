"""Reviewer F (opinion support, descriptive only): how close to O can a learner get from
N performed-action records if it is handed the TRUE observable features (source, lossy,
delayed, three pulse bits, exact reserves) instead of pixels? An idealised upper bound on
what representation learning could buy; not a protocol quantity.

Also decomposes the reference learner's excess into E[(p - p_O)^2] + cross term.
"""
import os
import sys
import json
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from acp_cl.acquisition.world import AcquisitionWorld, stage_law
from acp_cl.persistence.study import trial_seed

torch.set_num_threads(1)
RUN = ROOT / 'runs' / 'competence_gap'
world = AcquisitionWorld()
SEEDS = list(range(19101, 19107))


def feats(factors, signals, reserves):
    return np.concatenate([factors.astype(np.float32) * [1, 1, 1], signals.astype(np.float32),
                           (reserves.astype(np.float32) - 7) / 3.5], axis=1).astype(np.float32)


def stream(law, seed, first, count):
    X, A, Y, T = [], [], [], []
    for i in range(first, first + count):
        ds = trial_seed(seed, 'competence_gap_training', i)
        d = world.experience(law, 32, ds)
        c = world.dataset(law, 32, ds)
        X.append(feats(c.factors, c.signals, c.base.reserves))
        A.append(d.actions.astype(np.int64))
        Y.append(d.survival.astype(np.float32))
        T.append(world.counterfactuals(c).astype(np.float32))
    return map(np.concatenate, (X, A, Y, T))


def fit(X, A, Y, T, Xv, Av, Yv, all_actions, seed, hidden=128, wd=1e-4, epochs=400):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                        nn.Linear(hidden, 15))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
    X, A, Y, T = map(torch.as_tensor, (X, A, Y, T))
    Xv, Av, Yv = map(torch.as_tensor, (Xv, Av, Yv))
    best, state = 1e9, None
    g = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        for s in range(0, len(X), 256):
            idx = perm[s:s + 256]
            logits = net(X[idx]).reshape(-1, 5, 3)
            if all_actions:
                loss = F.binary_cross_entropy_with_logits(logits, T[idx])
            else:
                loss = F.binary_cross_entropy_with_logits(logits[torch.arange(len(idx)), A[idx]], Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 4:
            net.eval()
            with torch.no_grad():
                pv = net(Xv).reshape(-1, 5, 3).sigmoid().cummin(dim=-1).values
                v = float(((pv[torch.arange(len(Xv)), Av] - Yv) ** 2).mean())   # performed-action Brier
            if v < best:
                best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    return net


out = {}
for seed in SEEDS:
    law = stage_law(seed, 3)
    cases = world.dataset(law, 512, trial_seed(seed, 'acquisition_query', 0))
    truth = world.counterfactuals(cases).astype(np.float64)
    Xq = torch.as_tensor(feats(cases.factors, cases.signals, cases.base.reserves))
    with np.load(RUN / 'oracle' / f'seed_{seed}.npz') as z:
        pO = np.minimum.accumulate(z['endpoint_exact'][0], axis=-1)
    oB = float(((pO - truth) ** 2).mean())
    X, A, Y, T = stream(law, seed, 0, 1024)            # first 32768 training arrivals
    Xv, Av, Yv, _ = stream(law, seed, 1024, 128)        # validation: later arrivals of the same channel
    row = dict(O_brier=oB)
    for n in (8192, 32768):
        for all_actions in (False, True):
            net = fit(X[:n], A[:n], Y[:n], T[:n], Xv, Av, Yv, all_actions, seed)
            with torch.no_grad():
                p = net(Xq).reshape(-1, 5, 3).sigmoid().cummin(dim=-1).values.double().numpy()
            b = float(((p - truth) ** 2).mean())
            row[f'n{n}_{"all" if all_actions else "performed"}'] = dict(
                brier=b, excess=b - oB, mse_to_O=float(((p - pO) ** 2).mean()))
    # decomposition for the saved reference learner
    import json as _j
    recs = _j.loads((RUN / 'fits' / f'U12_R16_W64_{seed}' / 'evaluations.json').read_text())['records']
    with np.load(RUN / 'fits' / f'U12_R16_W64_{seed}' / 'evaluations.npz') as z:
        ps = [z[f'e{r["index"]}_probabilities'].astype(np.float64) for r in recs if r['cue'] is None]
    row['reference'] = dict(excess=float(np.mean([((p - truth) ** 2).mean() for p in ps])) - oB,
                            mse_to_O=float(np.mean([((p - pO) ** 2).mean() for p in ps])),
                            O_mean_p1mp=float((pO * (1 - pO)).mean()),
                            O_frac_cells_deterministic=float(((pO == 0) | (pO == 1)).mean()))
    out[str(seed)] = row
    print(seed, json.dumps(row), flush=True)
keys = [k for k in out[str(SEEDS[0])] if k.startswith('n')]
out['mean'] = {k: {m: float(np.mean([out[str(s)][k][m] for s in SEEDS])) for m in ('excess', 'mse_to_O')} for k in keys + ['reference']}
out['note'] = ('Descriptive reviewer baseline, not a protocol quantity: MLP 8->128->128->15 on TRUE observable '
               'features, AdamW lr 3e-3 wd 1e-4, 400 epochs, early stopping on performed-action Brier of 4096 '
               'later same-channel records; panel never used for selection.')
(ROOT / 'reports/competence_gap/review/review_f.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out['mean'], indent=1))

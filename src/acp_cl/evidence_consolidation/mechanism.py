"""Validate a fixed predictor only on observations arriving after its creation.

The trained draft is shared exactly across acceptance rules. This isolates
consolidation, not a representation-learning or gradient-protection mechanism.
"""

from __future__ import annotations

import copy

import numpy as np

from acp_cl.contextual.learner import ContextLearner
from acp_cl.persistence.learner import state_hash


POLICIES = ("immediate", "periodic", "single", "sustained")
DELAYED = POLICIES[1:]


def accept(policy, gains, margin, required_positive):
    values = np.asarray(gains, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty future-evidence window required")
    if policy == "periodic":
        return True
    if policy == "single":
        return bool(values[0] > margin)
    if policy == "sustained":
        return bool(values.mean() > margin and (values > 0).sum() >= required_positive)
    raise ValueError("unknown delayed policy")


class Snapshot(ContextLearner):
    """Immutable network only: no optimizer, raw replay or extra history."""

    def __init__(self, learner):
        self.method, self.device = learner.method, learner.device
        self.settings = dict(learner.settings)
        self.model = copy.deepcopy(learner.model).requires_grad_(False)
        self.history = None
        self.sha256 = state_hash(self.model.state_dict())


class Consolidation:
    def __init__(self, draft, settings):
        self.draft = draft
        self.settings = dict(settings)
        original = Snapshot(draft)
        self.committed = {policy: original for policy in DELAYED}
        self.proposal = None
        self.proposal_batch = None
        self.gains = {policy: [] for policy in DELAYED}
        self.batches = 0
        self.adoptions = {policy: 0 for policy in DELAYED}
        self.proposal_count = 0
        self.last_decision = None

    def predictors(self):
        return {"immediate": self.draft, **self.committed}

    def prepare(self, observations):
        """Called before receiving current feedback and before training on it."""
        if self.proposal is None:
            self.proposal = Snapshot(self.draft)
            self.proposal_batch = self.batches
            self.proposal_count += 1
        history = self.draft.history
        predictions = {name: learner.predict(observations, history)[0]
                       for name, learner in self.predictors().items()}
        proposed = self.proposal.predict(observations, history)[0]
        return predictions, proposed

    def observe(self, current, predictions, proposed):
        """Feedback may change the NEXT prediction; copy only the tested proposal."""
        rows, actions = np.arange(len(current)), current.actions
        y = current.survival[:, -1].astype(float)
        if not all(np.isfinite(p).all() for p in (*predictions.values(), proposed)):
            raise ValueError("nonfinite prediction before consolidation or training")
        proposal_loss = (proposed[rows, actions, -1]-y)**2
        for policy in DELAYED:
            stable_loss = (predictions[policy][rows, actions, -1]-y)**2
            self.gains[policy].append(float((stable_loss-proposal_loss).mean()))
        self.batches += 1
        decision = None
        if len(self.gains["periodic"]) == self.settings["evidence_window"]:
            decisions = {p: accept(p, self.gains[p], self.settings["evidence_margin"],
                                  self.settings["evidence_positive"]) for p in DELAYED}
            decision = dict(start_batch=self.proposal_batch, end_batch=self.batches,
                            proposal_hash=self.proposal.sha256,
                            committed_before={p: s.sha256 for p, s in self.committed.items()},
                            gains=copy.deepcopy(self.gains), accepted=decisions)
            for policy, adopted in decisions.items():
                if adopted:
                    self.committed[policy] = self.proposal
                    self.adoptions[policy] += 1
            self.gains = {policy: [] for policy in DELAYED}
            self.proposal = None
            self.proposal_batch = None
            self.last_decision = decision
        self.draft.train(current)
        return decision

    def state(self):
        return dict(batches=self.batches, adoptions=dict(self.adoptions),
                    proposal_count=self.proposal_count,
                    committed={p: s.sha256 for p, s in self.committed.items()},
                    proposal=None if self.proposal is None else self.proposal.sha256,
                    proposal_batch=self.proposal_batch, gains=copy.deepcopy(self.gains),
                    draft=self.draft.diagnostics())

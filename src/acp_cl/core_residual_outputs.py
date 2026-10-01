"""Read-only output ablations of saved core/residual learners.

An isolated residual is a correction trained jointly with a core, not a trained
standalone baseline. Crossed expert content/routing cells are diagnostics only.
"""

from __future__ import annotations

import torch

from acp_cl.conditional.learner import evidence_weights
from acp_cl.core_residual.learner import CoreResidualLearner, _recurrent_logits
from acp_cl.persistence.world import IMAGE_SHAPE


@torch.no_grad()
def component_predictions(learner, observations, support):
    """Return projected probabilities without modifying any learner state.

Only observable query images and preceding performed-action feedback enter.
For conditional learners, the first name denotes content and the second route.
Every component uses the same observations/support and original float32 math.
"""
    if not isinstance(learner, CoreResidualLearner):
        raise TypeError("a saved core/residual learner is required")
    modes = [(module, module.training) for module in learner.model.modules()]
    try:
        learner.model.eval()
        query = torch.as_tensor(observations, device=learner.device)
        if support is None:
            sx = torch.zeros((learner.settings["batch_size"], *IMAGE_SHAPE),
                             dtype=torch.uint8, device=learner.device)
            actions = torch.zeros(len(sx), dtype=torch.long, device=learner.device)
            outcomes = torch.zeros((len(sx), 3), device=learner.device)
        else:
            sx, actions, outcomes = learner.tensors(support)
        combined = torch.cat((sx, query))
        if learner.method == "conditional":
            core = learner.model.core(combined)
            residual = learner.model.residual(combined)
            content = {"core": core, "residual": residual, "full": core + residual}
            weights = {}
            for name, logits in content.items():
                if support is None:
                    weights[name] = torch.full(
                        (len(query), learner.settings["experts"]),
                        1 / learner.settings["experts"], device=learner.device)
                else:
                    weights[name] = evidence_weights(logits[:len(sx)], actions, outcomes,
                        learner.settings["evidence_strength"])[None].expand(len(query), -1)
            cells = {name: (name, name) for name in content}
            cells.update(full_core_route=("full", "core"), core_full_route=("core", "full"))
            probabilities = {
                name: (weights[route][:, :, None, None]
                       * content[body][len(sx):].sigmoid()).sum(dim=1)
                for name, (body, route) in cells.items()}
        elif learner.method == "recurrent":
            args = (combined, len(sx), len(query), actions, outcomes, support is not None)
            core = _recurrent_logits(learner.model.core, *args)
            residual = _recurrent_logits(learner.model.residual, *args)
            probabilities = {"full": (core + residual).sigmoid(),
                             "core": core.sigmoid(), "residual": residual.sigmoid()}
        else:
            raise ValueError("unsupported core/residual architecture")
        return {name: value.cummin(dim=-1).values.cpu().numpy().copy()
                for name, value in probabilities.items()}
    finally:
        for module, training in modes:
            module.training = training

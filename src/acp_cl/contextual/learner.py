"""A recurrent comparator and one query-dependent routing intervention.

The earlier conditional implementation is imported unchanged. Ordinary methods
receive exactly its causal packets and use its optimizer/replay update loop.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from acp_cl.conditional.learner import ConditionalLearner, Hypotheses
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.world import IMAGE_SHAPE


METHODS = ("pooled", "conditional", "recurrent", "query_routed", "recurrent_frozen", "oracle")


class HistoryNetwork(nn.Module):
    def __init__(self, width, context_width, decoder_width, interaction_features=False):
        super().__init__()
        # Initialization order matches the reference encoder exactly.
        self.frame = nn.Sequential(nn.Linear(192, width), nn.Tanh())
        self.temporal = nn.GRU(width, width, batch_first=True)
        self.interaction_features = interaction_features
        self.context = nn.GRU(width + 5 + 3 + (5*width if interaction_features else 0),
                              context_width, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(width + context_width, decoder_width), nn.Tanh(),
                                     nn.Linear(decoder_width, 15))

    def encode(self, observations):
        features = self.frame((observations.float() / 255).flatten(start_dim=2))
        return self.temporal(features)[1][-1]


class QueryHypotheses(Hypotheses):
    def __init__(self, width, experts):
        super().__init__(width, experts)
        self.query_route = nn.Linear(width, experts)
        nn.init.zeros_(self.query_route.weight)
        nn.init.zeros_(self.query_route.bias)

    def with_features(self, observations):
        features = self.frame((observations.float() / 255).flatten(start_dim=2))
        state = self.temporal(features)[1][-1]
        logits = torch.stack([head(state).reshape(-1, 5, 3) for head in self.heads], dim=1)
        return logits, state


class ContextLearner(ConditionalLearner):
    def __init__(self, method, seed, settings, device="cpu"):
        if method not in METHODS:
            raise ValueError("unknown contextual method")
        super().__init__(method if method in ("pooled", "conditional", "oracle") else "conditional",
                         seed, settings, device)
        self.method = method
        if method in ("recurrent", "recurrent_frozen", "query_routed"):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 73019)
                if method == "query_routed":
                    self.model = QueryHypotheses(settings["width"], settings["experts"]).to(self.device)
                else:
                    self.model = HistoryNetwork(settings["width"], settings["context_width"],
                        settings["decoder_width"], settings["interaction_features"]).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=settings["lr"])
            self.initial_hash = state_hash(self.model.state_dict())
        self.initial_encoder = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                                if k.startswith(("frame.", "temporal."))}
        self.initial_encoder_hash = state_hash(self.initial_encoder)

    def probabilities(self, observations, support, oracle_modes=None, override=None):
        if self.method in ("pooled", "conditional", "oracle"):
            return super().probabilities(observations, support, oracle_modes, override)
        if oracle_modes is not None:
            raise ValueError("ordinary models must not receive latent modes")
        query = torch.as_tensor(observations, device=self.device)
        if support is None:
            sx = torch.zeros((self.settings["batch_size"], *IMAGE_SHAPE), dtype=torch.uint8,
                             device=self.device)
            sa = torch.zeros(len(sx), dtype=torch.long, device=self.device)
            sy = torch.zeros((len(sx), 3), device=self.device)
        else:
            sx, sa, sy = self.tensors(support)
        if override == "shuffled":
            sx = sx.roll(1, dims=0)
        combined = torch.cat((sx, query))
        if self.method in ("recurrent", "recurrent_frozen"):
            features = self.model.encode(combined)
            events = torch.cat((features[:len(sx)], F.one_hot(sa, 5).float(), sy), dim=-1)
            # State is reset for each packet; no hidden memory beyond the shared
            # 32-record window, and no query outcome enters this state.
            if getattr(self.model, "interaction_features", False):
                interactions = (features[:len(sx), :, None] * F.one_hot(sa, 5)[:, None, :]
                                * (2*sy[:, -1]-1)[:, None, None]).flatten(start_dim=1)
                events = torch.cat((events, interactions), dim=-1)
                # Pool all causal recurrent outputs so every retained event has
                # a direct path to the context summary. No latent labels or
                # physical features are supplied; phi remains learned from scratch.
                context = self.model.context(events[None])[0].mean(dim=1)
            else:
                context = self.model.context(events[None])[1][-1]
            if support is None:
                context = context * 0
            logits = self.model.decoder(torch.cat((features[len(sx):],
                                      context.expand(len(query), -1)), dim=-1)).reshape(-1, 5, 3)
            weights = torch.ones((len(query), 1), device=self.device)
            return logits.sigmoid(), weights, logits[:, None]
        logits, features = self.model.with_features(combined)
        if support is None:
            log_prior = torch.zeros(self.settings["experts"], device=self.device)
        else:
            # Recompute log weights stably rather than log(clipped softmax).
            rows = torch.arange(len(sa), device=self.device)
            terminal = logits[:len(sx)][rows, :, sa, -1]
            score = -F.binary_cross_entropy_with_logits(terminal,
                sy[:, -1, None].expand_as(terminal), reduction="none").sum(dim=0)
            log_prior = self.settings["evidence_strength"] * score
        correction = self.model.query_route(features[len(sx):])
        weights = torch.softmax(log_prior[None] + correction, dim=-1)
        probabilities = (weights[:, :, None, None] * logits[len(sx):].sigmoid()).sum(dim=1)
        return probabilities, weights, logits[len(sx):]

    def diagnostics(self):
        report = super().diagnostics()
        report["initial_encoder_hash"] = self.initial_encoder_hash
        report["encoder_hash"] = state_hash({k: v for k, v in self.model.state_dict().items()
                                              if k.startswith(("frame.", "temporal."))})
        return report

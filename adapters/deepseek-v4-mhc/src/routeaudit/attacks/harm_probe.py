"""#3-MoE — distill a judge (Llama Guard) into a differentiable probe over the
target model's ROUTER features.  ⚠ EXPERIMENTAL scaffold (see README roadmap).

Why probe routing features, not the residual stream:
  - **residual-independent**: the probe reads the router input rather than assuming a
    particular residual-stream layout.
  - **MoE-native**: it reads where safety actually lives in these models (the experts).
  - **Differentiable** w.r.t. the suffix through the soft router logits — the same path
    `L_suppress` already uses — so it gives judge-aware GCG gradients at ~1× cost.

Pipeline (offline, one-time): generate responses on the target → label with Llama
Guard → capture the boundary-token router logits → train this probe to predict the
label. Then `probe_loss(...)` is a drop-in extra term in the attack that pushes the
suffix toward routing the judge calls harmful. Goodhart caveat: validate the trained
suffix with the REAL judge — the probe is a fast proxy, not ground truth.

Status: the probe + feature pooling + training here are complete and unit-tested for
shape/logic; wiring `probe_loss` into RouteAudit's loss and the distillation run
(scripts/distill_harm_probe.py) are the remaining integration steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def boundary_routing_features(router_logits: dict[int, torch.Tensor], boundary: int) -> torch.Tensor:
    """Pool captured router logits at the boundary token into one feature vector.

    `router_logits[layer]` is (T, E) (single sequence). Returns a (Σ_l E_l,) vector of
    per-layer softmax expert masses at the boundary — differentiable, gate-level, and
    independent of the residual-stream implementation."""
    feats = []
    for layer in sorted(router_logits):
        feats.append(router_logits[layer][boundary].softmax(-1))
    return torch.cat(feats, dim=-1)


def boundary_routing_features_b(router_logits: dict[int, torch.Tensor], boundary: int) -> torch.Tensor:
    """Batched: `router_logits[layer]` is (B, T, E) → (B, Σ_l E_l)."""
    feats = []
    for layer in sorted(router_logits):
        feats.append(router_logits[layer][:, boundary].softmax(-1))
    return torch.cat(feats, dim=-1)


class RoutingHarmProbe(torch.nn.Module):
    """Linear probe: routing features → P(harmful). Tiny (one Linear), so it adds a
    matmul to the attack forward. Trained on judge labels."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, 1)
        self.in_dim = in_dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.lin(feats).squeeze(-1)            # logit of "harmful"

    def harm_logprob(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.logsigmoid(self.forward(feats))


def probe_loss(probe: RoutingHarmProbe, router_logits: dict[int, torch.Tensor], boundary: int) -> torch.Tensor:
    """Differentiable attack term: −log P(harmful) at the boundary, for a candidate
    batch sharing one prompt. Lower = the routing looks more like judged-harmful.
    `router_logits[layer]` is (B, T, E). Returns (B,)."""
    feats = boundary_routing_features_b(router_logits, boundary)
    return -probe.harm_logprob(feats)


@dataclass
class ProbeTrainResult:
    probe: RoutingHarmProbe
    final_loss: float
    train_acc: float


def train_probe(features: torch.Tensor, labels: torch.Tensor, *, epochs: int = 300,
                lr: float = 1e-2, weight_decay: float = 1e-3) -> ProbeTrainResult:
    """Fit the probe (logistic regression) on (features, labels∈{0,1}). Pure/offline."""
    probe = RoutingHarmProbe(features.shape[1]).to(features.device, features.dtype)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    y = labels.to(features.dtype)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        logit = probe(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((probe(features) > 0).float() == y).float().mean().item()
    return ProbeTrainResult(probe=probe, final_loss=float(loss.item()), train_acc=acc)


def save_probe(probe: RoutingHarmProbe, path: str) -> None:
    torch.save({"in_dim": probe.in_dim, "state_dict": probe.state_dict()}, path)


def load_probe(path: str, device="cpu") -> RoutingHarmProbe:
    d = torch.load(path, map_location=device)
    probe = RoutingHarmProbe(d["in_dim"]).to(device)
    probe.load_state_dict(d["state_dict"])
    probe.eval()
    return probe

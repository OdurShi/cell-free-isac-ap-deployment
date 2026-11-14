from __future__ import annotations

import numpy as np
import torch
from torch import distributions as D

from cfisac.agents.networks import (
    GaussianActor,
    ObsNormalizer,
    QNetwork,
    SquashedGaussianActor,
    TwinQNetwork,
    soft_update,
)

LOW, HIGH = np.full(8, -10.0, dtype=np.float32), np.full(8, 60.0, dtype=np.float32)


def test_obs_normalizer_maps_box_to_unit_cube():
    norm = ObsNormalizer(LOW, HIGH)
    out = norm(torch.tensor([[-10.0] * 8, [60.0] * 8, [25.0] * 8]))
    torch.testing.assert_close(out, torch.tensor([[-1.0] * 8, [1.0] * 8, [0.0] * 8]))


def test_squashed_gaussian_log_prob_matches_torch_distributions():
    torch.manual_seed(0)
    actor = SquashedGaussianActor(8, 4, (64, 32), LOW, HIGH)
    obs = torch.rand(16, 8) * 50
    torch.manual_seed(1)
    action, log_prob = actor.sample(obs)
    assert action.shape == (16, 4) and log_prob.shape == (16,)
    assert action.abs().max() < 1.0
    mean, log_std = actor.distribution(obs)
    dist = D.TransformedDistribution(
        D.Normal(mean, log_std.exp()), [D.TanhTransform(cache_size=1)]
    )
    expected = dist.log_prob(action.clamp(-0.999999, 0.999999)).sum(-1)
    torch.testing.assert_close(log_prob, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actor(obs), torch.tanh(mean))


def test_network_shapes_match_table1():
    actor = SquashedGaussianActor(8, 8, (64, 32), LOW, HIGH)
    shapes = [tuple(p.shape) for p in actor.parameters()]
    assert shapes == [(64, 8), (64,), (32, 64), (32,), (8, 32), (8,), (8, 32), (8,)]
    critic = TwinQNetwork(8, 8, (64, 32), LOW, HIGH)
    assert [tuple(p.shape) for p in critic.q1.parameters()] == [
        (64, 16), (64,), (32, 64), (32,), (1, 32), (1,),
    ]
    q1, q2 = critic(torch.zeros(5, 8), torch.zeros(5, 8))
    assert q1.shape == q2.shape == (5,)
    q_min = critic.min_q(torch.zeros(5, 8), torch.zeros(5, 8))
    torch.testing.assert_close(q_min, torch.min(q1, q2))


def test_gaussian_actor_log_prob_and_entropy():
    torch.manual_seed(0)
    actor = GaussianActor(8, 3, (64, 32), LOW, HIGH)
    obs, raw = torch.rand(6, 8) * 50, torch.randn(6, 3)
    log_prob, entropy = actor.log_prob_entropy(obs, raw)
    dist = D.Normal(actor(obs), actor.log_std.exp())
    torch.testing.assert_close(log_prob, dist.log_prob(raw).sum(-1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, dist.entropy().sum(-1), rtol=1e-5, atol=1e-5)


def test_soft_update_rule():
    torch.manual_seed(0)
    online, target = (QNetwork(8, 2, (4,), LOW, HIGH) for _ in range(2))
    before = [p.clone() for p in target.parameters()]
    soft_update(target, online, tau=0.005)
    for old, new, src in zip(before, target.parameters(), online.parameters(), strict=True):
        torch.testing.assert_close(new, 0.005 * src + 0.995 * old)

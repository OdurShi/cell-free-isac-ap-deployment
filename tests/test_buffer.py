from __future__ import annotations

import numpy as np
import pytest
import torch

from cfisac.agents.buffer import ReplayBuffer, RolloutBuffer


def test_replay_buffer_shapes_and_consistency():
    buf = ReplayBuffer(obs_dim=3, act_dim=2, capacity=10, seed=0)
    with pytest.raises(ValueError):
        buf.sample(4)
    for i in range(6):
        buf.add(np.full(3, i), np.full(2, -i), float(i), np.full(3, i + 1), terminated=(i == 5))
    assert len(buf) == 6
    batch = buf.sample(64)
    assert batch["obs"].shape == (64, 3) and batch["action"].shape == (64, 2)
    assert batch["reward"].shape == (64,) and batch["terminal"].shape == (64,)
    assert all(v.dtype == torch.float32 for v in batch.values())
    # Every sampled row is one stored transition.
    np.testing.assert_array_equal(batch["obs"][:, 0], batch["reward"])
    np.testing.assert_array_equal(batch["next_obs"][:, 0], batch["reward"] + 1)
    np.testing.assert_array_equal(batch["action"][:, 0], -batch["reward"])
    np.testing.assert_array_equal(batch["terminal"], (batch["reward"] == 5).float())
    assert set(batch["reward"].tolist()) == set(range(6))


def test_replay_buffer_is_fifo_when_full():
    buf = ReplayBuffer(obs_dim=1, act_dim=1, capacity=4, seed=0)
    for i in range(10):
        buf.add([i], [0.0], float(i), [i + 1], False)
    assert len(buf) == 4
    assert set(buf.sample(200)["reward"].tolist()) == {6.0, 7.0, 8.0, 9.0}


def test_replay_buffer_sampling_is_seeded():
    a, b = (ReplayBuffer(1, 1, 50, seed=7) for _ in range(2))
    for i in range(50):
        for buf in (a, b):
            buf.add([i], [0.0], float(i), [i], False)
    np.testing.assert_array_equal(a.sample(16)["reward"], b.sample(16)["reward"])


def _fill(buf, rewards, values, terminated=None, truncated=None, next_values=None):
    n = len(rewards)
    terminated = terminated or [False] * n
    truncated = truncated or [False] * n
    next_values = next_values or [0.0] * n
    for i in range(n):
        buf.add(np.zeros(1), np.zeros(1), rewards[i], values[i], 0.0,
                terminated[i], truncated[i], next_values[i])


def test_gae_lambda_one_gives_discounted_returns():
    gamma = 0.9
    buf = RolloutBuffer(1, 1, 4)
    _fill(buf, rewards=[1.0, 2.0, 3.0, 4.0], values=[0.5, 0.4, 0.3, 0.2],
          next_values=[0.0, 0.0, 0.0, 10.0])
    assert buf.full
    buf.finish(gamma, 1.0)
    expected = [
        1 + gamma * 2 + gamma**2 * 3 + gamma**3 * 4 + gamma**4 * 10,
        2 + gamma * 3 + gamma**2 * 4 + gamma**3 * 10,
        3 + gamma * 4 + gamma**2 * 10,
        4 + gamma * 10,
    ]
    np.testing.assert_allclose(buf.returns, expected, rtol=1e-6)
    np.testing.assert_allclose(buf.advantage, np.array(expected) - buf.value, rtol=1e-6)


def test_gae_lambda_zero_is_one_step_td():
    buf = RolloutBuffer(1, 1, 3)
    _fill(buf, rewards=[1.0, 1.0, 1.0], values=[2.0, 3.0, 4.0], next_values=[0.0, 0.0, 5.0])
    buf.finish(0.5, 0.0)
    np.testing.assert_allclose(buf.advantage, [1 + 0.5 * 3 - 2, 1 + 0.5 * 4 - 3, 1 + 0.5 * 5 - 4])


def test_gae_handles_termination_and_truncation():
    gamma = 0.5
    buf = RolloutBuffer(1, 1, 4)
    # step 1 terminates (no bootstrap); step 2 is cut by the time limit (bootstrap with 8).
    _fill(buf, rewards=[1.0, 1.0, 1.0, 1.0], values=[0.0, 0.0, 0.0, 0.0],
          terminated=[False, True, False, False], truncated=[False, False, True, False],
          next_values=[0.0, 99.0, 8.0, 2.0])
    buf.finish(gamma, 1.0)
    np.testing.assert_allclose(buf.returns, [1 + gamma * 1, 1.0, 1 + gamma * 8, 1 + gamma * 2])
    data = buf.tensors()
    assert data["obs"].shape == (4, 1) and data["returns"].shape == (4,)
    buf.reset()
    assert len(buf) == 0 and not buf.full

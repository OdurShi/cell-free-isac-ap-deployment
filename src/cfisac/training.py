"""Training loop, periodic evaluation, CSV recorder, checkpointing, best-deployment tracking."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cfisac.agents import Agent, AgentConfig, make_agent
from cfisac.data import load_ue_positions
from cfisac.env import APDeploymentEnv
from cfisac.scenario import ScenarioConfig

logger = logging.getLogger(__name__)

#: Column names of the archived ``recorder_my.csv`` files, reused for new runs.
RECORDER_COLUMNS = (
    "steps",
    "times",
    "rewards",
    "std_rewards",
    "rewards_isac",
    "std_rewards_isac",
    "rewards_s",
    "std_rewards_s",
    "rewards_c",
    "std_rewards_c",
    "best_rewards",
    "best_rewards_isac",
    "best_rewards_s",
    "best_rewards_c",
)
RECORDER_FILE = "recorder.csv"
BEST_LOCATION_FILE = "recorder_best_location.csv"


@dataclass(frozen=True)
class TrainConfig:
    """Settings of one training run.

    Attributes:
        total_steps: environment steps (Fig. 3 shows 3e6).
        eval_interval: steps between evaluations (the archived recorders use 20224 = 79 * 256).
        eval_episodes: evaluation episodes per evaluation (deterministic policy).
        seed: seed of the networks, the training environment and the replay sampling.
        torch_threads: intra-op threads used by PyTorch.
        data_dir: directory with the UE position files (``None``: ``$CFISAC_DATA_DIR`` or the
            repository's ``data/``).
        online_ues: draw UE positions on the fly instead of reading them from ``data_dir``.
            This also happens automatically when the files are absent or belong to another
            UE distribution.
    """

    total_steps: int = 3_000_000
    eval_interval: int = 20224
    eval_episodes: int = 4
    seed: int = 1
    torch_threads: int = 1
    data_dir: str | None = None
    online_ues: bool = False


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one evaluation (``episodes`` episodes with the deterministic policy).

    ``episode_*`` hold the per-episode cumulative values (reward, ISAC objective, sensing
    factor, communication factor); ``tx`` / ``rx`` are the deployment chosen most often.
    """

    episode_reward: np.ndarray
    episode_isac: np.ndarray
    episode_sens: np.ndarray
    episode_comm: np.ndarray
    steps_per_episode: int
    tx: np.ndarray
    rx: np.ndarray

    @property
    def mean_step_reward(self) -> float:
        return float(self.episode_reward.mean() / self.steps_per_episode)


def evaluate_policy(agent: Agent, env: APDeploymentEnv, episodes: int) -> EvalResult:
    """Roll out the deterministic policy and accumulate the reward and its factors."""
    totals = np.zeros((episodes, 4))
    counts: Counter[bytes] = Counter()
    layouts: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    steps = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
            totals[ep] += (reward, info["isac"], info["sens"], info["comm"])
            key = np.concatenate([info["tx"], info["rx"]]).tobytes()
            counts[key] += 1
            layouts.setdefault(key, (info["tx"], info["rx"]))
            done = terminated or truncated
            steps += 1
    # Deterministic tie-break: most frequent deployment, then lexicographic order of its bytes.
    best_key = min(counts, key=lambda k: (-counts[k], k))
    tx, rx = layouts[best_key]
    return EvalResult(totals[:, 0], totals[:, 1], totals[:, 2], totals[:, 3], steps, tx, rx)


class Recorder:
    """Writes one CSV row per evaluation using the archived column names.

    ``rewards*`` are cumulative values over one evaluation episode (mean and standard
    deviation over the evaluation episodes), exactly like the archived files.  ``best_*`` are
    the per-step means of the best evaluation so far (the archive stores the best single
    step instead).  ``recorder_best_location.csv`` holds the best deployment so far, one row
    per evaluation, as ``x, y`` of the transmit APs followed by the receive APs.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.best: dict[str, Any] | None = None
        self._start = time.time()
        self._rows = 0
        with open(out_dir / RECORDER_FILE, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(("", *RECORDER_COLUMNS))
        (out_dir / BEST_LOCATION_FILE).write_text("", encoding="utf-8")

    def log(self, steps: int, result: EvalResult) -> bool:
        """Append a row; returns ``True`` when this evaluation is the best so far."""
        n = result.steps_per_episode
        improved = self.best is None or result.mean_step_reward > self.best["reward"]
        if improved:
            self.best = {
                "steps": steps,
                "reward": result.mean_step_reward,
                "isac": float(result.episode_isac.mean() / n),
                "sens": float(result.episode_sens.mean() / n),
                "comm": float(result.episode_comm.mean() / n),
                "tx": result.tx.tolist(),
                "rx": result.rx.tolist(),
            }
        assert self.best is not None
        row = (
            self._rows,
            steps,
            int(time.time() - self._start),
            result.episode_reward.mean(),
            result.episode_reward.std(),
            result.episode_isac.mean(),
            result.episode_isac.std(),
            result.episode_sens.mean(),
            result.episode_sens.std(),
            result.episode_comm.mean(),
            result.episode_comm.std(),
            self.best["reward"],
            self.best["isac"],
            self.best["sens"],
            self.best["comm"],
        )
        with open(self.out_dir / RECORDER_FILE, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        flat = [v for xy in (*self.best["tx"], *self.best["rx"]) for v in xy]
        with open(self.out_dir / BEST_LOCATION_FILE, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(flat)
        self._rows += 1
        return improved


def run_directory(
    root: str | Path, agent: AgentConfig, scenario: ScenarioConfig, seed: int
) -> Path:
    """``<root>/<algo>/<M>t<N>r/seed-<seed>`` -- the layout of the archived ``experiments/``."""
    return Path(root) / agent.algo / f"{scenario.n_tx}t{scenario.n_rx}r" / f"seed-{seed}"


def train(
    scenario: ScenarioConfig,
    agent_config: AgentConfig,
    train_config: TrainConfig,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Train one agent and write recorder, best deployment, hyper-parameters and checkpoint.

    The loop follows the collect-then-update pattern implied by the archived ``horizon_len``:
    ``horizon_len`` environment steps, then one update phase.  The policy is evaluated after
    the first horizon and then every ``eval_interval`` steps.

    Returns:
        Summary with the final and best evaluation and the output directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, train_config.torch_threads))
    torch.manual_seed(train_config.seed)

    pools = {"train": None, "eval": None}
    if not train_config.online_ues:
        pools = {
            split: load_ue_positions(split, scenario, train_config.data_dir) for split in pools
        }
    ue_source = {
        split: "data file" if pool is not None else "drawn on the fly"
        for split, pool in pools.items()
    }
    logger.info("UE positions: train %s, eval %s", ue_source["train"], ue_source["eval"])
    env = APDeploymentEnv(scenario, pools["train"])
    eval_env = APDeploymentEnv(scenario, pools["eval"])
    eval_env.reset(seed=train_config.seed + 1_000_003)
    capacity = min(agent_config.buffer_size, train_config.total_steps + agent_config.horizon_len)
    agent = make_agent(
        agent_config, env.observation_space, env.action_space, train_config.seed, capacity
    )
    with open(out / "hyperparameter.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "agent": agent_config.to_dict(),
                "scenario": scenario.to_dict(),
                "train": asdict(train_config),
                "ue_positions": ue_source,
            },
            handle,
            indent=2,
        )

    recorder = Recorder(out)
    obs, _ = env.reset(seed=train_config.seed)
    steps = 0
    next_eval = agent_config.horizon_len
    last: EvalResult | None = None
    start = time.time()
    while steps < train_config.total_steps:
        for _ in range(agent_config.horizon_len):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, terminated, truncated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
            steps += 1
        logs = agent.update()
        if steps >= next_eval or steps >= train_config.total_steps:
            last = evaluate_policy(agent, eval_env, train_config.eval_episodes)
            if recorder.log(steps, last):
                agent.save(out / "checkpoint_best.pt")
            next_eval += train_config.eval_interval
            logger.info(
                "steps=%d return=%.3f (per step %.4f) best=%.4f %s",
                steps,
                last.episode_reward.mean(),
                last.mean_step_reward,
                recorder.best["reward"] if recorder.best else float("nan"),
                " ".join(f"{k}={v:.3g}" for k, v in logs.items()),
            )
    agent.save(out / "checkpoint_last.pt")
    assert recorder.best is not None and last is not None
    with open(out / "best_deployment.json", "w", encoding="utf-8") as handle:
        json.dump(recorder.best, handle, indent=2)
    return {
        "out_dir": str(out),
        "steps": steps,
        "seconds": time.time() - start,
        "final_return": float(last.episode_reward.mean()),
        "final_step_reward": last.mean_step_reward,
        "final_tx": last.tx.tolist(),
        "final_rx": last.rx.tolist(),
        "best": recorder.best,
    }

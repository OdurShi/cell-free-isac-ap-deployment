"""Command-line interface: ``cfisac <command>``.

Commands: ``train``, ``optimize``, ``tables``, ``check-table3``, ``curves``, ``deployment``,
``evaluate`` and ``demo``.  Run ``cfisac <command> --help`` for the options of each.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from cfisac.agents import ALGORITHMS, AgentConfig
from cfisac.analysis import aggregate_runs, discover_runs, final_performance
from cfisac.metrics import AGGREGATIONS, OBJECTIVES, SENSING_METRICS
from cfisac.optimize import DeterministicObjective, optimise
from cfisac.paper_values import FIGURE_DEPLOYMENTS, TABLE3
from cfisac.scenario import ScenarioConfig, archived_scenario

logger = logging.getLogger("cfisac")


def _parse_points(text: str) -> np.ndarray:
    """Parse ``"x1,y1;x2,y2"`` into an array ``[L, 2]``."""
    return np.array([[float(v) for v in pair.split(",")] for pair in text.split(";")], dtype=float)


def _load_scenario(args: argparse.Namespace) -> ScenarioConfig:
    scenario = ScenarioConfig.from_yaml(args.scenario) if args.scenario else ScenarioConfig()
    changes = {
        key: getattr(args, key)
        for key in ("aggregation", "objective", "sensing_metric", "n_tx", "n_rx")
        if getattr(args, key, None) is not None
    }
    return scenario.with_updates(**changes) if changes else scenario


def _add_scenario_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", help="scenario YAML (default: built-in defaults)")
    parser.add_argument("--aggregation", choices=AGGREGATIONS, help="override the problem")
    parser.add_argument("--objective", choices=OBJECTIVES, help="override the objective")
    parser.add_argument("--sensing-metric", choices=SENSING_METRICS, dest="sensing_metric")
    parser.add_argument("--n-tx", type=int, dest="n_tx", help="override M")
    parser.add_argument("--n-rx", type=int, dest="n_rx", help="override N")


# ----------------------------------------------------------------------- train
def cmd_train(args: argparse.Namespace) -> int:
    from cfisac.training import TrainConfig, run_directory, train

    scenario = _load_scenario(args)
    agent = AgentConfig.from_yaml(args.agent) if args.agent else AgentConfig()
    if args.algo:
        agent = replace(agent, algo=args.algo)
    if args.learning_rate is not None:
        agent = replace(agent, learning_rate=args.learning_rate)
    train_cfg = TrainConfig(
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        torch_threads=args.threads,
        data_dir=args.data_dir,
        online_ues=args.online_ues,
    )
    out_dir = run_directory(args.out, agent, scenario, args.seed)
    summary = train(scenario, agent, train_cfg, out_dir)
    print(json.dumps(summary, indent=2))
    return 0


# -------------------------------------------------------------------- optimize
def cmd_optimize(args: argparse.Namespace) -> int:
    scenario = _load_scenario(args)
    result = optimise(
        scenario,
        method=args.method,
        n_ue_snapshots=args.ue_snapshots,
        n_starts=args.n_starts,
        seed=args.seed,
    )
    payload = {"scenario": scenario.to_dict(), "result": result.to_dict()}
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(json.dumps(payload["result"], indent=2))
    return 0


# ---------------------------------------------------------------------- tables
def cmd_tables(args: argparse.Namespace) -> int:
    from cfisac.tables import TableSettings, compute_tables, tables_to_markdown

    scenario = _load_scenario(args)
    settings = TableSettings(n_ue_snapshots=args.ue_snapshots, n_starts=args.n_starts,
                             seed=args.seed)
    result = compute_tables(scenario, tuple(args.counts), args.table3_count, settings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"tables_{scenario.sensing_metric}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown = tables_to_markdown(result)
    (out / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


# ---------------------------------------------------------------- check-table3
def figure_check(n_ue_snapshots: int = 8192, seed: int = 0) -> list[dict]:
    """Evaluate the deployments read off Figs. 4-5 in the archived 4T4R scenario."""
    scenario = archived_scenario(4, 4)
    objective = DeterministicObjective(scenario, n_ue_snapshots, seed)
    rows = []
    for name, (tx, rx) in FIGURE_DEPLOYMENTS.items():
        b = objective.breakdown(np.array(tx, dtype=float), np.array(rx, dtype=float))
        rows.append(
            {
                "deployment": name,
                "comm_sum": b.comm_sum,
                "sens_mean": b.sens_mean,
                "comm_min": b.comm_min,
                "sens_min": b.sens_min,
            }
        )
    return rows


def cmd_check_table3(args: argparse.Namespace) -> int:
    rows = figure_check(args.ue_snapshots, args.seed)
    lines = [
        "| deployment (read off the paper's figure) | sum_k R_k/Q | sum_p abs(Phi)/Q "
        "| min_k R_k | min_p abs(Phi) |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['deployment']} | {row['comm_sum']:.4f} | {row['sens_mean']:.2f} "
            f"| {row['comm_min']:.4f} | {row['sens_min']:.2f} |"
        )
    lines += ["", "Paper, Table 3:", "", "| objective | sum_k R_k/Q | sum_p abs(Phi)/Q "
              "| min_k R_k | min_p abs(Phi) |", "|---|---|---|---|---|"]
    for name, row in TABLE3.items():
        lines.append(
            f"| {name} | {row['comm_sum']} | {row['sens_mean']} | {row['comm_min']} "
            f"| {row['sens_min']} |"
        )
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


# ---------------------------------------------------------------------- curves
def cmd_curves(args: argparse.Namespace) -> int:
    from cfisac.plotting import plot_learning_curves

    runs = [r for r in discover_runs(args.runs) if r.setting == args.setting]
    if args.algos:
        runs = [r for r in runs if r.algo in args.algos]
    if not runs:
        print(f"no runs found under {args.runs} for setting {args.setting}", file=sys.stderr)
        return 1
    curves, summary = {}, {}
    for algo in sorted({r.algo for r in runs}):
        algo_runs = [r for r in runs if r.algo == algo]
        curves[algo] = aggregate_runs(algo_runs, args.column, args.max_steps)
        summary[algo] = final_performance(algo_runs, args.column)
    out = plot_learning_curves(curves, args.out, title=args.title, band=args.band)
    Path(out).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    for algo, stats in summary.items():
        print(f"{algo:5s} seeds={sorted(stats['per_seed'])} final {args.column}: "
              f"mean={stats['mean']:.1f} std={stats['std']:.1f} "
              f"min={stats['min']:.1f} max={stats['max']:.1f}")
    print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------ deployment
def _deployment_from_args(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.run:
        best = json.loads((Path(args.run) / "best_deployment.json").read_text("utf-8"))
        return np.array(best["tx"], dtype=float), np.array(best["rx"], dtype=float)
    if args.tx and args.rx:
        return _parse_points(args.tx), _parse_points(args.rx)
    raise SystemExit("give either --run <run directory> or both --tx and --rx")


def cmd_deployment(args: argparse.Namespace) -> int:
    from cfisac.plotting import plot_deployment

    scenario = _load_scenario(args)
    tx, rx = _deployment_from_args(args)
    scenario = scenario.with_updates(n_tx=len(tx), n_rx=len(rx))
    out = plot_deployment(scenario, tx, rx, args.out, title=args.title)
    print(f"wrote {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    scenario = _load_scenario(args)
    tx, rx = _deployment_from_args(args)
    scenario = scenario.with_updates(n_tx=len(tx), n_rx=len(rx))
    b = DeterministicObjective(scenario, args.ue_snapshots, args.seed).breakdown(tx, rx)
    print(json.dumps(
        {
            "comm_sum": b.comm_sum,
            "comm_min": b.comm_min,
            "sens_mean": b.sens_mean,
            "sens_min": b.sens_min,
            "objective": scenario.objective_value(b),
        },
        indent=2,
    ))
    return 0


# ------------------------------------------------------------------------ demo
def cmd_demo(args: argparse.Namespace) -> int:
    from cfisac.plotting import plot_deployment, plot_learning_curves
    from cfisac.training import TrainConfig, run_directory, train

    out = Path(args.out)
    scenario = archived_scenario(2, 2)
    print("Scenario: 2T2R, max-sum problem (configs/archived/maxsum_2t2r.yaml)")
    objective = DeterministicObjective(scenario, n_ue_snapshots=1024)
    tx, rx = np.array([[10.0, 10.0], [40.0, 10.0]]), np.array([[30.0, 40.0], [40.0, 10.0]])
    b = objective.breakdown(tx, rx)
    print("1) Objective of an example deployment, Eq. (17):")
    print(f"   tx={tx.tolist()}  rx={rx.tolist()}")
    print(f"   sum rate={b.comm_sum:.4f}  sensing={b.sens_mean:.3f}  "
          f"U={scenario.objective_value(b):.3f}")
    print("2) Exhaustive search over the 10 m candidate grid:")
    best = optimise(scenario, method="exhaustive", n_ue_snapshots=1024)
    print(f"   optimum U={best.value:.3f}  tx={best.tx.tolist()}  rx={best.rx.tolist()}")
    print(f"3) SAC, {args.steps} steps, learning rate {args.learning_rate:g}:")
    agent = AgentConfig(algo="sac", learning_rate=args.learning_rate)
    run_dir = run_directory(out / "runs", agent, scenario, args.seed)
    summary = train(
        scenario, agent,
        TrainConfig(total_steps=args.steps, eval_interval=2560, eval_episodes=2, seed=args.seed),
        run_dir,
    )
    per_step = summary["final_step_reward"]
    print(f"   reward per step={per_step:.3f} ({100 * per_step / best.value:.0f}% of the grid "
          f"optimum)  tx={summary['final_tx']}  rx={summary['final_rx']}")
    runs = discover_runs(out / "runs")
    plot_learning_curves({"sac": aggregate_runs(runs)}, out / "demo_learning_curve.png",
                         title="Demo: SAC on 2T2R max-sum")
    plot_deployment(scenario, summary["final_tx"], summary["final_rx"],
                    out / "demo_deployment.png", title="Demo: final SAC deployment")
    print(f"4) Learning curve, deployment map and logs written to {out}/")
    return 0


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfisac", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="log training progress")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train an agent (SAC by default)")
    _add_scenario_options(p)
    p.add_argument("--agent", help="agent YAML (default: Table 1 SAC)")
    p.add_argument("--algo", choices=ALGORITHMS, help="override the algorithm")
    p.add_argument("--learning-rate", type=float, dest="learning_rate")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=3_000_000, dest="total_steps")
    p.add_argument("--eval-interval", type=int, default=20224, dest="eval_interval")
    p.add_argument("--eval-episodes", type=int, default=4, dest="eval_episodes")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--data-dir", dest="data_dir",
                   help="directory with the UE position files (default: data/)")
    p.add_argument("--online-ues", action="store_true", dest="online_ues",
                   help="draw UE positions on the fly instead of reading data/")
    p.add_argument("--out", default="results/runs", help="root of the run directory tree")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("optimize", help="non-RL sanity optimiser on the same objective")
    _add_scenario_options(p)
    p.add_argument("--method", choices=("auto", "exhaustive", "local", "de"), default="auto")
    p.add_argument("--ue-snapshots", type=int, default=4096, dest="ue_snapshots")
    p.add_argument("--n-starts", type=int, default=32, dest="n_starts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("tables", help="Tables 2-3 from optimised deployments")
    _add_scenario_options(p)
    p.add_argument("--counts", type=int, nargs="+", default=[1, 2, 3, 4], help="values of M = N")
    p.add_argument("--table3-count", type=int, default=4, dest="table3_count")
    p.add_argument("--ue-snapshots", type=int, default=4096, dest="ue_snapshots")
    p.add_argument("--n-starts", type=int, default=48, dest="n_starts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/tables")
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("check-table3", help="evaluate the deployments shown in Figs. 4-5")
    p.add_argument("--ue-snapshots", type=int, default=8192, dest="ue_snapshots")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the comparison as Markdown")
    p.set_defaults(func=cmd_check_table3)

    p = sub.add_parser("curves", help="learning curves across algorithms and seeds (Fig. 3)")
    p.add_argument("--runs", default="experiments", help="root: <algo>/<setting>/seed-<n>/")
    p.add_argument("--setting", default="2t2r")
    p.add_argument("--algos", nargs="*", choices=ALGORITHMS)
    p.add_argument("--column", default="rewards")
    p.add_argument("--max-steps", type=float, default=None, dest="max_steps")
    p.add_argument("--band", choices=("std", "range"), default="std")
    p.add_argument("--title", default="")
    p.add_argument("--out", default="results/figures/learning_curves.png")
    p.set_defaults(func=cmd_curves)

    for name, func, text in (
        ("deployment", cmd_deployment, "deployment map (Figs. 4-5 style)"),
        ("evaluate", cmd_evaluate, "print the Table 3 quantities of one deployment"),
    ):
        p = sub.add_parser(name, help=text)
        _add_scenario_options(p)
        p.add_argument("--run", help="run directory containing best_deployment.json")
        p.add_argument("--tx", help='transmit APs, e.g. "40,10;40,0"')
        p.add_argument("--rx", help='receive APs, e.g. "40,10;40,10"')
        if name == "deployment":
            p.add_argument("--title", default="")
            p.add_argument("--out", default="results/figures/deployment.png")
        else:
            p.add_argument("--ue-snapshots", type=int, default=4096, dest="ue_snapshots")
            p.add_argument("--seed", type=int, default=0)
        p.set_defaults(func=func)

    p = sub.add_parser("demo", help="CPU quick start (one to two minutes)")
    p.add_argument("--steps", type=int, default=15360)
    p.add_argument("--learning-rate", type=float, default=3e-4, dest="learning_rate")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results/demo")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``cfisac`` console script."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

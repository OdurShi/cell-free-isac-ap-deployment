"""Tables 2 and 3 of the paper, recomputed with the non-learning optimisers.

The paper obtained its tables from trained SAC agents.  Here the same quantities are
computed for deployments found by exhaustive / local search on the same objective, which
gives an RL-independent reference for what is reachable in each setting.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfisac.optimize import (
    DeterministicObjective,
    SearchResult,
    best_from_table,
    count_grid_deployments,
    exhaustive_table,
    local_search,
)
from cfisac.scenario import ScenarioConfig

TABLE3_OBJECTIVES = ("isac", "sensing", "communication", "weighted")


@dataclass(frozen=True)
class TableSettings:
    """Numerical settings shared by both tables."""

    n_ue_snapshots: int = 4096
    n_starts: int = 48
    seed: int = 0
    exhaustive_limit: int = 500_000


def _search_all(
    scenario: ScenarioConfig, settings: TableSettings, objectives: tuple[str, ...]
) -> dict[tuple[str, str], SearchResult]:
    """Best deployment for every (aggregation, objective) pair of one antenna setting."""
    out: dict[tuple[str, str], SearchResult] = {}
    base = DeterministicObjective(scenario, settings.n_ue_snapshots, settings.seed)
    if count_grid_deployments(scenario) <= settings.exhaustive_limit:
        table = exhaustive_table(base)
        for aggregation in ("max-sum", "max-min"):
            for objective in objectives:
                out[(aggregation, objective)] = best_from_table(base, table, aggregation, objective)
        return out
    pairs = [(agg, name) for agg in ("max-sum", "max-min") for name in objectives]
    searchers = {
        pair: DeterministicObjective(
            scenario.with_updates(aggregation=pair[0], objective=pair[1]),
            settings.n_ue_snapshots,
            settings.seed,
        )
        for pair in pairs
    }
    for pair in pairs:
        out[pair] = local_search(searchers[pair], settings.n_starts, settings.seed)
    # Second pass: restart every search from the deployments found for the other objectives,
    # so that no reported optimum is beaten by a deployment already on the table.
    found = [base.site_indices(result.tx, result.rx) for result in out.values()]
    for pair in pairs:
        refined = local_search(
            searchers[pair], n_starts=0, seed=settings.seed, colocated_starts=False,
            extra_starts=found,
        )
        if refined.value > out[pair].value:
            out[pair] = refined
    return out


def compute_tables(
    base: ScenarioConfig,
    ap_counts: tuple[int, ...] = (1, 2, 3, 4),
    table3_count: int = 4,
    settings: TableSettings | None = None,
) -> dict:
    """Compute Table 2 and Table 3 for one scenario family.

    Args:
        base: scenario whose ``n_tx`` / ``n_rx`` are overridden; its ``sensing_metric`` and
            geometry are kept.
        ap_counts: values of ``M = N`` for Table 2.  The paper labels its columns
            ``M + N = 2, 4, 6, 8``, i.e. ``M = N = 1, 2, 3, 4``.
        table3_count: ``M = N`` used for Table 3 (the sensing entries of the paper's Table 3
            are reproduced by ``M = N = 4`` deployments, see ``docs/implementation_notes.md``).
        settings: numerical settings.

    Returns:
        ``{"table2": [...], "table3": {...}}`` with plain-Python content.
    """
    settings = settings or TableSettings()
    table2 = []
    searches: dict[int, dict[tuple[str, str], SearchResult]] = {}
    for count in sorted(set(ap_counts) | {table3_count}):
        scenario = base.with_updates(n_tx=count, n_rx=count)
        objectives = TABLE3_OBJECTIVES if count == table3_count else ("isac",)
        searches[count] = _search_all(scenario, settings, objectives)
    for count in ap_counts:
        best_sum = searches[count][("max-sum", "isac")]
        best_min = searches[count][("max-min", "isac")]
        table2.append(
            {
                "n_tx": count,
                "n_rx": count,
                "n_aps": 2 * count,
                "max_sum": best_sum.value,
                "max_sum_deployment": best_sum.to_dict(),
                "max_min": best_min.value,
                "max_min_deployment": best_min.to_dict(),
            }
        )
    table3 = {
        objective: {
            "max_sum": searches[table3_count][("max-sum", objective)].to_dict(),
            "max_min": searches[table3_count][("max-min", objective)].to_dict(),
        }
        for objective in TABLE3_OBJECTIVES
    }
    return {
        "sensing_metric": base.sensing_metric,
        "table3_n_tx": table3_count,
        "settings": dict(settings.__dict__),
        "table2": table2,
        "table3": table3,
    }


def _fmt(value: float) -> str:
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4g}"


def tables_to_markdown(result: dict) -> str:
    """Render the output of :func:`compute_tables` as GitHub-flavoured Markdown."""
    lines = [f"Sensing metric: `{result['sensing_metric']}`", ""]
    lines += ["**Table 2 layout** (best objective value found, Eq. 17)", ""]
    header = "| M = N (M+N) | " + " | ".join(
        f"{r['n_tx']} ({r['n_aps']})" for r in result["table2"]
    ) + " |"
    lines += [header, "|" + "---|" * (len(result["table2"]) + 1)]
    lines.append("| max-sum | " + " | ".join(_fmt(r["max_sum"]) for r in result["table2"]) + " |")
    lines.append("| max-min | " + " | ".join(_fmt(r["max_min"]) for r in result["table2"]) + " |")
    m = result["table3_n_tx"]
    lines += ["", f"**Table 3 layout** (M = N = {m}; columns are optimised objectives)", ""]
    names = {
        "isac": "ISAC",
        "sensing": "Sensing only",
        "communication": "Comm. only",
        "weighted": "Weighted sum",
    }
    lines += [
        "| quantity | " + " | ".join(names[o] for o in TABLE3_OBJECTIVES) + " |",
        "|---|---|---|---|---|",
    ]
    rows = (
        ("sum_k R_k / Q (max-sum optimum)", "max_sum", "comm_sum"),
        ("sum_p abs(Phi) / Q (max-sum optimum)", "max_sum", "sens_mean"),
        ("min_k R_k (max-min optimum)", "max_min", "comm_min"),
        ("min_p abs(Phi) (max-min optimum)", "max_min", "sens_min"),
    )
    for label, agg, key in rows:
        cells = " | ".join(_fmt(result["table3"][o][agg][key]) for o in TABLE3_OBJECTIVES)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines) + "\n"

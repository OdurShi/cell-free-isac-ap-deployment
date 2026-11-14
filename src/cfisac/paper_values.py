"""Numbers printed in the paper and deployments read off its figures.

Nothing here is produced by this code base.  ``TABLE2`` / ``TABLE3`` are copied from the
paper for side-by-side comparison; ``FIGURE_DEPLOYMENTS`` are AP positions read by eye from
Figs. 4-5 (10 m grid; APs sharing a site are drawn with a small offset in the paper) and are
used to check which evaluation setting reproduces the sensing entries of Table 3.
"""

from __future__ import annotations

#: Table 2, "ISAC value of different number of APs"; keys are the paper's column labels.
TABLE2 = {
    "max_sum": {2: 41.07, 4: 775.73, 6: 3804.47, 8: 11614.07},
    "max_min": {2: 0.042, 4: 10.48, 6: 86.06, 8: 257.28},
}

#: Table 3, "Communication rate and localizing accuracy of different objective".
TABLE3 = {
    "isac": {"comm_sum": 1.13, "sens_mean": 688.97, "comm_min": 0.042, "sens_min": 0.25},
    "sensing": {"comm_sum": 0.014, "sens_mean": 842.92, "comm_min": 0.0013, "sens_min": 696.89},
    "communication": {"comm_sum": 1.26, "sens_mean": 642.08, "comm_min": 0.043, "sens_min": 188.40},
    "weighted": {"comm_sum": 0.015, "sens_mean": 842.92, "comm_min": 0.0013, "sens_min": 696.89},
}

#: AP positions (metres) read off Figs. 4-5: ``(transmit APs, receive APs)``.
FIGURE_DEPLOYMENTS = {
    "fig4a_sac_max_sum": (
        ((40, 10), (40, 10), (40, 10), (40, 0)),
        ((40, 10), (40, 10), (40, 10), (40, 0)),
    ),
    "fig4e_comm_only_max_sum": (
        ((40, 10), (40, 10), (40, 10), (40, 10)),
        ((40, 10), (40, 10), (40, 10), (40, 20)),
    ),
    "fig4f_sensing_only": (
        ((0, 0), (0, 0), (0, 0), (0, 0)),
        ((0, 0), (0, 0), (0, 0), (0, 0)),
    ),
    "fig5a_sac_max_min": (
        ((20, 40), (30, 40), (10, 20), (40, 10)),
        ((20, 40), (30, 40), (10, 10), (40, 10)),
    ),
    "fig5e_comm_only_max_min": (
        ((20, 40), (30, 40), (30, 40), (10, 10)),
        ((30, 40), (10, 20), (10, 20), (40, 10)),
    ),
}

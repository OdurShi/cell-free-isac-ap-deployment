# Archive consistency analysis

Distinct recorded deployments: 77 (12 with 1T1R, 65 with 2T2R).

## Sensing factor (`best_rewards_s`) under candidate settings

| sensing value | trajectory centre | trajectory samples | max rel. error | median |
|---|---|---|---|---|
| A*B - C (`archived`) | (25.05, 25.05) | linspace(0, 2pi, 25), endpoint | 4.27e-16 | 1.23e-16 |
| A*B - C (`archived`) | (25, 25) | linspace(0, 2pi, 25), endpoint | 3.28e-03 | 9.38e-04 |
| A*B - C (`archived`) | (25.05, 25.05) | 24 distinct points | 4.20e-02 | 1.45e-02 |
| A*B - C (`archived`) | (25.05, 25.05) | 256 distinct points | 4.02e-02 | 1.47e-02 |
| A*B - C^2 (`eq8`) | (25.05, 25.05) | linspace(0, 2pi, 25), endpoint | 1.00e+00 | 7.89e-01 |

## Communication factor (`rewards_c / 256`) under candidate logarithm bases

Recomputed as the mean sum rate over a fixed sample of UE snapshots (UE centres [[25.0, 45.0], [10.0, 15.0], [40.0, 10.0]], variance 2).

| log base | median signed rel. error | median abs. rel. error | max abs. rel. error |
|---|---|---|---|
| 10 | +0.013 | 0.014 | 0.073 |
| e | +1.332 | 1.332 | 1.470 |
| 2 | +2.365 | 2.365 | 2.564 |

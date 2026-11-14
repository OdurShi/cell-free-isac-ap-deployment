#!/usr/bin/env bash
# Reduced-length training runs whose logs are stored under results/runs/.
#
# Hyper-parameters of Table 1, 163,840 environment steps each, 2T2R scenario of
# configs/archived/: SAC on the max-sum and max-min problems, TD3 and DDPG on the max-sum
# problem.  Four processes in parallel, one PyTorch thread each.
#
#   bash scripts/run_short_training.sh
#
# Set CFISAC to the console script if it is not on PATH, e.g. CFISAC=.venv/bin/cfisac.
set -euo pipefail
cd "$(dirname "$0")/.."
CFISAC="${CFISAC:-cfisac}"
OUT="${OUT:-results/runs}"
STEPS="${STEPS:-163840}"
# --online-ues: UE positions are drawn on the fly, as in the runs stored under results/runs/.
COMMON=(--seed 1 --eval-interval 5120 --eval-episodes 2 --total-steps "$STEPS" --online-ues)

"$CFISAC" train --scenario configs/archived/maxsum_2t2r.yaml --agent configs/agents/sac.yaml \
  "${COMMON[@]}" --out "$OUT/paper_hparams_maxsum" > /dev/null &
"$CFISAC" train --scenario configs/archived/maxmin_2t2r.yaml --agent configs/agents/sac.yaml \
  "${COMMON[@]}" --out "$OUT/paper_hparams_maxmin" > /dev/null &
"$CFISAC" train --scenario configs/archived/maxsum_2t2r.yaml --agent configs/agents/td3.yaml \
  "${COMMON[@]}" --out "$OUT/paper_hparams_maxsum" > /dev/null &
"$CFISAC" train --scenario configs/archived/maxsum_2t2r.yaml --agent configs/agents/ddpg.yaml \
  "${COMMON[@]}" --out "$OUT/paper_hparams_maxsum" > /dev/null &
wait

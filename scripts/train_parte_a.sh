#!/bin/bash
# Parte A: entrena A2C y PPO (2 semillas c/u) en paralelo, cada uno con su propia instancia de Gazebo.
#   scripts/train_parte_a.sh [TOTAL_STEPS] [MAX_HOURS]
STEPS=${1:-150000}
HOURS=${2:-3}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
i=0
for algo in a2c ppo; do
  for seed in 0 1; do
    i=$((i + 1))
    out=results/parte_a/${algo}_s${seed}
    mkdir -p $out
    scripts/run_sim.sh $((40 + i)) $out/train.log \
      python3 -m lab4_rl.train --algo $algo --seed $seed --total-steps $STEPS \
      --max-hours $HOURS --out $out > /dev/null 2>&1 &
    sleep 20   # escalonar el arranque de los gzserver
  done
done
wait
echo "PARTE A TRAINING DONE"

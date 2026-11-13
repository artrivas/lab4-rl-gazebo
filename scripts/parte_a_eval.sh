#!/bin/bash
# Parte A: evalua (20 episodios, politica determinista) actor_best y actor_last de cada corrida,
# elige el mejor -> results/actor_base.pth y genera las graficas.
EPISODES=${1:-20}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p results/parte_a/eval
i=0
for kind in best last; do
  for run in a2c_s0 a2c_s1 ppo_s0 ppo_s1; do
    a=results/parte_a/$run/actor_$kind.pth
    [ -f $a ] || continue
    i=$((i + 1))
    scripts/run_sim.sh $((50 + i)) results/parte_a/eval/${run}_${kind}.log \
      python3 -m lab4_rl.evaluate --actor $a --episodes $EPISODES \
      --out results/parte_a/eval/${run}_${kind} > /dev/null 2>&1 &
    sleep 15
  done
  wait
done

python3 -m lab4_rl.select_best --evals results/parte_a/eval --out results/actor_base.pth \
  | tee results/parte_a/eval/seleccion.txt
BEST_A2C=$(python3 -c "import json;print(json.load(open('results/actor_base.json'))['best_per_algo']['a2c'])")
BEST_PPO=$(python3 -c "import json;print(json.load(open('results/actor_base.json'))['best_per_algo']['ppo'])")
python3 -m lab4_rl.plots.parte_a --runs results/parte_a --out results/graficas \
  --evals "A2C ($BEST_A2C)=results/parte_a/eval/$BEST_A2C" "PPO ($BEST_PPO)=results/parte_a/eval/$BEST_PPO"
echo "PARTE A EVAL DONE"

#!/bin/bash
# Parte C completa: pares -> modelo Bradley-Terry -> PPO+KL (beta=0, 0.2, 1.0 en paralelo)
#                   -> prueba de estres con ruido LiDAR -> graficas B y C.
#   scripts/parte_c.sh [RLHF_STEPS=40000] [LABELER=auto|interactive]
STEPS=${1:-40000}
LABELER=${2:-auto}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
C=results/parte_c
DAG=results/parte_b/dagger_sim.pth
mkdir -p $C/stress

scripts/run_sim.sh 62 $C/pairs.log python3 -m lab4_rl.parte_c pairs --actor $DAG --out $C \
  --n-pairs 20 --labeler $LABELER
source scripts/ros_env.sh
python3 -m lab4_rl.parte_c rm --out $C | tee $C/rm.log

i=0
for beta in 0 0.2 1.0; do
  i=$((i + 1))
  scripts/run_sim.sh $((62 + i)) $C/rlhf_beta${beta}.log python3 -m lab4_rl.parte_c rlhf \
    --actor $DAG --out $C --beta $beta --total-steps $STEPS > /dev/null 2>&1 &
  sleep 20
done
wait

i=0
for spec in "BC=results/parte_b/bc.pth" "DAgger=$DAG" "RLHF_beta0=$C/rlhf_beta0/actor_rlhf.pth" \
            "RLHF_beta0.2=$C/rlhf_beta0.2/actor_rlhf.pth" "RLHF_beta1=$C/rlhf_beta1/actor_rlhf.pth"; do
  i=$((i + 1))
  label=${spec%%=*}; actor=${spec#*=}
  scripts/run_sim.sh $((70 + i)) $C/stress/$label.log python3 -m lab4_rl.stress --actor $actor \
    --label $label --noises 0 0.05 0.1 --episodes 10 --out $C/stress > /dev/null 2>&1 &
  sleep 15
done
wait

python3 -m lab4_rl.plots.parte_bc --b results/parte_b --c $C --out results/graficas
echo "PARTE C DONE"

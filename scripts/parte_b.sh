#!/bin/bash
# Parte B: D0 del experto -> BC -> 4 rondas de DAgger con intervencion -> dagger_sim.pth
#   scripts/parte_b.sh [oracle|keyboard]
#   (modo keyboard: el humano interviene con el teclado; usar GUI y tiempo real)
EXPERT=${1:-oracle}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [ "$EXPERT" = keyboard ]; then export GUI=true REALTIME=true; fi
scripts/run_sim.sh 61 results/parte_b/parte_b.log \
  python3 -m lab4_rl.parte_b --out results/parte_b --expert "$EXPERT"
source scripts/ros_env.sh
python3 -m lab4_rl.plots.parte_bc --b results/parte_b --c results/parte_c --out results/graficas

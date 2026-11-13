#!/bin/bash
# Abre Gazebo con GUI en tiempo real y corre un actor (para grabar el video de la Fase 2).
#   scripts/demo_video.sh [ACTOR=results/actor_base.pth] [EPISODIOS=5]
ACTOR=${1:-results/actor_base.pth}
EPS=${2:-5}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
GUI=true REALTIME=true scripts/run_sim.sh 30 results/demo.log \
  python3 -m lab4_rl.evaluate --actor "$ACTOR" --episodes "$EPS" --out results/demo

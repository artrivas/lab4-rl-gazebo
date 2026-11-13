#!/bin/bash
# VIDEO - Fase 2: A2C/PPO, BC vs DAgger (con correccion del experto) y RLHF, en Gazebo con GUI
# en TIEMPO REAL. Grabar la pantalla (Win+Alt+R) con la terminal y Gazebo lado a lado.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
GUI=true REALTIME=true scripts/run_sim.sh 30 results/demo.log python3 -m lab4_rl.demo "$@"

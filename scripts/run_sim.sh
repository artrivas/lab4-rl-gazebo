#!/bin/bash
# Lanza Gazebo (mundo lab4) + un comando python contra esa simulacion, aislado por ROS_DOMAIN_ID
# y puerto de Gazebo, de modo que se pueden correr varias simulaciones en paralelo.
#
#   scripts/run_sim.sh <DOMAIN_ID> <LOGFILE> <comando...>
#   GUI=true REALTIME=true scripts/run_sim.sh 30 results/eval.log python3 -m lab4_rl.evaluate ...
#
# Al terminar el comando se apaga SOLO esa instancia de Gazebo.
DOMAIN=$1; LOGFILE=$2; shift 2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/ros_env.sh"
export ROS_DOMAIN_ID=$DOMAIN
export GAZEBO_MASTER_URI=http://localhost:$((11345 + DOMAIN))
mkdir -p "$(dirname "$LOGFILE")"

setsid ros2 launch "$REPO/launch/lab4_world.launch.py" \
  gui:=${GUI:-false} realtime:=${REALTIME:-false} > "${LOGFILE%.log}_gazebo.log" 2>&1 &
GZ_PGID=$!
stop_sim() { kill -INT -- -$GZ_PGID 2>/dev/null; sleep 4; kill -9 -- -$GZ_PGID 2>/dev/null; }
trap stop_sim EXIT INT TERM

cd "$REPO"
"$@" 2>&1 | tee "$LOGFILE"

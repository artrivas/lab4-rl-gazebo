#!/bin/bash
# VIDEO - Fase 1: entorno levantado y validado con el DQN base preentrenado (stage 4).
# Abre Gazebo, dqn_gazebo, dqn_environment y dqn_test (model1.h5, sin entrenar) por ~60 s.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/ros_env.sh"          # incluye el fix del plugin de obstaculos
export ROS_DOMAIN_ID=30
export GAZEBO_MASTER_URI=http://localhost:11345
SECS=${1:-60}
setsid ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage4.launch.py > /tmp/v1_gz.log 2>&1 &
G=$!
trap 'kill -INT -- -$G 2>/dev/null; pkill -f "turtlebot3_dqn/dqn_"; sleep 3; kill -9 -- -$G 2>/dev/null' EXIT
sleep 40
ros2 run turtlebot3_dqn dqn_gazebo 4 > /tmp/v1_g.log 2>&1 &
sleep 3
ros2 run turtlebot3_dqn dqn_environment > /tmp/v1_e.log 2>&1 &
sleep 3
timeout $SECS ros2 run turtlebot3_dqn dqn_test --ros-args -p model_file:=model1.h5 -p use_gpu:=false -p verbose:=true

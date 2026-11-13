#!/bin/bash
# Tarea Previa Lab 4 - levanta el pipeline DQN completo (stage 4) en la distro Curso-Robotica.
# Uso (desde Windows):  wsl -d Curso-Robotica -u rlutec -- bash ~/lab4_run.sh
# Variables opcionales: USE_GPU=true|false  MODEL_FILE=model1.h5  MAX_EPISODES=1000  STAGE=4
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models:/opt/ros/humble/share/turtlebot3_description/models:/opt/ros/humble/share
export LC_NUMERIC=C
export ROS_DOMAIN_ID=30
export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger}
source /usr/share/gazebo/setup.sh
source /opt/ros/humble/setup.bash
source "$HOME/rl_ws/install/setup.bash"

LOGDIR="$HOME/lab4_logs"
mkdir -p "$LOGDIR"
STAGE=${STAGE:-4}
GPU=${USE_GPU:-false}
MODEL=${MODEL_FILE:-model1.h5}
MAXEP=${MAX_EPISODES:-1000}

cleanup() { kill $(jobs -p) 2>/dev/null; pkill -f gzserver; pkill -f gzclient; }
trap cleanup EXIT

# 1. Gazebo
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage${STAGE}.launch.py > "$LOGDIR/1_gazebo.log" 2>&1 &
sleep 15
# 2. Controlador del simulador (goal / reset)
ros2 run turtlebot3_dqn dqn_gazebo "$STAGE" > "$LOGDIR/2_dqn_gazebo.log" 2>&1 &
sleep 3
# 3. Entorno DQN (estado, reward, done)
ros2 run turtlebot3_dqn dqn_environment > "$LOGDIR/3_dqn_environment.log" 2>&1 &
sleep 3
# 4. Agente DQN
PYTHONUNBUFFERED=1 ros2 run turtlebot3_dqn dqn_agent --ros-args \
  -p epsilon_decay:=6000 -p max_training_episodes:=$MAXEP \
  -p use_gpu:=$GPU -p model_file:=$MODEL -p verbose:=false > "$LOGDIR/4_dqn_agent.log" 2>&1
echo "AGENT EXITED code=$?"

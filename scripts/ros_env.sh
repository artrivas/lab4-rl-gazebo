# source este archivo para tener ROS 2 + Gazebo + workspace TurtleBot3 listos
export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger}
export LC_NUMERIC=C
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models:/opt/ros/humble/share/turtlebot3_description/models:/opt/ros/humble/share
source /usr/share/gazebo/setup.sh
source /opt/ros/humble/setup.bash
source "$HOME/rl_ws/install/setup.bash"
# libobstacles.so (plugin que hace girar los cilindros) solo existe en build/ del workspace:
# sin esto Gazebo lo ignora en silencio y los obstaculos quedan ESTATICOS.
export GAZEBO_PLUGIN_PATH=$HOME/rl_ws/build/turtlebot3_gazebo:$GAZEBO_PLUGIN_PATH
export PYTHONUNBUFFERED=1
# evita que gzserver intente descargar modelos de internet al arrancar (arranque lento)
export GAZEBO_MODEL_DATABASE_URI=""

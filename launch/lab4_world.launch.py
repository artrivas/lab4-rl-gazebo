#!/usr/bin/env python3
"""Lab 4 - Levanta Gazebo con el mundo de obstaculos moviles + TurtleBot3 burger.

Argumentos:
  gui:=true|false        abre gzclient (false para entrenar headless)
  realtime:=true|false   true -> tiempo real (video); false -> lo mas rapido posible (entrenamiento)

Ejemplo:
  ros2 launch ~/lab4_rl/launch/lab4_world.launch.py gui:=false realtime:=false
"""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))
WORLD_TEMPLATE = os.path.join(HERE, '..', 'worlds', 'lab4_stage3.world.in')


def _make_world(context, *args, **kwargs):
    realtime = LaunchConfiguration('realtime').perform(context).lower() == 'true'
    with open(WORLD_TEMPLATE) as f:
        sdf = f.read().replace('@RTU@', '1000.0' if realtime else '0')
    fd, path = tempfile.mkstemp(prefix='lab4_world_', suffix='.world')
    with os.fdopen(fd, 'w') as f:
        f.write(sdf)

    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    tb3_share = get_package_share_directory('turtlebot3_gazebo')
    tb3_launch = os.path.join(tb3_share, 'launch')
    model = os.environ.get('TURTLEBOT3_MODEL', 'burger')
    model_sdf = os.path.join(tb3_share, 'models', f'turtlebot3_{model}', 'model.sdf')
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': path}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
            condition=IfCondition(LaunchConfiguration('gui'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3_launch, 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items()),
        # spawn propio con timeout largo (gzserver puede tardar >30 s en levantar)
        Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
             arguments=['-entity', model, '-file', model_sdf,
                        '-x', '0.0', '-y', '0.0', '-z', '0.01', '-timeout', '240']),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('realtime', default_value='true'),
        OpaqueFunction(function=_make_world),
    ])

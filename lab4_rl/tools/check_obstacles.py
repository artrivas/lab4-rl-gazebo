#!/usr/bin/env python3
"""Verifica que los obstaculos se muevan: imprime la pose del modelo de obstaculos en el tiempo."""
import math
import time

import rclpy
from gazebo_msgs.msg import ModelStates
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data

rclpy.init()
node = rclpy.create_node('lab4_check_obstacles')
state = {}
node.create_subscription(ModelStates, 'model_states', lambda m: state.update(ms=m), 10)
node.create_subscription(Clock, 'clock', lambda m: state.update(t=m.clock.sec + m.clock.nanosec * 1e-9), qos_profile_sensor_data)
t_end = time.time() + 20
last = -10
while time.time() < t_end:
    rclpy.spin_once(node, timeout_sec=0.1)
    if 'ms' in state and 't' in state and state['t'] - last >= 2.0:
        last = state['t']
        out = []
        for name, p in zip(state['ms'].name, state['ms'].pose):
            if 'obstacle' in name or name == 'burger':
                q = p.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                out.append(f'{name}: ({p.position.x:.2f},{p.position.y:.2f}) yaw={math.degrees(yaw):.0f}')
        print(f't_sim={state["t"]:.1f}  ' + ' | '.join(out), flush=True)
rclpy.shutdown()

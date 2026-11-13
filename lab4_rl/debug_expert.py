#!/usr/bin/env python3
"""Compara la distancia a obstaculos predicha por el experto (t=0) con la medida por el LiDAR."""
import numpy as np

from lab4_rl.env import OBSTACLE_RADIUS, TB3ContinuousEnv
from lab4_rl.expert import OracleExpert

env = TB3ContinuousEnv(seed=7, node_name='lab4_dbg')
ex = OracleExpert()
env._spin_until(lambda: env.obstacles_yaw is not None, 30)
for ep in range(3):
    obs, info = env.reset()
    done = False
    while not done:
        x, y, yaw = env.pose
        obsxy = env.obstacles_xy()
        d_pred = np.hypot(obsxy[:, 0] - x, obsxy[:, 1] - y).min() - OBSTACLE_RADIUS
        v, w = ex.act(env)
        if d_pred < 0.8 or env.scan.min() < 0.5:
            print(f'ep{ep} t={info["t"]:5.1f} pos=({x:.2f},{y:.2f}) obs_yaw={np.degrees(env.obstacles_yaw):6.1f} '
                  f'pred_surface={d_pred:.2f} lidar_min={env.scan.min():.2f} '
                  f'lidar_argmin_deg={np.argmin(env.scan) * 7.5:.0f} cmd=({v:.2f},{w:.2f})', flush=True)
        obs, r, term, trunc, info = env.step([v, w])
        done = term or trunc
    print('->', info['outcome'], flush=True)
env.close()

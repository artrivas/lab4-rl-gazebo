#!/usr/bin/env python3
"""Prueba rapida del entorno: controlador proporcional hacia la meta (sin evasion).
Mide pasos/s (wall) y factor de aceleracion respecto al tiempo real."""
import time

from lab4_rl.env import V_MAX, W_MAX, TB3ContinuousEnv

env = TB3ContinuousEnv(seed=1, node_name='lab4_smoke')
print('action_space', env.action_space, 'obs_space', env.observation_space.shape)
t0, steps, sim_t = time.time(), 0, 0.0
for ep in range(4):
    obs, info = env.reset()
    done, ret = False, 0.0
    while not done:
        ang = info['goal_angle'] if info['step'] else obs[25] * 3.14159
        w = max(-W_MAX, min(W_MAX, 2.0 * ang))
        v = V_MAX if abs(ang) < 0.5 else 0.05
        obs, r, term, trunc, info = env.step([v, w])
        ret += r
        steps += 1
        done = term or trunc
    sim_t += info['t']
    print(f'ep {ep}: {info["outcome"]:9s} ret {ret:7.1f} steps {info["step"]} sim_t {info["t"]:.1f}s '
          f'pos ({info["x"]:.2f},{info["y"]:.2f}) min_scan {info["min_scan"]:.2f}', flush=True)
wall = time.time() - t0
print(f'{steps} steps in {wall:.1f}s wall -> {steps / wall:.1f} steps/s, '
      f'speed-up x{sim_t / wall:.1f}')
env.close()

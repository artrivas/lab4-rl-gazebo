#!/usr/bin/env python3
"""Mide la tasa de exito del experto oraculo."""
import argparse

import numpy as np

from lab4_rl.env import TB3ContinuousEnv
from lab4_rl.expert import OracleExpert

p = argparse.ArgumentParser()
p.add_argument('--episodes', type=int, default=15)
p.add_argument('--start-noise', type=float, default=0.25)
args = p.parse_args()

env = TB3ContinuousEnv(seed=7, start_noise=args.start_noise, node_name='lab4_test_expert')
expert = OracleExpert()
env._spin_until(lambda: env.obstacles_yaw is not None, 30)
outs, lens, dws = [], [], []
for ep in range(args.episodes):
    obs, info = env.reset()
    done, ws = False, []
    while not done:
        v, w = expert.act(env)
        obs, r, term, trunc, info = env.step([v, w])
        ws.append(w)
        done = term or trunc
    outs.append(info['outcome'])
    lens.append(info['t'])
    dws.append(np.abs(np.diff(ws)).mean() if len(ws) > 1 else 0)
    print(f'ep {ep}: {info["outcome"]:9s} t={info["t"]:.1f}s mean|dw|={dws[-1]:.2f}', flush=True)
print('success', np.mean([o == 'goal' for o in outs]), 'collision',
      np.mean([o == 'collision' for o in outs]), 'mean t', np.mean(lens), 'mean|dw|', np.mean(dws))
env.close()

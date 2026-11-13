#!/usr/bin/env python3
"""Evalua un actor entrenado en Gazebo y guarda trayectorias paso a paso.

  python3 -m lab4_rl.evaluate --actor results/actor_base.pth --episodes 10 --out results/eval_ppo

Salidas: <out>_steps.csv (t, x, y, v, w, ...), <out>_summary.json (success rate, colisiones, ...)
"""
import argparse
import csv
import json

import numpy as np

from lab4_rl.agents import load_actor
from lab4_rl.env import TB3ContinuousEnv, norm_to_cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--actor', required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--stochastic', action='store_true', help='muestrear en vez de usar la media')
    p.add_argument('--lidar-noise', type=float, default=0.0, help='std (m) de ruido en el LiDAR')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--start-noise', type=float, default=0.25, help='+- m de la posicion inicial')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    actor, meta = load_actor(args.actor)
    env = TB3ContinuousEnv(seed=args.seed, lidar_noise_std=args.lidar_noise,
                           start_noise=args.start_noise, node_name='lab4_eval')
    rows, outcomes, returns, times = [], [], [], []
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        rows.append([ep, 0, 0.0, info['x'], info['y'], info['yaw'], 0.0, 0.0, 0.0,
                     info['dist'], info['min_scan'], 'start'])
        ret, done = 0.0, False
        while not done:
            u, _ = actor.act(obs, deterministic=not args.stochastic)
            v, w = norm_to_cmd(u)
            obs, r, term, trunc, info = env.step([v, w])
            ret += r
            done = term or trunc
            rows.append([ep, info['step'], round(info['t'], 3), round(info['x'], 4),
                         round(info['y'], 4), round(info['yaw'], 4), round(v, 4), round(w, 4),
                         round(r, 4), round(info['dist'], 4), round(info['min_scan'], 4),
                         info['outcome']])
        outcomes.append(info['outcome'])
        returns.append(ret)
        times.append(info['t'])
        print(f'ep {ep:2d}: {info["outcome"]:9s} return {ret:7.1f}  t={info["t"]:.1f}s', flush=True)

    with open(args.out + '_steps.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'step', 't', 'x', 'y', 'yaw', 'v', 'w', 'reward', 'dist',
                    'min_scan', 'outcome'])
        w.writerows(rows)
    succ = [o == 'goal' for o in outcomes]
    summary = {
        'actor': args.actor, 'actor_meta': meta, 'episodes': args.episodes,
        'deterministic': not args.stochastic, 'lidar_noise_std': args.lidar_noise,
        'success_rate': float(np.mean(succ)),
        'collision_rate': float(np.mean([o == 'collision' for o in outcomes])),
        'timeout_rate': float(np.mean([o == 'timeout' for o in outcomes])),
        'mean_return': float(np.mean(returns)),
        'mean_time_to_goal_s': float(np.mean([t for t, s in zip(times, succ) if s])) if any(succ)
        else None,
        'outcomes': outcomes,
    }
    with open(args.out + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ('success_rate', 'collision_rate', 'timeout_rate',
                                               'mean_return')}))
    env.close()


if __name__ == '__main__':
    main()

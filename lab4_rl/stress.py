#!/usr/bin/env python3
"""Prueba de estres: tasa de exito en N episodios inyectando ruido Gaussiano en el LiDAR.
Todas las politicas usan los MISMOS inicios perturbados (yaw aleatorio, +-0.5 m).

  python3 -m lab4_rl.stress --actor results/parte_b/dagger_sim.pth --label DAgger \
      --noises 0 0.05 0.1 --episodes 10 --out results/parte_c/stress
"""
import argparse
import csv
import json
import os

import numpy as np

from lab4_rl.agents import load_actor
from lab4_rl.env import TB3ContinuousEnv, norm_to_cmd
from lab4_rl.parte_b import perturbed_starts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--actor', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--noises', type=float, nargs='+', default=[0.0, 0.05, 0.1])
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--out', default='results/parte_c/stress')
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    os.makedirs(args.out, exist_ok=True)

    actor, _ = load_actor(args.actor)
    env = TB3ContinuousEnv(seed=5, node_name='lab4_stress')
    starts = perturbed_starts(args.episodes, seed=777)
    summary = {'label': args.label, 'actor': args.actor, 'episodes': args.episodes, 'results': {}}
    f = open(os.path.join(args.out, f'{args.label}_steps.csv'), 'w', newline='')
    w = csv.writer(f)
    w.writerow(['noise', 'episode', 't', 'x', 'y', 'v', 'w', 'min_scan', 'outcome'])
    for noise in args.noises:
        env.lidar_noise_std = noise
        outs, dws = [], []
        for ep, st in enumerate(starts):
            obs, info = env.reset(options=st)
            done, ws = False, []
            while not done:
                u, _ = actor.act(obs, deterministic=True)
                v, om = norm_to_cmd(u)
                obs, r, term, trunc, info = env.step([v, om])
                ws.append(om)
                w.writerow([noise, ep, round(info['t'], 3), round(info['x'], 4), round(info['y'], 4),
                            round(v, 4), round(om, 4), round(info['min_scan'], 4), info['outcome']])
                done = term or trunc
            outs.append(info['outcome'])
            dws.append(float(np.abs(np.diff(ws)).mean()) if len(ws) > 1 else 0.0)
        res = {'success_rate': float(np.mean([o == 'goal' for o in outs])),
               'collision_rate': float(np.mean([o == 'collision' for o in outs])),
               'timeout_rate': float(np.mean([o == 'timeout' for o in outs])),
               'mean_abs_dw': float(np.mean(dws)), 'outcomes': outs}
        summary['results'][str(noise)] = res
        print(f'[stress {args.label}] noise={noise}: exito {res["success_rate"]:.0%} '
              f'choque {res["collision_rate"]:.0%} |dw| {res["mean_abs_dw"]:.2f}', flush=True)
    f.close()
    with open(os.path.join(args.out, f'{args.label}_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    env.close()


if __name__ == '__main__':
    main()

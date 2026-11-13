#!/usr/bin/env python3
"""Entrena el actor Gaussiano con A2C o PPO en Gazebo.

  python3 -m lab4_rl.train --algo ppo --total-steps 150000 --out results/ppo

Salidas en --out:
  episodes.csv       una fila por episodio (retorno, largo, resultado, ...)
  updates.csv        perdidas por actualizacion
  actor_best.pth     mejor actor segun la media movil (20 ep.) del retorno
  actor_last.pth / checkpoint_last.pt
"""
import argparse
import csv
import json
import os
import time
from collections import deque

import numpy as np
import torch

from lab4_rl.agents import A2C, PPO, Critic, GaussianActor, RolloutBuffer, save_actor
from lab4_rl.env import OBS_DIM, TB3ContinuousEnv, norm_to_cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['a2c', 'ppo'], required=True)
    p.add_argument('--total-steps', type=int, default=150_000)
    p.add_argument('--max-hours', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--reward-scale', type=float, default=0.1)
    p.add_argument('--n-steps', type=int, default=None)
    args = p.parse_args()

    out = args.out or os.path.join('results', args.algo)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)

    actor, critic = GaussianActor(OBS_DIM, 2), Critic(OBS_DIM)
    if args.algo == 'a2c':
        algo = A2C(actor, critic, lr=args.lr, n_steps=args.n_steps or 32)
    else:
        algo = PPO(actor, critic, lr=args.lr, n_steps=args.n_steps or 1024)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'hyper': {k: v for k, v in vars(algo).items()
                                           if isinstance(v, (int, float))}}, f, indent=2)

    env = TB3ContinuousEnv(seed=args.seed, node_name=f'lab4_train_{args.algo}')
    ep_f = open(os.path.join(out, 'episodes.csv'), 'w', newline='')
    ep_w = csv.writer(ep_f)
    ep_w.writerow(['episode', 'global_step', 'return', 'length', 'outcome', 'final_dist',
                   'mean_abs_w', 'wall_time_s'])
    up_f = open(os.path.join(out, 'updates.csv'), 'w', newline='')
    up_w = None

    t_start = time.time()
    obs, _ = env.reset()
    ep, ep_ret, ep_len, ep_w_abs = 0, 0.0, 0, 0.0
    recent = deque(maxlen=20)
    recent_goal = deque(maxlen=20)
    best = -np.inf
    global_step, n_updates = 0, 0

    while global_step < args.total_steps and time.time() - t_start < args.max_hours * 3600:
        buf = RolloutBuffer()
        for _ in range(algo.n_steps):
            u, logp = actor.act(obs)
            with torch.no_grad():
                val = critic(torch.as_tensor(obs).unsqueeze(0)).item()
            v, w = norm_to_cmd(u)
            obs2, r, term, trunc, info = env.step([v, w])
            r_s = r * args.reward_scale
            if trunc and not term:  # bootstrap en truncamiento por tiempo
                with torch.no_grad():
                    r_s += algo.gamma * critic(torch.as_tensor(obs2).unsqueeze(0)).item()
            buf.add(obs, u, logp, r_s, float(term or trunc), val)
            obs = obs2
            ep_ret += r
            ep_len += 1
            ep_w_abs += abs(w)
            global_step += 1

            if term or trunc:
                ep += 1
                recent.append(ep_ret)
                recent_goal.append(info['outcome'] == 'goal')
                ep_w.writerow([ep, global_step, round(ep_ret, 3), ep_len, info['outcome'],
                               round(info['dist'], 3), round(ep_w_abs / ep_len, 3),
                               round(time.time() - t_start, 1)])
                ep_f.flush()
                ma = float(np.mean(recent))
                if len(recent) >= 10 and ma > best:
                    best = ma
                    save_actor(actor, os.path.join(out, 'actor_best.pth'), algo=args.algo,
                               episode=ep, global_step=global_step, ma20_return=ma,
                               success_rate20=float(np.mean(recent_goal)))
                if ep % 10 == 0:
                    print(f'[{args.algo}] ep {ep:5d} step {global_step:7d} ret {ep_ret:8.1f} '
                          f'ma20 {ma:8.1f} succ20 {np.mean(recent_goal):.2f} '
                          f'{info["outcome"]:9s} {(time.time() - t_start) / 60:6.1f} min',
                          flush=True)
                obs, _ = env.reset()
                ep_ret, ep_len, ep_w_abs = 0.0, 0, 0.0
            if global_step >= args.total_steps:
                break

        with torch.no_grad():
            last_val = critic(torch.as_tensor(obs).unsqueeze(0)).item()
        env.pause()
        stats = algo.update(buf, last_val)
        env.unpause()
        n_updates += 1
        stats.update({'update': n_updates, 'global_step': global_step,
                      'std_v': actor.log_std.exp()[0].item(),
                      'std_w': actor.log_std.exp()[1].item()})
        if up_w is None:
            up_w = csv.DictWriter(up_f, fieldnames=list(stats.keys()))
            up_w.writeheader()
        up_w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()})
        up_f.flush()
        if n_updates % 20 == 0 or args.algo == 'ppo':
            save_actor(actor, os.path.join(out, 'actor_last.pth'), algo=args.algo,
                       episode=ep, global_step=global_step)
            torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(),
                        'opt': algo.opt.state_dict(), 'episode': ep, 'global_step': global_step},
                       os.path.join(out, 'checkpoint_last.pt'))

    save_actor(actor, os.path.join(out, 'actor_last.pth'), algo=args.algo, episode=ep,
               global_step=global_step)
    print(f'[{args.algo}] done: {ep} episodes, {global_step} steps, best ma20 {best:.1f}, '
          f'{(time.time() - t_start) / 60:.1f} min', flush=True)
    ep_f.close()
    up_f.close()
    env.close()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Parte C: extension de DAgger con modelo de recompensa Bradley-Terry + ajuste PPO con penalizacion KL.

Etapas (subcomandos):
  pairs  : genera 15-20 pares de trayectorias cortas (tau1, tau2, 60 pasos = 12 s) con el agente DAgger desde el MISMO
           inicio y las etiqueta (tau_w > tau_l) segun "evasion mas fluida y segura".
           --labeler auto        -> etiquetador simulado con el criterio explicito (ver score_traj)
           --labeler interactive -> el humano ve las metricas/figura de cada par y elige 1 o 2
  rm     : entrena r_psi(s, a) minimizando L_RM = -ln sigma(R(tau_w) - R(tau_l)),  R(tau) = sum_t r_psi
  rlhf   : PPO desde pi_DAgger maximizando R_RLHF = r_psi(s,a) - beta * ln(pi_theta(a|s) / pi_DAgger(a|s))

  python3 -m lab4_rl.parte_c pairs --actor results/parte_b/dagger_sim.pth --out results/parte_c
  python3 -m lab4_rl.parte_c rm    --out results/parte_c
  python3 -m lab4_rl.parte_c rlhf  --actor results/parte_b/dagger_sim.pth --out results/parte_c --beta 0.2
"""
import argparse
import copy
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from lab4_rl.agents import PPO, Critic, RolloutBuffer, load_actor, save_actor
from lab4_rl.env import OBS_DIM, TB3ContinuousEnv, norm_to_cmd

PERTURBED = dict(start_noise=0.5, random_yaw=True)


# ============================================================================ pares
def rollout_short(env, actor, start, steps, sigma):
    """Trayectoria corta muestreando u ~ N(mu, sigma) (sigma controla la 'brusquedad')."""
    obs, info = env.reset(options=start)
    S, U, W, V, XY, scans = [], [], [], [], [], []
    out = 'running'
    for _ in range(steps):
        with torch.no_grad():
            mu = actor(torch.as_tensor(obs).unsqueeze(0)).mean.squeeze(0).numpy()
        u = np.clip(mu + np.random.normal(0, sigma, 2), -1, 1).astype(np.float32)
        v, w = norm_to_cmd(u)
        S.append(obs)
        U.append(u)
        obs, r, term, trunc, info = env.step([v, w])
        V.append(v)
        W.append(w)
        XY.append((info['x'], info['y']))
        scans.append(info['min_scan'])
        out = info['outcome']
        if term or trunc:
            break
    return {'obs': np.array(S), 'act': np.array(U), 'v': np.array(V), 'w': np.array(W),
            'xy': np.array(XY), 'min_scan': np.array(scans), 'outcome': out, 'sigma': sigma,
            'progress': float(np.hypot(*(env.goal - np.array(XY[0]))) - info['dist'])}


def score_traj(tr):
    """Criterio del etiquetador simulado: seguro (sin choque, lejos de obstaculos) y fluido
    (poco w^2 y poco cambio de w), avanzando hacia la meta."""
    dw = np.diff(np.concatenate([[0.0], tr['w']]))
    return (-100.0 * (tr['outcome'] == 'collision')
            + 10.0 * min(tr['min_scan'].min(), 0.6)
            - 1.0 * np.mean(dw ** 2)
            - 0.3 * np.mean(tr['w'] ** 2)
            + 5.0 * tr['progress'])


def cmd_pairs(args):
    actor, _ = load_actor(args.actor)
    env = TB3ContinuousEnv(seed=args.seed, node_name='lab4_pairs', **PERTURBED)
    rng = np.random.default_rng(args.seed)
    sigmas = [0.05, 0.25, 0.5, 0.8]
    pairs = []
    for k in range(args.n_pairs):
        start = dict(x=float(rng.uniform(-0.5, 0.5)), y=float(rng.uniform(-0.5, 0.5)),
                     yaw=float(rng.uniform(-math.pi, math.pi)))
        s1, s2 = rng.choice(sigmas, 2, replace=False)
        t1 = rollout_short(env, actor, start, args.steps, s1)
        t2 = rollout_short(env, actor, start, args.steps, s2)
        if args.labeler == 'interactive':
            for j, t in ((1, t1), (2, t2)):
                print(f'  tau{j}: {t["outcome"]:9s} min_dist={t["min_scan"].min():.2f} m  '
                      f'mean|dw|={np.abs(np.diff(t["w"])).mean():.2f}  avance={t["progress"]:.2f} m')
            pref = int(input(f'par {k + 1}: cual es mas fluida y segura? [1/2]: ').strip() or '1')
        else:
            pref = 1 if score_traj(t1) >= score_traj(t2) else 2
        win, lose = (t1, t2) if pref == 1 else (t2, t1)
        pairs.append((win, lose))
        print(f'[pairs] {k + 1}/{args.n_pairs} w: sigma={win["sigma"]} {win["outcome"]} '
              f'score={score_traj(win):.1f} | l: sigma={lose["sigma"]} {lose["outcome"]} '
              f'score={score_traj(lose):.1f}', flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save(pairs, os.path.join(args.out, 'pairs.pt'))
    env.close()


# ============================================================================ reward model
class RewardModel(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sd', torch.ones(1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1)).squeeze(-1)

    def normalized(self, obs, act):
        return (self.forward(obs, act) - self.mu) / self.sd


def cmd_rm(args):
    pairs = torch.load(os.path.join(args.out, 'pairs.pt'), weights_only=False)
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    data = [((T(w['obs']), T(w['act'])), (T(l['obs']), T(l['act']))) for w, l in pairs]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(data))
    n_val = max(2, len(data) // 5)
    val, tr = [data[i] for i in idx[:n_val]], [data[i] for i in idx[n_val:]]
    torch.manual_seed(args.seed)
    rm = RewardModel()
    opt = torch.optim.Adam(rm.parameters(), lr=1e-3, weight_decay=1e-3)

    def loss_acc(batch):
        L, acc = 0.0, 0.0
        for (ow, aw), (ol, al) in batch:
            diff = rm(ow, aw).sum() - rm(ol, al).sum()
            L = L - torch.nn.functional.logsigmoid(diff)
            acc += float(diff.item() > 0)
        return L / len(batch), acc / len(batch)

    hist = []
    for ep in range(args.rm_epochs):
        loss, acc = loss_acc(tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            vl, va = loss_acc(val)
        hist.append((ep, loss.item(), acc, vl.item(), va))
    # normalizacion de r_psi sobre todos los (s, a) vistos
    with torch.no_grad():
        allr = torch.cat([rm(o, a) for p in data for (o, a) in p])
        rm.mu.fill_(allr.mean().item())
        rm.sd.fill_(allr.std().item() + 1e-6)
    torch.save(rm.state_dict(), os.path.join(args.out, 'reward_model.pth'))
    with open(os.path.join(args.out, 'rm_loss.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([('epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc')] + hist)
    print(f'[rm] {len(tr)} train / {len(val)} val pares | final train L={hist[-1][1]:.3f} '
          f'acc={hist[-1][2]:.2f} | val L={hist[-1][3]:.3f} acc={hist[-1][4]:.2f}', flush=True)


# ============================================================================ RLHF (PPO + KL)
def cmd_rlhf(args):
    actor, _ = load_actor(args.actor)
    actor.train()
    with torch.no_grad():  # misma sigma de exploracion para pi_theta y la referencia pi_DAgger
        actor.log_std.fill_(math.log(args.sigma))
    ref = copy.deepcopy(actor).eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)
    rm = RewardModel()
    rm.load_state_dict(torch.load(os.path.join(args.out, 'reward_model.pth')))
    rm.eval()
    critic = Critic(OBS_DIM)
    algo = PPO(actor, critic, lr=args.lr, n_steps=512, epochs=5, minibatch=64, clip_eps=0.2,
               ent_coef=0.0)
    tag = f'beta{args.beta:g}'
    out = os.path.join(args.out, f'rlhf_{tag}')
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    env = TB3ContinuousEnv(seed=args.seed, node_name=f'lab4_rlhf_{tag.replace(".", "_")}',
                           **PERTURBED)
    ep_f = open(os.path.join(out, 'episodes.csv'), 'w', newline='')
    ep_w = csv.writer(ep_f)
    ep_w.writerow(['episode', 'global_step', 'rlhf_return', 'rm_return', 'kl_sum', 'env_return',
                   'length', 'outcome', 'mean_abs_dw', 'mean_v'])
    obs, _ = env.reset()
    ep, step, t0 = 0, 0, time.time()
    acc = dict(rlhf=0.0, rm=0.0, kl=0.0, env=0.0, n=0, dw=0.0, v=0.0, w_prev=0.0)
    while step < args.total_steps:
        buf = RolloutBuffer()
        for _ in range(algo.n_steps):
            ot = torch.as_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = actor(ot)
                u = dist.sample()
                logp = dist.log_prob(u).sum(-1).item()
                logp_ref = ref(ot).log_prob(u).sum(-1).item()
                val = critic(ot).item()
                r_psi = rm.normalized(ot, u.clamp(-1, 1)).item()
            kl = logp - logp_ref
            r = r_psi - args.beta * kl
            v, w = norm_to_cmd(u.squeeze(0).numpy())
            obs2, r_env, term, trunc, info = env.step([v, w])
            r_train = r
            if trunc and not term:
                with torch.no_grad():
                    r_train += algo.gamma * critic(torch.as_tensor(obs2).unsqueeze(0)).item()
            buf.add(obs, u.squeeze(0).numpy(), logp, r_train, float(term or trunc), val)
            obs = obs2
            step += 1
            acc['rlhf'] += r
            acc['rm'] += r_psi
            acc['kl'] += kl
            acc['env'] += r_env
            acc['dw'] += abs(w - acc['w_prev'])
            acc['v'] += v
            acc['w_prev'] = w
            acc['n'] += 1
            if term or trunc:
                ep += 1
                n = acc['n']
                ep_w.writerow([ep, step, round(acc['rlhf'], 3), round(acc['rm'], 3),
                               round(acc['kl'], 3), round(acc['env'], 3), n, info['outcome'],
                               round(acc['dw'] / n, 4), round(acc['v'] / n, 4)])
                ep_f.flush()
                if ep % 10 == 0:
                    print(f'[rlhf {tag}] ep {ep} step {step} R_rlhf {acc["rlhf"]:.1f} '
                          f'KL {acc["kl"]:.1f} {info["outcome"]} {(time.time() - t0) / 60:.1f} min',
                          flush=True)
                acc = dict(rlhf=0.0, rm=0.0, kl=0.0, env=0.0, n=0, dw=0.0, v=0.0, w_prev=0.0)
                obs, _ = env.reset()
            if step >= args.total_steps:
                break
        with torch.no_grad():
            last_val = critic(torch.as_tensor(obs).unsqueeze(0)).item()
        env.pause()
        algo.update(buf, last_val)
        env.unpause()
        save_actor(actor, os.path.join(out, 'actor_rlhf.pth'), algo='rlhf', beta=args.beta,
                   step=step)
    save_actor(actor, os.path.join(out, 'actor_rlhf.pth'), algo='rlhf', beta=args.beta, step=step)
    ep_f.close()
    print(f'[rlhf {tag}] done {step} steps {(time.time() - t0) / 60:.1f} min', flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['pairs', 'rm', 'rlhf'])
    p.add_argument('--actor', default='results/parte_b/dagger_sim.pth')
    p.add_argument('--out', default='results/parte_c')
    p.add_argument('--n-pairs', type=int, default=20)
    p.add_argument('--steps', type=int, default=60, help='largo de cada trayectoria corta')
    p.add_argument('--sigma', type=float, default=0.1, help='sigma inicial de pi_theta y pi_DAgger')
    p.add_argument('--labeler', choices=['auto', 'interactive'], default='auto')
    p.add_argument('--rm-epochs', type=int, default=300)
    p.add_argument('--beta', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--total-steps', type=int, default=40000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    {'pairs': cmd_pairs, 'rm': cmd_rm, 'rlhf': cmd_rlhf}[args.stage](args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Parte B: Behavior Cloning (linea base) vs DAgger con intervencion.

  python3 -m lab4_rl.parte_b --out results/parte_b                 # experto oraculo (automatico)
  python3 -m lab4_rl.parte_b --out results/parte_b_humano --expert keyboard   # humano con teclado
      (para modo teclado usar GUI=true REALTIME=true en run_sim.sh)

Flujo
  1. D0: trayectorias del experto en condiciones "faciles" (inicio ~ origen, mirando a la meta).
  2. pi_BC = regresion supervisada (MSE sobre la media del actor Gaussiano) sobre D0   -> bc.pth
  3. Para cada politica pi_i (i = 0 es BC):
       a) despliegue CON supervision en condiciones perturbadas (yaw aleatorio, +-0.5 m): el experto
          interviene cuando el robot va a chocar (o se queda dando vueltas sin avanzar); se guardan los
          pares (s_peligro, a_experto) = D_{i+1} y la tasa de intervencion;
       b) evaluacion SIN supervision (mismos 15 inicios perturbados para todas las politicas):
          tasa de exito / choque y trayectorias (x, y) de /odom;
       c) D <- D U D_{i+1};  fine-tuning in situ de la red  -> dagger_round{i+1}.pth
  4. La ultima ronda se guarda como dagger_sim.pth
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from lab4_rl.agents import GaussianActor, save_actor
from lab4_rl.env import OBS_DIM, TB3ContinuousEnv, cmd_to_norm, norm_to_cmd
from lab4_rl.expert import KeyboardTeleop, OracleExpert

EASY = dict(start_noise=0.05, yaw_noise=0.3)
PERTURBED_NOISE = 0.5


def perturbed_starts(n, seed):
    rng = np.random.default_rng(seed)
    return [dict(x=float(rng.uniform(-PERTURBED_NOISE, PERTURBED_NOISE)),
                 y=float(rng.uniform(-PERTURBED_NOISE, PERTURBED_NOISE)),
                 yaw=float(rng.uniform(-math.pi, math.pi))) for _ in range(n)]


def policy_action(actor, obs):
    u, _ = actor.act(obs, deterministic=True)
    return norm_to_cmd(u)


def run_episode(env, actor=None, expert=None, supervisor=None, start=None, stuck_steps=25):
    """Ejecuta un episodio. actor=None -> controla el experto todo el tiempo (recoleccion de D0).
    supervisor='oracle'|'keyboard' -> el agente pilotea y el supervisor puede intervenir."""
    obs, info = env.reset(options=start)
    rows, data = [], []
    done, n_int, hold = False, 0, 0
    dist_hist = [info['dist']]
    while not done:
        if actor is None:
            v, w = expert.act(env)
            data.append((obs, cmd_to_norm(v, w)))
            intervened = True
        else:
            v, w = policy_action(actor, obs)
            intervened = False
            if supervisor == 'oracle':
                stuck = (len(dist_hist) > stuck_steps
                         and dist_hist[-stuck_steps] - dist_hist[-1] < 0.05)
                if hold > 0 or expert.danger(env, (v, w)) or stuck:
                    hold = hold - 1 if hold > 0 else (10 if stuck else 5)
                    intervened = True
            elif supervisor == 'keyboard':
                intervened = expert.poll()
            if intervened:
                v, w = expert.act(env) if supervisor == 'oracle' else expert.action()
                data.append((obs, cmd_to_norm(v, w)))
                n_int += 1
        obs, r, term, trunc, info = env.step([v, w])
        dist_hist.append(info['dist'])
        rows.append((info['t'], info['x'], info['y'], v, w, int(intervened), info['outcome']))
        done = term or trunc
        if supervisor == 'keyboard' and expert.quit:
            break
    return rows, data, info['outcome'], n_int


def train_bc(actor, data, epochs, lr, log_std=-1.4, seed=0):
    """Regresion supervisada de la media del actor sobre las acciones del experto."""
    torch.manual_seed(seed)
    X = torch.as_tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    Y = torch.as_tensor(np.clip(np.array([d[1] for d in data]), -0.98, 0.98), dtype=torch.float32)
    n = len(X)
    idx = torch.randperm(n)
    n_val = max(1, n // 10)
    val, tr = idx[:n_val], idx[n_val:]
    opt = torch.optim.Adam(actor.mu_net.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = ((torch.tanh(actor.mu_net(X[b])) - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        with torch.no_grad():
            vl = ((torch.tanh(actor.mu_net(X[val])) - Y[val]) ** 2).mean().item()
        hist.append((ep, tot / len(tr), vl))
    with torch.no_grad():
        actor.log_std.fill_(log_std)
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/parte_b')
    p.add_argument('--expert', choices=['oracle', 'keyboard'], default='oracle')
    p.add_argument('--d0-episodes', type=int, default=30)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--supervised-episodes', type=int, default=15)
    p.add_argument('--eval-episodes', type=int, default=15)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    env = TB3ContinuousEnv(seed=args.seed, node_name='lab4_parte_b', **EASY)
    env._spin_until(lambda: env.obstacles_yaw is not None, 30)
    oracle = OracleExpert()
    sup = KeyboardTeleop() if args.expert == 'keyboard' else oracle

    # ---------------------------------------------------------------- 1. D0 (experto)
    D, d0_out = [], []
    for ep in range(args.d0_episodes):
        if args.expert == 'oracle':
            _, data, out, _ = run_episode(env, actor=None, expert=oracle)
        else:  # el humano maneja todo el episodio
            sup.active = True
            dummy = GaussianActor(OBS_DIM, 2)
            _, data, out, _ = run_episode(env, actor=dummy, expert=sup, supervisor='keyboard')
        D += data
        d0_out.append(out)
        print(f'[D0] ep {ep + 1}/{args.d0_episodes} {out} |D|={len(D)}', flush=True)
    print(f'[D0] exito experto {np.mean([o == "goal" for o in d0_out]):.2f}', flush=True)

    # ---------------------------------------------------------------- 2. BC
    actor = GaussianActor(OBS_DIM, 2)
    hist = train_bc(actor, D, epochs=200, lr=1e-3, seed=args.seed)
    print(f'[BC] train mse {hist[-1][1]:.4f} val mse {hist[-1][2]:.4f}', flush=True)
    save_actor(actor, os.path.join(args.out, 'bc.pth'), algo='bc', n_data=len(D))
    with open(os.path.join(args.out, 'bc_loss.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([('epoch', 'train_mse', 'val_mse')] + hist)

    # ---------------------------------------------------------------- 3. rondas
    env.start_noise, env.yaw_noise, env.random_yaw = PERTURBED_NOISE, None, True
    eval_starts = perturbed_starts(args.eval_episodes, seed=2026)
    rounds = []
    traj_f = open(os.path.join(args.out, 'trajectories.csv'), 'w', newline='')
    traj_w = csv.writer(traj_f)
    traj_w.writerow(['policy', 'phase', 'episode', 't', 'x', 'y', 'v', 'w', 'intervened', 'outcome'])
    for i in range(args.rounds + 1):
        name = 'BC' if i == 0 else f'DAgger-{i}'
        # a) despliegue supervisado
        new, outs, n_steps, n_int = [], [], 0, 0
        for ep in range(args.supervised_episodes):
            rows, data, out, k = run_episode(env, actor, sup, supervisor=args.expert)
            new += data
            outs.append(out)
            n_steps += len(rows)
            n_int += k
            for r in rows:
                traj_w.writerow([name, 'supervised', ep] + [round(x, 4) if isinstance(x, float)
                                                             else x for x in r])
        int_rate = n_int / max(n_steps, 1)
        # b) evaluacion sin supervision (mismos inicios para todas las politicas)
        ev = []
        for ep, st in enumerate(eval_starts):
            rows, _, out, _ = run_episode(env, actor, None, supervisor=None, start=st)
            ev.append(out)
            for r in rows:
                traj_w.writerow([name, 'eval', ep] + [round(x, 4) if isinstance(x, float)
                                                       else x for x in r])
        traj_f.flush()
        row = {'round': i, 'policy': name, 'dataset_size': len(D),
               'intervention_rate': round(int_rate, 4),
               'supervised_collision_rate': round(float(np.mean([o == 'collision' for o in outs])), 4),
               'eval_success_rate': round(float(np.mean([o == 'goal' for o in ev])), 4),
               'eval_collision_rate': round(float(np.mean([o == 'collision' for o in ev])), 4),
               'eval_timeout_rate': round(float(np.mean([o == 'timeout' for o in ev])), 4),
               'new_pairs': len(new), 'minutes': round((time.time() - t0) / 60, 1)}
        rounds.append(row)
        print(f'[{name}] {json.dumps(row)}', flush=True)
        if i == args.rounds:
            break
        # c) agregacion + fine-tuning in situ (simulacion en pausa)
        D += new
        env.pause()
        train_bc(actor, D, epochs=80, lr=5e-4, seed=args.seed + i + 1)
        env.unpause()
        save_actor(actor, os.path.join(args.out, f'dagger_round{i + 1}.pth'), algo='dagger',
                   round=i + 1, n_data=len(D))
    traj_f.close()
    save_actor(actor, os.path.join(args.out, 'dagger_sim.pth'), algo='dagger', round=args.rounds,
               n_data=len(D))
    with open(os.path.join(args.out, 'rounds.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rounds[0].keys()))
        w.writeheader()
        w.writerows(rounds)
    np.savez(os.path.join(args.out, 'dataset_D.npz'), obs=np.array([d[0] for d in D]),
             act=np.array([d[1] for d in D]))
    if args.expert == 'keyboard':
        sup.close()
    print(f'[parte_b] done in {(time.time() - t0) / 60:.1f} min', flush=True)
    env.close()


if __name__ == '__main__':
    main()

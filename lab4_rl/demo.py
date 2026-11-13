#!/usr/bin/env python3
"""Secuencia de demostracion para el VIDEO (Fase 2), en una sola sesion de Gazebo en tiempo real.

  GUI=true REALTIME=true scripts/run_sim.sh 30 results/demo.log python3 -m lab4_rl.demo

Segmentos (con titulos grandes en la terminal):
  1. Parte A  - A2C (actor_base.pth) vs PPO
  2. Parte B  - BC con perturbacion inicial (falla por distribution shift)
                DAgger: el agente pilotea y el experto interviene (se marca en la terminal)
  3. Parte C  - RLHF beta=1 (navegacion suavizada) vs beta=0 (reward hacking)
"""
import argparse
import math
import os

from lab4_rl.agents import load_actor
from lab4_rl.env import TB3ContinuousEnv, norm_to_cmd
from lab4_rl.expert import OracleExpert


def banner(txt):
    print('\n' + '=' * 70 + f'\n   {txt}\n' + '=' * 70, flush=True)


def episode(env, actor, start, label, expert=None, max_steps=None):
    obs, info = env.reset(options=start)
    done, k, hold = False, 0, 0
    while not done:
        u, _ = actor.act(obs, deterministic=True)
        v, w = norm_to_cmd(u)
        tag = ''
        if expert is not None and (hold > 0 or expert.danger(env, (v, w))):
            hold = hold - 1 if hold > 0 else 5
            v, w = expert.act(env)
            tag = '   <<< EXPERTO INTERVIENE (DAgger) >>>'
        obs, r, term, trunc, info = env.step([v, w])
        k += 1
        if k % 5 == 0 or tag:
            print(f'  [{label}] t={info["t"]:5.1f}s v={v:.2f} m/s w={w:+.2f} rad/s '
                  f'd_meta={info["dist"]:.2f} m{tag}', flush=True)
        done = term or trunc or (max_steps is not None and k >= max_steps)
    res = info['outcome'].upper() if info['outcome'] != 'running' else 'NO LLEGA (corte de la demo)'
    print(f'  [{label}] RESULTADO: {res}  ({info["t"]:.1f} s)', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=2)
    args = p.parse_args()
    R = 'results'
    env = TB3ContinuousEnv(seed=11, node_name='lab4_demo', start_noise=0.5)
    env._spin_until(lambda: env.obstacles_yaw is not None, 30)
    starts = [dict(x=0.0, y=0.0, yaw=math.pi / 4), dict(x=-0.3, y=0.2, yaw=-2.5),
              dict(x=0.3, y=-0.3, yaw=2.0)]
    # (titulo, actor, experto que supervisa, inicios, max_pasos por episodio)
    segs = [
        ('PARTE A - A2C (actor_base.pth)', f'{R}/actor_base.pth', None, starts[:args.episodes], None),
        ('PARTE A - PPO (optimo local conservador: no cruza el anillo)',
         f'{R}/parte_a/ppo_s1/actor_best.pth', None, starts[:1], 100),
        ('PARTE B - Behavior Cloning con perturbacion inicial (distribution shift)',
         f'{R}/parte_b/bc.pth', None, starts[1:3], None),
        ('PARTE B - DAgger: el agente pilotea y el experto corrige', f'{R}/parte_b/bc.pth',
         OracleExpert(), starts[1:2], None),
        ('PARTE B - DAgger final (dagger_sim.pth) sin ayuda', f'{R}/parte_b/dagger_sim.pth', None,
         starts[1:3], None),
        ('PARTE C - RLHF beta=1 (penalizacion KL activa: navegacion suavizada)',
         f'{R}/parte_c/rlhf_beta1/actor_rlhf.pth', None, starts[:args.episodes], None),
        ('PARTE C - RLHF beta=0 (sin KL -> reward hacking)', f'{R}/parte_c/rlhf_beta0/actor_rlhf.pth',
         None, starts[:1], None),
    ]
    for title, path, expert, sts, max_steps in segs:
        if not os.path.exists(path):
            print(f'(saltando {title}: no existe {path})')
            continue
        actor, _ = load_actor(path)
        banner(title)
        for i, st in enumerate(sts):
            episode(env, actor, st, f'ep{i + 1}', expert=expert, max_steps=max_steps)
    banner('FIN DE LA DEMOSTRACION')
    env.close()


if __name__ == '__main__':
    main()

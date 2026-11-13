#!/usr/bin/env python3
"""Graficas de la Parte A (A2C vs PPO).

  python3 -m lab4_rl.plot_parte_a --runs results/parte_a --evals results/parte_a/eval --out results/parte_a/figs

(1) fig1_recompensa_a2c_vs_ppo.png : retorno por episodio (media movil 20) y tasa de exito vs pasos
(2) fig2_perfil_v_w.png            : v(t) y w(t) en un episodio de prueba (A2C vs PPO)
(3) fig3_trayectorias.png          : trayectorias (x, y) de evaluacion sobre el mapa
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {'a2c': '#d62728', 'ppo': '#1f77b4'}
GOAL = (1.75, 1.75)


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def moving(x, k=20):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - k + 1):i + 1].mean()
    return out


def fig_rewards(runs_dir, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    table = []
    for run in sorted(glob.glob(os.path.join(runs_dir, '*_s*'))):
        f = os.path.join(run, 'episodes.csv')
        if not os.path.exists(f):
            continue
        rows = read_csv(f)
        if not rows:
            continue
        name = os.path.basename(run)
        algo = name.split('_')[0]
        ep = np.array([int(r['episode']) for r in rows])
        step = np.array([int(r['global_step']) for r in rows])
        ret = np.array([float(r['return']) for r in rows])
        goal = np.array([r['outcome'] == 'goal' for r in rows], dtype=float)
        coll = np.array([r['outcome'] == 'collision' for r in rows], dtype=float)
        ls = '-' if name.endswith('s0') else '--'
        axes[0].plot(ep, ret, color=COLORS[algo], alpha=0.12, lw=0.6)
        axes[0].plot(ep, moving(ret), color=COLORS[algo], ls=ls, lw=1.8, label=name.upper())
        axes[1].plot(step / 1000, moving(ret), color=COLORS[algo], ls=ls, lw=1.8, label=name.upper())
        axes[2].plot(step / 1000, moving(goal) * 100, color=COLORS[algo], ls=ls, lw=1.8,
                     label=name.upper())
        last = slice(-50, None)
        table.append({'run': name, 'episodes': int(ep[-1]), 'steps': int(step[-1]),
                      'ret_last50': float(ret[last].mean()),
                      'ret_last50_std': float(ret[last].std()),
                      'success_last50': float(goal[last].mean()),
                      'collision_last50': float(coll[last].mean()),
                      'best_ma20': float(moving(ret).max())})
    axes[0].set(xlabel='Episodio', ylabel='Recompensa acumulada (episodio)',
                title='Retorno por episodio (linea: media movil 20)')
    axes[1].set(xlabel='Pasos de entorno (miles)', ylabel='Retorno (media movil 20)',
                title='Retorno vs. pasos de interaccion')
    axes[2].set(xlabel='Pasos de entorno (miles)', ylabel='% episodios que llegan a la meta',
                title='Tasa de exito (media movil 20)', ylim=(-2, 102))
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle('Parte A - A2C vs PPO (actor Gaussiano, control continuo v, w)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig1_recompensa_a2c_vs_ppo.png'), dpi=150)
    plt.close(fig)
    return table


def load_eval(prefix):
    rows = read_csv(prefix + '_steps.csv')
    with open(prefix + '_summary.json') as f:
        summ = json.load(f)
    return rows, summ


def fig_velocities(evals, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for label, prefix in evals:
        rows, summ = load_eval(prefix)
        algo = label.split()[0].lower()
        # primer episodio exitoso (si lo hay) para el perfil temporal
        eps = sorted({int(r['episode']) for r in rows})
        pick = next((e for e in eps if any(r['episode'] == str(e) and r['outcome'] == 'goal'
                                           for r in rows)), eps[0])
        ep_rows = [r for r in rows if int(r['episode']) == pick and r['outcome'] != 'start']
        t = np.array([float(r['t']) for r in ep_rows])
        v = np.array([float(r['v']) for r in ep_rows])
        w = np.array([float(r['w']) for r in ep_rows])
        dw = np.abs(np.diff(w)).mean() if len(w) > 1 else 0.0
        lab = f'{label} (ep {pick}, {ep_rows[-1]["outcome"]}, mean|dw|={dw:.2f} rad/s)'
        axes[0].step(t, v, where='post', color=COLORS.get(algo, 'k'), lw=1.6, label=lab)
        axes[1].step(t, w, where='post', color=COLORS.get(algo, 'k'), lw=1.6, label=lab)
    axes[0].axhline(0.22, color='gray', ls=':', lw=1)
    axes[1].axhline(2.84, color='gray', ls=':', lw=1)
    axes[1].axhline(-2.84, color='gray', ls=':', lw=1)
    axes[0].set(ylabel='v(t) [m/s]', title='Perfil de velocidades en un episodio de prueba '
                '(politica determinista: media del actor)')
    axes[1].set(xlabel='t [s] (tiempo simulado)', ylabel='w(t) [rad/s]')
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_perfil_v_w.png'), dpi=150)
    plt.close(fig)


def fig_trajectories(evals, out_dir):
    fig, axes = plt.subplots(1, len(evals), figsize=(5.5 * len(evals), 5.5))
    axes = np.atleast_1d(axes)
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, (label, prefix) in zip(axes, evals):
        rows, summ = load_eval(prefix)
        ax.add_patch(plt.Rectangle((-2.35, -2.35), 4.7, 4.7, fill=False, lw=2, color='saddlebrown'))
        ax.fill_between(np.sqrt(2) * np.cos(th), np.sqrt(2) * np.sin(th), alpha=0)
        for rr in (np.sqrt(2) - 0.15, np.sqrt(2) + 0.15):
            ax.plot(rr * np.cos(th), rr * np.sin(th), color='gray', ls=':', lw=1)
        ax.add_patch(plt.Circle(GOAL, 0.25, color='red', alpha=0.3))
        for e in sorted({int(r['episode']) for r in rows}):
            er = [r for r in rows if int(r['episode']) == e]
            x = [float(r['x']) for r in er]
            y = [float(r['y']) for r in er]
            oc = er[-1]['outcome']
            c = {'goal': 'tab:green', 'collision': 'tab:red'}.get(oc, 'tab:orange')
            ax.plot(x, y, color=c, lw=1.2, alpha=0.8)
            ax.plot(x[0], y[0], 'k.', ms=5)
            if oc == 'collision':
                ax.plot(x[-1], y[-1], 'rx', ms=8)
        ax.set_aspect('equal')
        ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xlabel='x [m]', ylabel='y [m]',
               title=f'{label}: exito {summ["success_rate"] * 100:.0f}% '
                     f'/ choque {summ["collision_rate"] * 100:.0f}%')
        ax.grid(alpha=0.3)
    fig.suptitle('Trayectorias de evaluacion (verde=meta, rojo=choque, naranja=timeout; '
                 'gris: banda de orbita de los cilindros)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_trayectorias.png'), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='results/parte_a')
    p.add_argument('--evals', nargs='*', default=[],
                   help='pares LABEL=prefix, p.ej. "PPO=results/parte_a/eval/ppo"')
    p.add_argument('--out', default='results/parte_a/figs')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    table = fig_rewards(args.runs, args.out)
    with open(os.path.join(args.out, 'training_summary.json'), 'w') as f:
        json.dump(table, f, indent=2)
    for t in table:
        print(t)
    evals = [tuple(e.split('=', 1)) for e in args.evals]
    if evals:
        fig_velocities(evals, args.out)
        fig_trajectories(evals, args.out)
    print('figures in', args.out)


if __name__ == '__main__':
    main()

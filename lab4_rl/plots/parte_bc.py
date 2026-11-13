#!/usr/bin/env python3
"""Graficas de las Partes B y C.

  python3 -m lab4_rl.plots.parte_bc --b results/parte_b --c results/parte_c --out results/graficas

(3) 3_parteB_intervencion_y_colision.png : tasa de intervencion y de colision por ronda (BC -> DAgger)
(4) 4_parteB_trayectorias_bc_vs_dagger.png : mapa 2D (x, y) de /odom, BC vs DAgger (mismos inicios)
(5) 5_parteC_bradley_terry_y_estres.png : curva L_RM (train/val) + tabla de exito con ruido LiDAR
    extra_parteC_rlhf_entrenamiento.png : r_psi y KL durante el ajuste PPO para cada beta
    extra_parteC_suavidad_w.png : w(t) DAgger vs RLHF (beta>0 y beta=0)
    5_parteC_tabla_estres.md
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

GOAL = (1.75, 1.75)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def moving(x, k=20):
    x = np.asarray(x, dtype=float)
    return np.array([x[max(0, i - k + 1):i + 1].mean() for i in range(len(x))])


def draw_map(ax):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.add_patch(plt.Rectangle((-2.35, -2.35), 4.7, 4.7, fill=False, lw=2, color='saddlebrown'))
    for rr in (np.sqrt(2) - 0.15, np.sqrt(2) + 0.15):
        ax.plot(rr * np.cos(th), rr * np.sin(th), color='gray', ls=':', lw=1)
    ax.add_patch(plt.Circle(GOAL, 0.25, color='red', alpha=0.3))
    ax.set_aspect('equal')
    ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xlabel='x [m]', ylabel='y [m]')
    ax.grid(alpha=0.3)


def fig3(bdir, out):
    rows = read_csv(os.path.join(bdir, 'rounds.csv'))
    names = [r['policy'] for r in rows]
    x = np.arange(len(rows))
    ir = [100 * float(r['intervention_rate']) for r in rows]
    cr = [100 * float(r['eval_collision_rate']) for r in rows]
    sc = [100 * float(r['supervised_collision_rate']) for r in rows]
    su = [100 * float(r['eval_success_rate']) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, ir, 'o-', lw=2.2, color='tab:purple', label='Tasa de intervencion (% de pasos con experto)')
    ax.plot(x, cr, 's-', lw=2.2, color='tab:red', label='Tasa de colision sin supervision (%)')
    ax.plot(x, sc, 's--', lw=1.2, color='tab:red', alpha=0.5,
            label='Colisiones aun con supervision (%)')
    ax.plot(x, su, '^-', lw=1.5, color='tab:green', alpha=0.8, label='Tasa de exito sin supervision (%)')
    for xi, a, b in zip(x, ir, cr):
        ax.annotate(f'{a:.0f}%', (xi, a), textcoords='offset points', xytext=(0, 7), ha='center',
                    fontsize=8, color='tab:purple')
        ax.annotate(f'{b:.0f}%', (xi, b), textcoords='offset points', xytext=(0, -13), ha='center',
                    fontsize=8, color='tab:red')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n|D|={r["dataset_size"]}' for n, r in zip(names, rows)])
    ax.set(ylabel='%', ylim=(-5, 105),
           title='Parte B - Intervencion y colision por ronda (BC = ronda 0)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(out, '3_parteB_intervencion_y_colision.png'), dpi=150)
    plt.close(fig)


def fig4(bdir, out):
    rows = [r for r in read_csv(os.path.join(bdir, 'trajectories.csv')) if r['phase'] == 'eval']
    pols = list(dict.fromkeys(r['policy'] for r in rows))
    show = [pols[0], pols[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, pol in zip(axes, show):
        draw_map(ax)
        pr = [r for r in rows if r['policy'] == pol]
        eps = sorted({int(r['episode']) for r in pr})
        n_goal = n_col = 0
        for e in eps:
            er = [r for r in pr if int(r['episode']) == e]
            xs = [float(r['x']) for r in er]
            ys = [float(r['y']) for r in er]
            oc = er[-1]['outcome']
            n_goal += oc == 'goal'
            n_col += oc == 'collision'
            c = {'goal': 'tab:green', 'collision': 'tab:red'}.get(oc, 'tab:orange')
            ax.plot(xs, ys, color=c, lw=1.2, alpha=0.85)
            ax.plot(xs[0], ys[0], 'k.', ms=6)
            if oc == 'collision':
                ax.plot(xs[-1], ys[-1], 'rx', ms=8)
        ax.set_title(f'{pol}: exito {n_goal}/{len(eps)}, choque {n_col}/{len(eps)}')
    fig.suptitle('Parte B - Trayectorias (x, y) de /odom con perturbaciones iniciales '
                 '(mismos 15 inicios; verde=meta, rojo=choque, naranja=desvio/timeout)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, '4_parteB_trayectorias_bc_vs_dagger.png'), dpi=150)
    plt.close(fig)


def stress_table(cdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(cdir, 'stress', '*_summary.json'))):
        with open(f) as fh:
            s = json.load(fh)
        rows.append(s)
    order = {'BC': 0, 'DAgger': 1}
    rows.sort(key=lambda s: (order.get(s['label'], 2), s['label']))
    return rows


def fig5(cdir, out):
    h = read_csv(os.path.join(cdir, 'rm_loss.csv'))
    ep = [int(r['epoch']) for r in h]
    fig = plt.figure(figsize=(13, 4.8))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(ep, [float(r['train_loss']) for r in h], lw=2, label='L_RM train')
    ax.plot(ep, [float(r['val_loss']) for r in h], lw=2, label='L_RM validacion')
    ax.axhline(np.log(2), color='gray', ls=':', lw=1, label='ln 2 (azar)')
    ax2 = ax.twinx()
    ax2.plot(ep, [float(r['train_acc']) for r in h], color='tab:green', alpha=0.4, lw=1,
             label='acc train')
    ax2.plot(ep, [float(r['val_acc']) for r in h], color='tab:olive', alpha=0.4, lw=1,
             label='acc val')
    ax2.set_ylabel('precision de preferencia', color='tab:green')
    ax2.set_ylim(0, 1.05)
    ax.set(xlabel='epoca', ylabel=r'$\mathcal{L}_{RM}(\psi) = -\ln\sigma(\hat R(\tau_w)-\hat R(\tau_l))$',
           title='Modelo de recompensa Bradley-Terry')
    ax.grid(alpha=0.3)
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, fontsize=8)

    rows = stress_table(cdir)
    ax = fig.add_subplot(1, 2, 2)
    ax.axis('off')
    if rows:
        noises = list(rows[0]['results'].keys())
        cell = [[s['label']] + [f'{100 * s["results"][n]["success_rate"]:.0f}% '
                                f'({100 * s["results"][n]["collision_rate"]:.0f}%)' for n in noises]
                for s in rows]
        tb = ax.table(cellText=cell, colLabels=['politica'] + [f'ruido LiDAR\nsigma={n} m' for n in noises],
                      loc='center', cellLoc='center')
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 2.0)
        ax.set_title(f'Success rate en {rows[0]["episodes"]} episodios (entre parentesis: choques)',
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, '5_parteC_bradley_terry_y_estres.png'), dpi=150)
    plt.close(fig)

    if rows:
        noises = list(rows[0]['results'].keys())
        lines = ['| politica | ' + ' | '.join(f'exito (choque) sigma={n} m' for n in noises)
                 + ' | media abs(dw) sin ruido |',
                 '|---|' + '---|' * len(noises) + '---|']
        for s in rows:
            lines.append(f'| {s["label"]} | ' + ' | '.join(
                f'{100 * s["results"][n]["success_rate"]:.0f}% ({100 * s["results"][n]["collision_rate"]:.0f}%)'
                for n in noises) + f' | {s["results"][noises[0]]["mean_abs_dw"]:.2f} |')
        with open(os.path.join(out, '5_parteC_tabla_estres.md'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('\n'.join(lines))


def fig6(cdir, out):
    runs = sorted(glob.glob(os.path.join(cdir, 'rlhf_beta*')))
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for run in runs:
        f = os.path.join(run, 'episodes.csv')
        if not os.path.exists(f):
            continue
        r = read_csv(f)
        lab = os.path.basename(run).replace('rlhf_beta', 'beta=')
        st = np.array([int(x['global_step']) for x in r]) / 1000
        n = np.array([int(x['length']) for x in r])
        axes[0].plot(st, moving(np.array([float(x['rm_return']) for x in r]) / n), lw=1.8, label=lab)
        axes[1].plot(st, moving(np.array([float(x['kl_sum']) for x in r]) / n), lw=1.8, label=lab)
        axes[2].plot(st, 100 * moving([x['outcome'] == 'goal' for x in r]), lw=1.8, label=lab)
    axes[0].set(xlabel='pasos (miles)', ylabel='r_psi medio por paso', title='Recompensa aprendida')
    axes[1].set(xlabel='pasos (miles)', ylabel='ln pi_theta/pi_DAgger medio por paso',
                title='Divergencia con la politica DAgger')
    axes[2].set(xlabel='pasos (miles)', ylabel='% episodios a la meta', title='Exito (media movil 20)',
                ylim=(-2, 102))
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle('Parte C - Ajuste PPO con R_RLHF = r_psi - beta ln(pi_theta/pi_DAgger)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'extra_parteC_rlhf_entrenamiento.png'), dpi=150)
    plt.close(fig)


def fig7(cdir, out):
    files = sorted(glob.glob(os.path.join(cdir, 'stress', '*_steps.csv')))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for f in files:
        lab = os.path.basename(f).replace('_steps.csv', '')
        if lab == 'BC':
            continue
        r = [x for x in read_csv(f) if float(x['noise']) == 0.0]
        eps = sorted({int(x['episode']) for x in r})
        pick = next((e for e in eps if [x for x in r if int(x['episode']) == e][-1]['outcome'] == 'goal'),
                    eps[0])
        er = [x for x in r if int(x['episode']) == pick]
        ax.step([float(x['t']) for x in er], [float(x['w']) for x in er], where='post', lw=1.5,
                label=f'{lab} (ep {pick}, {er[-1]["outcome"]})')
    ax.set(xlabel='t [s]', ylabel='w(t) [rad/s]', title='Suavidad del giro: DAgger vs RLHF')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'extra_parteC_suavidad_w.png'), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--b', default='results/parte_b')
    p.add_argument('--c', default='results/parte_c')
    p.add_argument('--out', default='results/graficas')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.b, 'rounds.csv')):
        fig3(args.b, args.out)
        fig4(args.b, args.out)
    if os.path.exists(os.path.join(args.c, 'rm_loss.csv')):
        fig5(args.c, args.out)
    fig6(args.c, args.out)
    fig7(args.c, args.out)
    print('figures in', args.out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Elige el mejor actor de la Parte A segun la evaluacion determinista y lo copia a actor_base.pth.

Criterio: mayor tasa de exito; empate -> mayor retorno medio.
Tambien imprime la tabla comparativa A2C vs PPO (markdown).
"""
import argparse
import glob
import json
import os
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evals', default='results/parte_a/eval')
    p.add_argument('--out', default='results/actor_base.pth')
    args = p.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.evals, '*_summary.json'))):
        with open(f) as fh:
            s = json.load(fh)
        rows.append((os.path.basename(f).replace('_summary.json', ''), s))
    if not rows:
        raise SystemExit('no evaluations found')

    lines = ['| actor | exito | choque | timeout | retorno medio | t medio a meta (s) |',
             '|---|---|---|---|---|---|']
    for name, s in rows:
        t = s['mean_time_to_goal_s']
        lines.append(f'| {name} | {s["success_rate"] * 100:.0f}% | {s["collision_rate"] * 100:.0f}% '
                     f'| {s["timeout_rate"] * 100:.0f}% | {s["mean_return"]:.1f} '
                     f'| {"-" if t is None else f"{t:.1f}"} |')
    table = '\n'.join(lines)
    print(table)

    best = {}
    for name, s in rows:
        algo = name.split('_')[0]
        key = (s['success_rate'], s['mean_return'])
        if algo not in best or key > best[algo][0]:
            best[algo] = (key, name, s)
    overall = max(best.values(), key=lambda b: b[0])
    _, name, s = overall
    shutil.copy2(s['actor'], args.out)
    info = {'selected': name, 'source': s['actor'], 'success_rate': s['success_rate'],
            'collision_rate': s['collision_rate'], 'mean_return': s['mean_return'],
            'best_per_algo': {a: b[1] for a, b in best.items()}}
    with open(args.out.replace('.pth', '.json'), 'w') as fh:
        json.dump(info, fh, indent=2)
    with open(os.path.join(args.evals, 'tabla_eval.md'), 'w') as fh:
        fh.write(table + '\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Tarea Previa Lab 4 - exporta los pesos de la red DQN entrenada.

Para cada modelN.h5 en saved_model/ genera, en OUT_DIR:
  - modelN.h5              modelo Keras completo (copia)
  - modelN.json            metadatos de entrenamiento (epsilon, step_counter, trained_episodes)
  - modelN.weights.h5      solo pesos (formato Keras 3)
  - modelN_weights.npz     pesos como arrays numpy (portables: PyTorch, etc.)
  - modelN_summary.txt     arquitectura + shapes + estadisticas de cada capa

Uso: python3 export_weights.py [SAVED_MODEL_DIR] [OUT_DIR]
"""
import glob
import json
import os
import shutil
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/rl_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/saved_model')
out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/exported_weights')
os.makedirs(out, exist_ok=True)

for h5 in sorted(glob.glob(os.path.join(src, 'model*.h5'))):
    if h5.endswith('.weights.h5'):
        continue
    name = os.path.splitext(os.path.basename(h5))[0]
    model = load_model(h5, compile=False)
    shutil.copy2(h5, os.path.join(out, name + '.h5'))
    meta_path = h5.replace('.h5', '.json')
    meta = {}
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(out, name + '.json'))
        with open(meta_path) as f:
            meta = json.load(f)
    model.save_weights(os.path.join(out, name + '.weights.h5'))

    arrays = {}
    lines = [f'Modelo: {name}', f'Metadatos: {json.dumps(meta)}', '']
    summary = []
    model.summary(print_fn=lambda s, **_: summary.append(s))
    lines += summary + ['', 'Capas (pesos):']
    for layer in model.layers:
        for w, kind in zip(layer.get_weights(), ('kernel', 'bias')):
            key = f'{layer.name}_{kind}'
            arrays[key] = w
            lines.append(f'  {key:20s} shape={str(w.shape):12s} mean={w.mean():+.5f} '
                         f'std={w.std():.5f} min={w.min():+.4f} max={w.max():+.4f}')
    np.savez(os.path.join(out, name + '_weights.npz'), **arrays)

    # Sanity check: Q-values para un estado de ejemplo (24 lidar=1.0 m, dist=1.0, angulo=0)
    state = np.array([[1.0] * 24 + [1.0, 0.0]], dtype=np.float32)
    q = model.predict(state, verbose=0)[0]
    lines += ['', f'Q(s_ejemplo) = {np.round(q, 3).tolist()}  -> accion {int(q.argmax())}']
    with open(os.path.join(out, name + '_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[OK] {name}: {meta}')

print('Exportado en', out)

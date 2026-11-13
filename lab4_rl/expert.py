"""Expertos para imitacion (Parte B).

OracleExpert  - planificador con informacion PRIVILEGIADA: conoce la posicion y velocidad angular
                de los cilindros (de /model_states de Gazebo). En cada paso evalua un abanico de arcos
                (v, w) mantenidos durante un horizonte de 2 s, predice la orbita de los obstaculos y
                elige el arco que mas avanza a la meta manteniendo distancia de seguridad, con costo
                por w^2 y por cambios de w (suave). Hace las veces de "teleoperador ideal" para generar
                D0 y las correcciones de DAgger de forma reproducible.
KeyboardTeleop - teleoperacion humana por teclado (w/x: v, a/d: w, s: parar, i: tomar/soltar control).
"""
import math
import select
import sys
import termios
import tty

import numpy as np

from lab4_rl.env import (GOAL_RADIUS, OBSTACLE_ANG_VEL, OBSTACLE_RADIUS, V_MAX, W_MAX)

# radio efectivo: el entorno declara choque si el LiDAR mide < 0.19 m a la superficie del obstaculo
ROBOT_RADIUS = 0.20
WALL = 2.35  # cara interior de las paredes


class OracleExpert:
    def __init__(self, horizon=2.4, dt=0.2, margin=0.15, n_v=6, n_w=17):
        self.dt = dt
        self.H = int(round(horizon / dt))
        self.margin = margin
        vs = np.linspace(0.0, V_MAX, n_v)
        ws = np.linspace(-W_MAX, W_MAX, n_w)
        self.cands = np.array([(v, w) for v in vs for w in ws])  # (C, 2)

    def _rollout(self, x, y, yaw):
        """Trayectorias unicycle (C, H, 2) para cada candidato mantenido constante."""
        v = self.cands[:, 0:1]
        w = self.cands[:, 1:2]
        t = np.arange(1, self.H + 1)[None, :] * self.dt  # (1, H)
        th = yaw + w * t
        small = np.abs(w) < 1e-3
        w_safe = np.where(small, 1.0, w)
        xs = np.where(small, x + v * t * math.cos(yaw), x + v / w_safe * (np.sin(th) - math.sin(yaw)))
        ys = np.where(small, y + v * t * math.sin(yaw), y - v / w_safe * (np.cos(th) - math.cos(yaw)))
        return np.stack([xs, ys], axis=-1), t

    def clearance(self, traj, t, obstacles):
        """Distancia libre minima (C, H) entre robot y cilindros/paredes a lo largo del horizonte."""
        if len(obstacles):
            ang = OBSTACLE_ANG_VEL * t[0]  # (H,)
            c, s = np.cos(ang), np.sin(ang)
            # obstaculos rotados por la orbita: (H, K, 2)
            ox = obstacles[None, :, 0] * c[:, None] - obstacles[None, :, 1] * s[:, None]
            oy = obstacles[None, :, 0] * s[:, None] + obstacles[None, :, 1] * c[:, None]
            dx = traj[:, :, None, 0] - ox[None]
            dy = traj[:, :, None, 1] - oy[None]
            d_obs = np.sqrt(dx ** 2 + dy ** 2).min(-1) - OBSTACLE_RADIUS - ROBOT_RADIUS
        else:
            d_obs = np.full(traj.shape[:2], 10.0)
        d_wall = WALL - np.abs(traj).max(-1) - ROBOT_RADIUS
        return np.minimum(d_obs, d_wall)

    def act(self, env):
        """Accion fisica (v, w) del experto para el estado actual del entorno."""
        x, y, yaw = env.pose
        traj, t = self._rollout(x, y, yaw)
        clr = self.clearance(traj, t, env.obstacles_xy())
        gx, gy = env.goal
        d_end = np.hypot(traj[:, -1, 0] - gx, traj[:, -1, 1] - gy)
        d_min = np.hypot(traj[:, :, 0] - gx, traj[:, :, 1] - gy).min(1)
        d0 = math.hypot(gx - x, gy - y)
        progress = d0 - np.minimum(d_end, np.where(d_min < GOAL_RADIUS, 0.0, d_end))

        w_prev = env.prev_cmd[1]
        cost = (-10.0 * progress
                + 0.02 * self.cands[:, 1] ** 2
                + 0.05 * (self.cands[:, 1] - w_prev) ** 2)
        # seguridad: penalizacion suave dentro del margen, dura si hay contacto previsto
        viol = np.clip(self.margin - clr, 0, None)
        weight = np.linspace(1.0, 0.4, self.H)[None, :]  # el futuro cercano pesa mas
        cost += 200.0 * (viol ** 2 * weight).sum(1)
        cost += 1000.0 * (clr.min(1) < 0.02)
        best = int(np.argmin(cost))
        return float(self.cands[best, 0]), float(self.cands[best, 1])

    def danger(self, env, action, horizon_steps=6, threshold=0.05):
        """True si mantener `action` (fisica) lleva a un contacto/casi-contacto en ~1.2 s."""
        x, y, yaw = env.pose
        old = self.cands
        self.cands = np.array([action], dtype=np.float64)
        H = self.H
        self.H = horizon_steps
        traj, t = self._rollout(x, y, yaw)
        clr = self.clearance(traj, t, env.obstacles_xy())
        self.cands, self.H = old, H
        return bool(clr.min() < threshold) or float(env.scan.min()) < 0.25


class KeyboardTeleop:
    """Lectura no bloqueante del teclado en la terminal (modo cbreak).
    i: tomar / devolver el control   w/x: +/- v   a/d: +/- w   s: parar   q: salir"""

    def __init__(self):
        self.active = False
        self.v, self.w = 0.0, 0.0
        self.quit = False
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        print('[teleop] i: tomar/soltar control | w/x: v +/- | a/d: w +/- | s: stop | q: salir',
              flush=True)

    def poll(self):
        while select.select([sys.stdin], [], [], 0)[0]:
            k = sys.stdin.read(1)
            if k == 'i':
                self.active = not self.active
                print(f'[teleop] control humano = {self.active}', flush=True)
            elif k == 'w':
                self.v = min(V_MAX, self.v + 0.04)
            elif k == 'x':
                self.v = max(0.0, self.v - 0.04)
            elif k == 'a':
                self.w = min(W_MAX, self.w + 0.5)
            elif k == 'd':
                self.w = max(-W_MAX, self.w - 0.5)
            elif k == 's':
                self.v, self.w = 0.0, 0.0
            elif k == 'q':
                self.quit = True
            if k in 'wxads':
                self.active = True
        return self.active

    def action(self):
        return self.v, self.w

    def close(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

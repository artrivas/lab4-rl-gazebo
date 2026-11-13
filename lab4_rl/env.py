"""Entorno continuo TurtleBot3 + Gazebo (ROS 2 Humble) con interfaz gymnasium.

Mision: llegar a una meta FIJA esquivando 4 cilindros que orbitan el origen
(mundo turtlebot3_dqn_stage3, radio de orbita ~1.41 m, periodo 40 s).

MDP
---
Estado (28):  24 rayos LiDAR (min-pooling de los 48 del LDS, /3.5 m)
              + distancia a la meta (/5 m) + angulo a la meta (/pi)
              + accion previa normalizada (v, w) en [-1, 1]
Accion  (2):  gym.spaces.Box([0, -2.84], [0.22, 2.84])  ->  Twist en /cmd_vel
Reward     :  r = 20*(d_{t-1} - d_t)            avance hacia la meta
                 - 0.02 * w^2                    giros bruscos
                 - 0.005 * (w_t - w_{t-1})^2     cambios bruscos de giro
                 - 0.5 * prox(d_obs)             cercania a obstaculos (< 0.35 m)
                 - 0.05                          costo por paso
              +100 al llegar (d < 0.25 m), -100 al chocar (LiDAR < 0.19 m)
Episodio   :  max 250 pasos de 0.2 s simulados (= 50 s), 1 paso por scan (5 Hz, tiempo simulado),
              por lo que acelerar Gazebo no cambia la dinamica que ve el agente.
"""
import math
import os
import time

import gymnasium as gym
import numpy as np
import rclpy
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

V_MAX = 0.22
W_MAX = 2.84
LIDAR_MAX = 3.5
GOAL_XY = (1.75, 1.75)
GOAL_RADIUS = 0.25
COLLISION_DIST = 0.19
PROX_DIST = 0.35
N_LIDAR = 24
OBS_DIM = N_LIDAR + 4

R_PROGRESS = 20.0
R_W2 = 0.02
R_DW2 = 0.005
R_PROX = 0.5
R_STEP = 0.05
R_GOAL = 100.0
R_COLLISION = -100.0

OBSTACLES_MODEL = 'turtlebot3_dqn_obstacles'
OBSTACLES_LOCAL = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
OBSTACLE_RADIUS = 0.15
OBSTACLE_ANG_VEL = 2 * math.pi / 40.0  # rad/s (animacion de libobstacles.so)

GOAL_SDF = os.path.expanduser(
    '~/rl_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/'
    'turtlebot3_dqn_world/goal_box/model.sdf')


def norm_to_cmd(u):
    """Accion normalizada u in [-1,1]^2  ->  (v, w) fisicos dentro de los limites del TB3."""
    u = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
    return float((u[0] + 1.0) * 0.5 * V_MAX), float(u[1] * W_MAX)


def cmd_to_norm(v, w):
    return np.array([2.0 * v / V_MAX - 1.0, w / W_MAX], dtype=np.float32)


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TB3ContinuousEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, goal=GOAL_XY, max_steps=250, start_noise=0.25, random_yaw=True,
                 lidar_noise_std=0.0, seed=None, node_name='lab4_env', entity='burger',
                 yaw_noise=None):
        """yaw_noise=None -> yaw inicial ~ U(-pi, pi) (si random_yaw);
        yaw_noise=a     -> yaw inicial = rumbo a la meta + U(-a, a) (condiciones 'faciles' del experto)."""
        super().__init__()
        self.goal = np.array(goal, dtype=np.float64)
        self.max_steps = max_steps
        self.start_noise = start_noise
        self.random_yaw = random_yaw
        self.yaw_noise = yaw_noise
        self.lidar_noise_std = lidar_noise_std
        self.entity = entity
        self.rng = np.random.default_rng(seed)

        self.action_space = gym.spaces.Box(low=np.array([0.0, -W_MAX], dtype=np.float32),
                                           high=np.array([V_MAX, W_MAX], dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,),
                                                dtype=np.float32)

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        self.cmd_pub = self.node.create_publisher(Twist, 'cmd_vel', 10)
        self.node.create_subscription(LaserScan, 'scan', self._scan_cb, qos_profile_sensor_data)
        self.node.create_subscription(Odometry, 'odom', self._odom_cb, 10)
        # pose del modelo de obstaculos (solo lo usa el experto oraculo, NO el agente)
        self.obstacles_yaw = None
        self.node.create_subscription(ModelStates, 'model_states', self._model_states_cb, 10)
        self.set_state_cli = self.node.create_client(SetEntityState, 'set_entity_state')
        self.spawn_cli = self.node.create_client(SpawnEntity, 'spawn_entity')
        self.pause_cli = self.node.create_client(Empty, 'pause_physics')
        self.unpause_cli = self.node.create_client(Empty, 'unpause_physics')

        self.scan = None
        self.scan_count = 0
        self.scan_time = 0.0
        self.pose = None  # (x, y, yaw)
        self._wait_ready()
        self._spawn_goal_marker()

        self.prev_cmd = (0.0, 0.0)
        self.prev_dist = None
        self.steps = 0
        self.t0 = 0.0

    # ------------------------------------------------------------------ ROS
    def _scan_cb(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float64)
        r[np.isnan(r)] = 0.0
        r[np.isinf(r)] = LIDAR_MAX
        self.scan = np.clip(r, 0.0, LIDAR_MAX)
        self.scan_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.scan_count += 1

    def _odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, _yaw(p.orientation))

    def _model_states_cb(self, msg):
        for name, pose in zip(msg.name, msg.pose):
            if name == OBSTACLES_MODEL:
                self.obstacles_yaw = _yaw(pose.orientation)

    def obstacles_xy(self):
        """Posiciones (4, 2) de los cilindros moviles (informacion privilegiada para el experto)."""
        if self.obstacles_yaw is None:
            return np.zeros((0, 2))
        c, s = math.cos(self.obstacles_yaw), math.sin(self.obstacles_yaw)
        return OBSTACLES_LOCAL @ np.array([[c, s], [-s, c]])

    def _spin_until(self, cond, timeout=10.0):
        t_end = time.time() + timeout
        while not cond():
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if time.time() > t_end:
                return False
        return True

    def _call(self, cli, req, timeout=10.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'service {cli.srv_name} not available')
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        return fut.result()

    def _wait_ready(self):
        self.node.get_logger().info('waiting for /scan, /odom and gazebo services ...')
        if not self._spin_until(lambda: self.scan is not None and self.pose is not None, 120.0):
            raise RuntimeError('no /scan or /odom received - is Gazebo running?')
        for cli in (self.set_state_cli, self.pause_cli, self.unpause_cli):
            if not cli.wait_for_service(timeout_sec=60.0):
                raise RuntimeError(f'service {cli.srv_name} not available')

    def _spawn_goal_marker(self):
        try:
            with open(GOAL_SDF) as f:
                xml = f.read()
            req = SpawnEntity.Request()
            req.name = 'lab4_goal'
            req.xml = xml
            req.initial_pose.position.x = float(self.goal[0])
            req.initial_pose.position.y = float(self.goal[1])
            self._call(self.spawn_cli, req, timeout=10.0)
        except Exception as e:  # el marcador es solo visual
            self.node.get_logger().warn(f'could not spawn goal marker: {e}')

    def pause(self):
        """Congela la simulacion (p.ej. mientras se actualiza la red)."""
        self._call(self.pause_cli, Empty.Request())

    def unpause(self):
        self._call(self.unpause_cli, Empty.Request())

    def _publish(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _next_scan(self, n=1):
        target = self.scan_count + n
        if not self._spin_until(lambda: self.scan_count >= target, 10.0):
            # la simulacion pudo quedar en pausa
            self.unpause()
            if not self._spin_until(lambda: self.scan_count >= target, 30.0):
                raise RuntimeError('no new /scan - simulation stalled')

    # ------------------------------------------------------------------ MDP
    def _goal_polar(self):
        x, y, yaw = self.pose
        dx, dy = self.goal[0] - x, self.goal[1] - y
        dist = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) - yaw
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        return dist, ang

    def _lidar24(self):
        s = self.scan
        if len(s) % N_LIDAR == 0:
            s = s.reshape(N_LIDAR, -1).min(axis=1)
        else:
            idx = np.linspace(0, len(s), N_LIDAR, endpoint=False).astype(int)
            s = s[idx]
        return s

    def _obs(self):
        lid = self._lidar24()
        if self.lidar_noise_std > 0:
            lid = np.clip(lid + self.rng.normal(0, self.lidar_noise_std, lid.shape), 0, LIDAR_MAX)
        dist, ang = self._goal_polar()
        return np.concatenate([
            lid / LIDAR_MAX,
            [min(dist / 5.0, 1.0), ang / math.pi],
            cmd_to_norm(*self.prev_cmd),
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._publish(0.0, 0.0)
        x0 = self.rng.uniform(-self.start_noise, self.start_noise)
        y0 = self.rng.uniform(-self.start_noise, self.start_noise)
        if self.yaw_noise is not None:
            yaw0 = (math.atan2(self.goal[1] - y0, self.goal[0] - x0)
                    + self.rng.uniform(-self.yaw_noise, self.yaw_noise))
        elif self.random_yaw:
            yaw0 = self.rng.uniform(-math.pi, math.pi)
        else:
            yaw0 = math.pi / 4
        if options:
            x0 = options.get('x', x0)
            y0 = options.get('y', y0)
            yaw0 = options.get('yaw', yaw0)
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.entity
        st.reference_frame = 'world'
        st.pose.position.x, st.pose.position.y, st.pose.position.z = x0, y0, 0.01
        st.pose.orientation.z = math.sin(yaw0 / 2)
        st.pose.orientation.w = math.cos(yaw0 / 2)
        req.state = st
        self._call(self.set_state_cli, req)
        self._publish(0.0, 0.0)
        self._next_scan(2)
        self.prev_cmd = (0.0, 0.0)
        self.prev_dist, _ = self._goal_polar()
        self.steps = 0
        self.t0 = self.scan_time
        return self._obs(), self._info(0.0, 0.0, 'running')

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), self.action_space.low,
                    self.action_space.high)
        v, w = float(a[0]), float(a[1])
        self._publish(v, w)
        self._next_scan(1)
        self.steps += 1

        dist, _ = self._goal_polar()
        d_obs = float(self.scan.min())
        w_prev = self.prev_cmd[1]

        reward = (R_PROGRESS * (self.prev_dist - dist)
                  - R_W2 * w ** 2
                  - R_DW2 * (w - w_prev) ** 2
                  - R_STEP)
        if d_obs < PROX_DIST:
            reward -= R_PROX * (PROX_DIST - d_obs) / (PROX_DIST - COLLISION_DIST)

        terminated, truncated, outcome = False, False, 'running'
        if dist < GOAL_RADIUS:
            reward, terminated, outcome = R_GOAL, True, 'goal'
        elif d_obs < COLLISION_DIST:
            reward, terminated, outcome = R_COLLISION, True, 'collision'
        elif self.steps >= self.max_steps:
            truncated, outcome = True, 'timeout'

        self.prev_dist = dist
        self.prev_cmd = (v, w)
        if terminated or truncated:
            self._publish(0.0, 0.0)
        return self._obs(), float(reward), terminated, truncated, self._info(v, w, outcome)

    def _info(self, v, w, outcome):
        x, y, yaw = self.pose
        dist, ang = self._goal_polar()
        return {'x': x, 'y': y, 'yaw': yaw, 'v': v, 'w': w, 'dist': dist, 'goal_angle': ang,
                'min_scan': float(self.scan.min()), 't': self.scan_time - self.t0,
                'outcome': outcome, 'step': self.steps}

    def close(self):
        try:
            self._publish(0.0, 0.0)
            self.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()

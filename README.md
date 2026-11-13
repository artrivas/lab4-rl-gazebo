# Lab 4 – RL en Gazebo (TurtleBot3, ROS 2 Humble, PyTorch)

Repositorio de la Práctica 4 (DS5345, 2026-II). Estado: **Fase 1 ✅ · Fase 2 Partes A, B y C ✅ · Fase 3: informe ✅
(`report/Informe_Lab4.pdf`), código ✅, video ⏳ (grabar con `scripts/video_fase1.sh` + `scripts/video_fase2.sh`)**.

```
lab4_rl/
├── launch/lab4_world.launch.py     Gazebo (mundo de obstáculos móviles) + spawn del burger, gui/realtime opcionales
├── worlds/lab4_stage3.world.in     stage3 de ROBOTIS + plugin gazebo_ros_state + física acelerable
├── lab4_rl/
│   ├── env.py                      entorno gymnasium continuo (ROS 2 node): Box(v, w) -> /cmd_vel
│   ├── agents.py                   Actor Gaussiano, Crítico V(s), A2C y PPO-clip (PyTorch)
│   ├── train.py                    Parte A: entrenamiento (--algo a2c|ppo)
│   ├── evaluate.py / select_best.py   evaluación y selección de actor_base.pth
│   ├── expert.py                   Parte B: experto oráculo (planificador) + teleoperación por teclado
│   ├── parte_b.py                  Parte B: D0 -> BC -> rondas DAgger con intervención -> dagger_sim.pth
│   ├── parte_c.py                  Parte C: pares de preferencias, modelo Bradley-Terry, PPO + KL (RLHF)
│   ├── stress.py                   Parte C: success rate con ruido Gaussiano en el LiDAR
│   ├── demo.py                     secuencia para el video (A2C/PPO, BC/DAgger, RLHF)
│   ├── plot_parte_a.py / plot_parte_bc.py   gráficas (1)-(5) + complementarias
│   └── smoke_test.py / check_obstacles.py / test_expert.py / debug_expert.py   pruebas
├── scripts/
│   ├── ros_env.sh                  source de ROS 2 / Gazebo / workspace (+ fix del plugin de obstáculos)
│   ├── run_sim.sh                  Gazebo aislado (ROS_DOMAIN_ID + puerto) + comando; permite correr en paralelo
│   ├── train_parte_a.sh / eval_parte_a.sh   Parte A
│   ├── parte_c.sh                  Parte C completa (pares, RM, RLHF β=0/0.2/1, estrés, gráficas)
│   ├── video_fase1.sh / video_fase2.sh / demo_video.sh   video (Gazebo con GUI en tiempo real)
│   ├── fase1_dqn_run.sh            Fase 1: pipeline DQN de ROBOTIS (stage 4)
│   └── fase1_export_dqn_weights.py Fase 1: exporta pesos del DQN (.h5/.npz)
├── report/                         build_report.py + make_pdf.sh -> Informe_Lab4.pdf
└── results/                        pesos (.pth), CSV/JSON de entrenamiento/evaluación y figuras
```

### Pesos entregables
| archivo | qué es |
|---|---|
| `results/actor_base.pth` | Parte A – mejor actor (A2C semilla 1, 70 % éxito determinista) |
| `results/parte_b/bc.pth` | Parte B – Behavior Cloning sobre D0 |
| `results/parte_b/dagger_sim.pth` | Parte B – DAgger tras 4 rondas |
| `results/parte_c/reward_model.pth` | Parte C – r_ψ(s, a) Bradley-Terry |
| `results/parte_c/rlhf_beta{0,0.2,1}/actor_rlhf.pth` | Parte C – PPO + KL desde π_DAgger |

Todos los actores se cargan con `lab4_rl.agents.load_actor(path)`.

## Fase 1 – Validación del simulador

Máquina WSL `Curso-Robotica` (Ubuntu 22.04, ROS 2 Humble, Gazebo 11, TF 2.19) importada y validada con el DQN
preentrenado de ROBOTIS en stage 4 (`ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage4.launch.py`,
`dqn_gazebo 4`, `dqn_environment`, `dqn_agent ... model_file:=model1.h5`): el robot se mueve en Gazebo y se
imprimen los steps. Se continuó el entrenamiento de 800 → 1000 episodios; pesos exportados en `pesos_exportados/`.
Se instaló PyTorch 2.5.1 (CPU) + gymnasium 0.29.1.

**Hallazgo:** `libobstacles.so` (plugin que mueve los cilindros de stage 3/4) sólo existe en
`~/rl_ws/build/turtlebot3_gazebo/` y no está en `GAZEBO_PLUGIN_PATH`, así que **Gazebo lo ignora en silencio y los
obstáculos quedan estáticos**. `scripts/ros_env.sh` lo agrega al path (verificado con `check_obstacles.py`: el
modelo gira ≈9°/s = periodo de 40 s).

## Fase 2 – Parte A: control continuo con A2C vs PPO

### Misión y escenario
Mundo stage 3 de ROBOTIS (arena 4.7 × 4.7 m): 4 cilindros (r = 0.15 m) orbitan el origen con radio √2 m
y periodo 40 s (≈0.22 m/s, la misma velocidad máxima del robot). El robot parte cerca del origen
(x, y ~ U(−0.25, 0.25), yaw ~ U(−π, π)) y debe llegar a la **meta fija (1.75, 1.75)**, lo que obliga a cruzar
el anillo por donde pasan los obstáculos móviles.

### Formulación del MDP

**Estado** s ∈ ℝ²⁸ (todo normalizado a [−1, 1]):

| componente | dim | definición |
|---|---|---|
| LiDAR | 24 | LDS de 48 rayos (360°) agrupado de a 2 tomando el **mínimo** (conservador), / 3.5 m |
| distancia a la meta | 1 | ‖p_goal − p_robot‖ / 5 m (de `/odom`) |
| ángulo a la meta | 1 | atan2(Δy, Δx) − yaw, envuelto a [−π, π], / π |
| acción previa | 2 | (v_{t−1}, ω_{t−1}) normalizados — necesario porque se penaliza el cambio de ω |

**Acción** continua a = (v, ω): `gym.spaces.Box(low=[0, −2.84], high=[0.22, 2.84])`, publicada como
`geometry_msgs/Twist` en `/cmd_vel`. El actor produce u ∈ ℝ², se recorta a [−1, 1] y se re-escala:
v = (u₁+1)/2 · 0.22, ω = u₂ · 2.84 → nunca se excede el límite cinemático del TurtleBot3.

**Dinámica / paso:** un paso = un scan nuevo del LiDAR (5 Hz en **tiempo simulado**, Δt = 0.2 s). Por eso se puede
acelerar Gazebo (`real_time_update_rate = 0`, ~×5–10) sin cambiar lo que ve el agente. La simulación se pausa
(`/pause_physics`) mientras se actualiza la red.

**Recompensa** (por paso):

$$
r_t = \underbrace{20\,(d_{t-1}-d_t)}_{\text{avance a la meta}}
\;-\; \underbrace{0.02\,\omega_t^2}_{\text{giros bruscos}}
\;-\; \underbrace{0.005\,(\omega_t-\omega_{t-1})^2}_{\text{cambios de giro}}
\;-\; \underbrace{0.5\,\frac{\max(0,\,0.35-d^{obs}_t)}{0.35-0.19}}_{\text{cercanía a obstáculos}}
\;-\; 0.05
$$

- Terminal: **+100** si d_t < 0.25 m (meta); **−100** si d^obs_t < 0.19 m (colisión, LiDAR mínimo).
- Truncamiento a 250 pasos (50 s) → se hace bootstrap con γ·V(s′).
- El término de avance es una diferencia de potencial (−20·d), así que no cambia la política óptima y no se
  puede "farmear" girando en círculos; el costo por paso empuja a llegar rápido.
- Para el entrenamiento las recompensas se escalan × 0.1.

### Arquitectura (PyTorch)

- **Actor Gaussiano** π_θ(u|s) = 𝒩(μ_θ(s), diag σ²): MLP 28 → 256 → 256 → 2 (tanh), μ = tanh(·) ∈ [−1, 1]²;
  σ = exp(log σ) con log σ **parámetro aprendible** por dimensión (v, ω), init −0.5, acotado a [−3, 0.5].
  Inicialización ortogonal (última capa gain 0.01).
- **Crítico** V_φ(s): MLP 28 → 256 → 256 → 1.
- Optimizador Adam, lr = 3·10⁻⁴, γ = 0.99, gradiente recortado a 0.5, c_v = 0.5, c_ent = 10⁻³.

### Algoritmos

| | A2C (estándar) | PPO (clip) |
|---|---|---|
| rollout | n = 32 pasos | 1024 pasos |
| ventaja | retornos n-step (GAE λ = 1), sin normalizar | GAE λ = 0.95, normalizada por minibatch |
| pérdida actor | −E[log π_θ(u\|s) · Â] | −E[min(r_t Â, clip(r_t, 1−ε, 1+ε) Â)], r_t = π_θ/π_θold, **ε = 0.2** |
| actualización | 1 paso de gradiente por rollout (on-policy puro) | 10 épocas × minibatches de 64 |

Ambos usan exactamente la misma red, entorno, recompensa y presupuesto (150k pasos o 3 h), con 2 semillas cada uno
(4 Gazebos en paralelo, cada uno con su `ROS_DOMAIN_ID` y puerto).

### Cómo correr

```bash
source ~/lab4_rl/scripts/ros_env.sh
cd ~/lab4_rl
scripts/train_parte_a.sh 150000 3     # entrena A2C y PPO (2 semillas c/u) en paralelo, headless
scripts/eval_parte_a.sh 20            # evalúa, elige results/actor_base.pth y genera figuras
scripts/demo_video.sh results/actor_base.pth 5   # Gazebo con GUI en tiempo real (video)
```

### Resultados

Entrenamiento: 4 corridas × 150 000 pasos (≈ 8.3 h simuladas c/u, ≈ 2 h 23 min reales, 4 Gazebos en paralelo).
Evaluación: 20 episodios por actor, política determinista (media del actor), mismas semillas de inicio.

| actor | éxito | choque | timeout | retorno medio | t medio a meta (s) |
|---|---|---|---|---|---|
| a2c_s0_best | 70% | 30% | 0% | 71.3 | 13.2 |
| a2c_s0_last | 50% | 50% | 0% | 25.2 | 12.4 |
| **a2c_s1_best → `actor_base.pth`** | **70%** | **30%** | 0% | **73.4** | **11.8** |
| a2c_s1_last | 65% | 35% | 0% | 61.2 | 13.2 |
| ppo_s0_best | 0% | 5% | 95% | −1.7 | – |
| ppo_s0_last | 0% | 45% | 55% | −41.4 | – |
| ppo_s1_best | 0% | 0% | 100% | 0.9 | – |
| ppo_s1_last | 0% | 0% | 100% | −1.3 | – |

Figuras (`results/parte_a/figs/`):
1. `fig1_recompensa_a2c_vs_ppo.png` – recompensa acumulada vs episodios / vs pasos y tasa de éxito.
2. `fig2_perfil_v_w.png` – v(t) y ω(t) en un episodio de prueba (A2C vs PPO).
3. `fig3_trayectorias.png` – trayectorias (x, y) de evaluación sobre el mapa.

**Análisis**

- **A2C** aprende a cruzar el anillo de obstáculos: desde ~15k pasos llega a la meta y se estabiliza en 40–60 % de
  éxito (estocástico) y **70 %** con la política determinista, llegando en ~12 s a casi v_max. Pero es **inestable**:
  la curva oscila fuerte (desv. estándar del retorno ≈ 105–109 en los últimos 50 episodios) y el actor "last" es peor
  que el "best" — típico de un gradiente de política sin restricción de paso, con ventajas de alta varianza (n-step,
  sin normalizar) y ~4 700 actualizaciones.
- **PPO** es **mucho más estable** (desv. ≈ 32, curvas casi superpuestas entre semillas, ~0–16 % de choques) pero
  converge a un **óptimo local conservador**: aprende rápido a *no chocar* (retorno −130 → −15) y luego se queda dentro
  del anillo, avanzando ~0.6 m y deteniéndose (v → 0) hasta el timeout, sin arriesgar el cruce. Causas: (i) el
  objetivo recortado limita cuánto cambia la política por actualización (ε = 0.2) y con rollouts de 1024 pasos hace
  sólo ~146 actualizaciones en 150k pasos (vs ~4 700 de A2C), (ii) los pocos episodios exitosos de la exploración
  inicial quedan diluidos en el batch y la normalización de ventajas, y (iii) el −100 de choque hace que "quedarse
  quieto" (≈ −1 a −15) sea un máximo local seguro frente a cruzar con ~50 % de riesgo.
- **Suavidad de control** (fig. 2): PPO produce ω(t) mucho más suave (media |Δω| = 0.22 rad/s vs 0.97 rad/s de A2C);
  A2C va a velocidad máxima corrigiendo el rumbo con giros bruscos (el término −0.02 ω² es pequeño frente al +100 de
  la meta). Esto motiva las Partes B/C (imitación y RLHF para suavizar).
- Posibles mejoras para PPO (no aplicadas, para mantener la comparación justa): rollouts más cortos (256), más
  entropía/σ inicial mayor, currículo con la meta más cerca, o penalización de choque menor.
- **`actor_base.pth`** = `a2c_s1/actor_best.pth` (mejor éxito y retorno en evaluación). Formato:
  `torch.load(...)` → `{'state_dict', 'config': {obs_dim: 28, act_dim: 2, hidden: [256, 256]}, 'meta'}`;
  cargar con `lab4_rl.agents.load_actor`.

## Fase 2 – Parte B: BC (línea base) vs DAgger

- **Experto:** `OracleExpert` (`lab4_rl/expert.py`) es un planificador con información privilegiada (pose y velocidad
  angular de los cilindros desde `/model_states`): evalúa 102 arcos (v, ω) a 2.4 s, predice la órbita de los obstáculos
  y elige el de mayor avance con margen de seguridad y costo por ω² y Δω → 97 % de éxito, muy suave. Cumple el rol del
  teleoperador de forma reproducible. **Modo humano:** `--expert keyboard` (el agente pilotea; `i` tomar/soltar control,
  `w/x` v, `a/d` ω, `s` stop), correr con `GUI=true REALTIME=true`.
- **D0:** 30 episodios del experto con inicio "fácil" (±5 cm, mirando a la meta) → 2 022 pares. **BC:** regresión MSE
  de la media del actor → `bc.pth`.
- **Distribution shift:** con inicio perturbado (±0.5 m, rumbo aleatorio) BC tiene **0 % de éxito y 100 % de choques**.
- **DAgger (4 rondas):** el agente pilotea, el experto interviene si la acción actual lleva a contacto en ~1.2 s (o si
  no avanza en 5 s); D ← D ∪ D_i (sólo pares de intervención) y fine-tuning in situ con la simulación en pausa.

| política | |D| | intervención | éxito sin ayuda | choque sin ayuda |
|---|---|---|---|---|
| BC | 2022 | 35 % | 0 % | 100 % |
| DAgger-1 | 2772 | 16 % | 47 % | 53 % |
| DAgger-2 | 2973 | 5 % | 73 % | 27 % |
| DAgger-3 | 3030 | 11 % | 60 % | 40 % |
| DAgger-4 → `dagger_sim.pth` | 3163 | 22 % | 60 % | 40 % |

```bash
scripts/run_sim.sh 61 results/parte_b/parte_b.log python3 -m lab4_rl.parte_b --out results/parte_b
```

## Fase 2 – Parte C: Bradley-Terry + RLHF (PPO con penalización KL)

- 20 pares de trayectorias de 12 s del agente DAgger (mismo inicio, ruido σ ∈ {0.05, 0.25, 0.5, 0.8}), etiquetados
  "más segura y fluida" por un etiquetador simulado con criterio explícito (`--labeler interactive` para un humano).
- r_ψ(s, a): MLP 30→128→128→1, ℒ_RM = −ln σ(R̂(τw) − R̂(τl)); 100 % de precisión en train y validación.
- PPO desde π_DAgger con R = r_ψ − β ln(π_θ/π_DAgger), 40k pasos, β ∈ {0, 0.2, 1}.

| política | éxito σ=0 | σ=0.05 m | σ=0.1 m | ω² medio | t a meta |
|---|---|---|---|---|---|
| BC | 0 % | 10 % | 0 % | 0.05 | – |
| DAgger | **90 %** | 60 % | 60 % | 0.65 | 24.7 s |
| RLHF β=0 (reward hacking) | 0 % | 0 % | 0 % | 0.32 | – |
| RLHF β=0.2 | 10 % | 20 % | 20 % | 0.15 | 18.0 s |
| RLHF β=1 | 60 % | 50 % | **80 %** | 0.34 | **15.1 s** |

Con β = 0 el PPO explota r_ψ (la recompensa aprendida sube, la KL crece a ~2 nats/paso y el robot va siempre a v_máx
chocando); con β = 1 la política queda anclada a DAgger, gira la mitad (ω²) y llega más rápido, más robusta con ruido
fuerte pero con menor éxito nominal. Detalle en el informe.

```bash
scripts/parte_c.sh 40000 auto          # o: scripts/parte_c.sh 40000 interactive
```

## Fase 3 – Informe y video

- Informe: `report/Informe_Lab4.pdf` (generado con `report/make_pdf.sh`: HTML con figuras embebidas → PDF vía Edge).
- Video (≤ 4 min): grabar la pantalla (Win + Alt + R) con la terminal y la ventana de Gazebo visibles:
  ```bash
  wsl -d Curso-Robotica -u rlutec -e bash -c "cd ~/lab4_rl && scripts/video_fase1.sh 50"   # Fase 1: DQN base en stage 4
  wsl -d Curso-Robotica -u rlutec -e bash -c "cd ~/lab4_rl && scripts/video_fase2.sh"      # Fase 2: A2C/PPO, BC/DAgger, RLHF
  ```

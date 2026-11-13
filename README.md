# Lab 4 – Aprendizaje por Refuerzo en Gazebo

Navegación de un TurtleBot3 (burger) en Gazebo con ROS 2 Humble y PyTorch: control continuo con
**A2C vs PPO**, imitación con **Behavior Cloning vs DAgger** y ajuste por preferencias con un modelo
**Bradley-Terry + PPO con penalización KL (RLHF)**.

- 📄 Informe: [`report/Informe_Lab4.pdf`](report/Informe_Lab4.pdf)
- 🎥 Videos: [Google Drive](https://drive.google.com/drive/folders/1dMslm1XBuKaOEY9_Ot0jMoJNVFFXcuxt?usp=sharing)

## Escenario

Mundo *stage 3* de ROBOTIS: arena de 4.7 × 4.7 m con 4 cilindros que orbitan el origen (radio √2 m, periodo 40 s).
El robot parte cerca del centro y debe llegar a una **meta fija (1.75, 1.75)** cruzando la zona por donde pasan
los obstáculos móviles.

| | |
|---|---|
| **Estado** (28) | 24 rayos LiDAR + distancia y ángulo a la meta + acción previa, normalizados |
| **Acción** | `gym.spaces.Box([0, −2.84], [0.22, 2.84])` → `Twist` en `/cmd_vel` (v, ω) |
| **Recompensa** | 20·Δd − 0.02·ω² − 0.005·Δω² − cercanía a obstáculos − 0.05; +100 meta, −100 choque |
| **Paso** | un scan del LiDAR (0.2 s simulados); episodio de máx. 250 pasos |

## Estructura

```
lab4_rl/
├── lab4_rl/                 paquete Python
│   ├── env.py               entorno gymnasium continuo (nodo ROS 2)
│   ├── agents.py            actor Gaussiano, crítico V(s), A2C y PPO
│   ├── train.py             entrenamiento A2C / PPO
│   ├── evaluate.py          evaluación con registro paso a paso
│   ├── select_best.py       selección del mejor actor
│   ├── expert.py            experto (planificador) y teleoperación por teclado
│   ├── parte_b.py           Behavior Cloning y DAgger
│   ├── parte_c.py           preferencias, modelo Bradley-Terry y RLHF
│   ├── stress.py            prueba de robustez con ruido en el LiDAR
│   ├── demo.py              secuencia de demostración en Gazebo
│   ├── plots/               generación de gráficas
│   └── tools/               pruebas del entorno y del experto
├── launch/                  launch de Gazebo (gui / tiempo real configurables)
├── worlds/                  mundo de obstáculos móviles
├── scripts/                 scripts de ejecución
├── report/                  informe (PDF) y su generador
└── results/                 pesos, métricas y gráficas (results/graficas/)
```

## Requisitos

Máquina WSL del curso (`Curso-Robotica`: Ubuntu 22.04, ROS 2 Humble, Gazebo 11, workspace `~/rl_ws` con
TurtleBot3) más:

```bash
pip3 install --user torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install --user gymnasium==0.29.1
```

`scripts/ros_env.sh` carga ROS 2, Gazebo y el workspace. También agrega `~/rl_ws/build/turtlebot3_gazebo` a
`GAZEBO_PLUGIN_PATH`: sin eso, el plugin que mueve los cilindros no se carga y los obstáculos quedan estáticos.

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio. `scripts/run_sim.sh` levanta una instancia de
Gazebo aislada (ROS_DOMAIN_ID y puerto propios), por lo que se pueden correr varias en paralelo.

```bash
# Fase 1: DQN preentrenado de ROBOTIS (stage 4)
scripts/fase1_dqn_run.sh

# Parte A: A2C y PPO (2 semillas c/u, en paralelo) y evaluación -> results/actor_base.pth
scripts/parte_a_train.sh 150000 3
scripts/parte_a_eval.sh 20

# Parte B: D0 del experto, BC y 4 rondas de DAgger -> results/parte_b/dagger_sim.pth
scripts/parte_b.sh              # o: scripts/parte_b.sh keyboard  (intervención humana)

# Parte C: pares de preferencias, modelo de recompensa, RLHF (beta = 0, 0.2, 1) y prueba de estrés
scripts/parte_c.sh 40000 auto   # o: scripts/parte_c.sh 40000 interactive  (etiquetado humano)

# Demostraciones en Gazebo con GUI en tiempo real
scripts/video_fase1.sh 50
scripts/video_fase2.sh --episodes 1

# Informe
report/make_pdf.sh
```

Para correr un actor entrenado:

```bash
GUI=true REALTIME=true scripts/run_sim.sh 30 results/eval.log \
  python3 -m lab4_rl.evaluate --actor results/actor_base.pth --episodes 5 --out results/eval
```

Los actores (`*.pth`) se cargan con `lab4_rl.agents.load_actor(path)`.

## Resultados

Las gráficas están en [`results/graficas/`](results/graficas/), y el análisis completo en el
[informe](report/Informe_Lab4.pdf).

| Pesos | Descripción |
|---|---|
| `results/actor_base.pth` | mejor actor de A2C / PPO |
| `results/parte_b/bc.pth` | Behavior Cloning |
| `results/parte_b/dagger_sim.pth` | DAgger |
| `results/parte_c/reward_model.pth` | modelo de recompensa r_ψ(s, a) |
| `results/parte_c/rlhf_beta*/actor_rlhf.pth` | PPO + KL para cada β |

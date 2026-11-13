#!/usr/bin/env python3
"""Genera el informe (HTML autocontenido con figuras embebidas) a partir de results/.
Luego se imprime a PDF con Edge headless (ver report/make_pdf.sh)."""
import base64
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'report', 'informe_lab4.html')


def img(rel, width='100%', caption=''):
    path = os.path.join(R, rel)
    if not os.path.exists(path):
        return f'<p class="missing">[falta figura {rel}]</p>'
    b64 = base64.b64encode(open(path, 'rb').read()).decode()
    cap = f'<figcaption>{caption}</figcaption>' if caption else ''
    return f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}">{cap}</figure>'


def table(header, rows):
    th = ''.join(f'<th>{h}</th>' for h in header)
    trs = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def pct(x):
    return f'{100 * float(x):.0f}%'


# ---------------------------------------------------------------- datos
eval_a = []
for f in sorted(glob.glob(os.path.join(R, 'parte_a', 'eval', '*_summary.json'))):
    s = json.load(open(f))
    name = os.path.basename(f).replace('_summary.json', '')
    t = s['mean_time_to_goal_s']
    eval_a.append([('<b>' + name + ' → actor_base.pth</b>') if name == 'a2c_s1_best' else name,
                   pct(s['success_rate']), pct(s['collision_rate']), pct(s['timeout_rate']),
                   f'{s["mean_return"]:.1f}', '–' if t is None else f'{t:.1f}'])
train_a = json.load(open(os.path.join(R, 'parte_a', 'figs', 'training_summary.json')))
train_rows = [[t['run'], t['episodes'], f'{t["ret_last50"]:.1f} ± {t["ret_last50_std"]:.1f}',
               pct(t['success_last50']), pct(t['collision_last50']), f'{t["best_ma20"]:.1f}']
              for t in train_a]
rounds = list(csv.DictReader(open(os.path.join(R, 'parte_b', 'rounds.csv'))))
round_rows = [[r['policy'], r['dataset_size'], r['new_pairs'], pct(r['intervention_rate']),
               pct(r['supervised_collision_rate']), pct(r['eval_success_rate']),
               pct(r['eval_collision_rate'])] for r in rounds]

stress = []
for f in sorted(glob.glob(os.path.join(R, 'parte_c', 'stress', '*_summary.json'))):
    stress.append(json.load(open(f)))
order = {'BC': 0, 'DAgger': 1, 'RLHF_beta0': 2, 'RLHF_beta0.2': 3, 'RLHF_beta1': 4}
stress.sort(key=lambda s: order.get(s['label'], 9))
noises = list(stress[0]['results'].keys()) if stress else []
stress_rows = [[s['label'].replace('_beta', ' β=')] +
               [f'{pct(s["results"][n]["success_rate"])} ({pct(s["results"][n]["collision_rate"])})'
                for n in noises] + [f'{s["results"][noises[0]]["mean_abs_dw"]:.2f}']
               for s in stress]
rm_log = ''
if os.path.exists(os.path.join(R, 'parte_c', 'rm.log')):
    rm_log = open(os.path.join(R, 'parte_c', 'rm.log')).read().strip()
analysis_c = ''
p = os.path.join(ROOT, 'report', 'parte_c_analisis.html')
if os.path.exists(p):
    analysis_c = open(p).read()

CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;font-size:10.5pt;line-height:1.42;color:#1d1d1f;margin:0}
h1{font-size:20pt;margin:0 0 4px} h2{font-size:14pt;border-bottom:2px solid #0b5394;padding-bottom:3px;
margin-top:22px;color:#0b5394} h3{font-size:11.5pt;margin:14px 0 4px}
.sub{color:#555;margin:0 0 10px} table{border-collapse:collapse;margin:8px 0;font-size:9.5pt;width:100%}
th,td{border:1px solid #bbb;padding:3px 6px;text-align:center} th{background:#e8eef6}
td:first-child{text-align:left} figure{margin:8px 0;text-align:center;page-break-inside:avoid}
figcaption{font-size:9pt;color:#444;margin-top:2px} .eq{background:#f5f7fa;border-left:3px solid #0b5394;
padding:6px 10px;margin:6px 0;font-family:Cambria,'Times New Roman',serif;font-size:11pt}
code{background:#f1f1f1;padding:0 3px;border-radius:3px;font-size:9pt} .note{background:#fff8e1;
border-left:3px solid #f0a500;padding:6px 10px;margin:8px 0} .pb{page-break-before:always}
ul{margin:4px 0 4px 18px;padding:0} li{margin:2px 0} .missing{color:#b00}
@page{size:A4;margin:16mm 14mm}
"""

html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Informe Lab 4 - RL en Gazebo</title><style>{CSS}</style></head><body>

<h1>Práctica de Laboratorio 4: RL en Gazebo</h1>
<p class="sub">DS5345 Aprendizaje por Refuerzo · 2026-II · TurtleBot3 burger · ROS 2 Humble · Gazebo 11 · PyTorch 2.5</p>

<p><b>Misión:</b> navegar hasta una <b>meta fija (1.75, 1.75)</b> evadiendo 4 cilindros (r = 0.15 m) que orbitan el
origen con radio √2 m y periodo 40 s (≈ 0.22 m/s, la misma velocidad máxima del robot), en el mundo
<i>stage 3</i> de ROBOTIS (arena 4.7 × 4.7 m). El robot parte cerca del origen, por lo que siempre debe cruzar el
anillo por donde pasan los obstáculos móviles.</p>

<h2>1. Fase 1 – Configuración y validación del simulador</h2>
<ul>
<li>Opción A: máquina WSL del curso importada (<code>wsl --import Curso-Robotica C:\\WSL\\Curso_Robotica entornoLabRL.tar</code>):
Ubuntu 22.04.5, ROS 2 Humble, Gazebo 11, TensorFlow 2.19. Se instaló PyTorch 2.5.1 (CPU) y gymnasium 0.29.1.
La PC no tiene GPU NVIDIA (AMD Radeon 780M integrada), por lo que todo corre en CPU (el cuello de botella es la física de Gazebo, no la red).</li>
<li><b>Prueba de validación:</b> <code>ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage4.launch.py</code>,
<code>ros2 run turtlebot3_dqn dqn_gazebo 4</code>, <code>dqn_environment</code> y <code>dqn_agent ... model_file:=model1.h5</code>:
el robot se mueve en Gazebo y la terminal imprime los <i>steps</i>. Con el modelo preentrenado (800 episodios) se continuó
el entrenamiento hasta el episodio 1000 (<code>model2.h5</code>, <code>model3.h5</code>; pesos exportados a .h5/.npz).</li>
<li><b>Hallazgo:</b> el plugin <code>libobstacles.so</code> que anima los cilindros de stage 3/4 sólo existe en
<code>~/rl_ws/build/turtlebot3_gazebo/</code> y no está en <code>GAZEBO_PLUGIN_PATH</code>: Gazebo lo ignora en silencio y
<b>los obstáculos quedan estáticos</b>. Se corrigió agregando esa ruta (<code>scripts/ros_env.sh</code>) y se verificó que el
modelo gira ≈ 9°/s.</li>
</ul>

<h2>2. Formulación del MDP (común a las Partes A–C)</h2>
<p><b>Estado</b> s ∈ ℝ<sup>28</sup> (normalizado a [−1, 1]): 24 rayos LiDAR (los 48 del LDS agrupados de a 2 tomando el
<b>mínimo</b>, / 3.5 m) + distancia a la meta (de <code>/odom</code>, / 5 m) + ángulo a la meta relativo al rumbo (/ π)
+ acción previa (v<sub>t−1</sub>, ω<sub>t−1</sub>) normalizada.</p>
<p><b>Acción continua</b> a = (v, ω) ∈ <code>gym.spaces.Box([0, −2.84], [0.22, 2.84])</code> publicada como
<code>geometry_msgs/Twist</code> en <code>/cmd_vel</code>. El actor entrega u ∈ ℝ², que se recorta a [−1, 1] y se
re-escala: v = (u₁+1)/2 · 0.22 m/s, ω = u₂ · 2.84 rad/s, por lo que nunca se saturan los motores.</p>
<p><b>Transición:</b> un paso = un scan nuevo (5 Hz en tiempo <i>simulado</i>, Δt = 0.2 s). Gracias a esto Gazebo puede
correr acelerado (×5, <code>real_time_update_rate = 0</code>) sin cambiar lo que ve el agente; la simulación se pausa
mientras se actualizan las redes. Episodio: máx. 250 pasos (50 s).</p>
<p><b>Recompensa (reward shaping):</b></p>
<div class="eq">r<sub>t</sub> = 20·(d<sub>t−1</sub> − d<sub>t</sub>) − 0.02·ω<sub>t</sub>² − 0.005·(ω<sub>t</sub> − ω<sub>t−1</sub>)²
− 0.5·max(0, 0.35 − d<sup>obs</sup><sub>t</sub>)/0.16 − 0.05 &nbsp;&nbsp;;&nbsp;&nbsp; +100 si d<sub>t</sub> &lt; 0.25 m (meta),
−100 si d<sup>obs</sup><sub>t</sub> &lt; 0.19 m (colisión)</div>
<ul><li>Avance continuo hacia la meta (diferencia de potencial: no se puede "farmear" dando vueltas).</li>
<li>Penaliza giros bruscos (ω²) y cambios bruscos de giro; penaliza acercarse a los obstáculos móviles (&lt; 0.35 m) y el tiempo.</li>
<li>Truncamiento por tiempo con <i>bootstrap</i> γ·V(s′). En entrenamiento las recompensas se escalan ×0.1.</li></ul>

<h2>3. Parte A – Control continuo: A2C vs PPO</h2>
<p><b>Actor Gaussiano</b> π<sub>θ</sub>(u|s) = 𝒩(μ<sub>θ</sub>(s), diag σ²): MLP 28→256→256→2 (tanh), μ = tanh(·);
σ = exp(log σ) con log σ <b>parámetro aprendible</b> para v y ω (init −0.5). <b>Crítico</b> V<sub>φ</sub>(s): MLP 28→256→256→1.
Adam lr 3·10⁻⁴, γ = 0.99, clip de gradiente 0.5, c<sub>v</sub> = 0.5, c<sub>ent</sub> = 10⁻³.</p>
{table(['', 'A2C (estándar)', 'PPO (clip)'], [
    ['rollout', 'n = 32 pasos', '1024 pasos'],
    ['ventaja', 'retornos n-step (GAE λ=1), sin normalizar', 'GAE λ=0.95, normalizada'],
    ['pérdida del actor', '−E[log π<sub>θ</sub>(u|s) · Â]',
     '−E[min(r<sub>t</sub>Â, clip(r<sub>t</sub>, 1−ε, 1+ε)Â)], r<sub>t</sub> = π<sub>θ</sub>/π<sub>θold</sub>, <b>ε = 0.2</b>'],
    ['actualización', '1 paso de gradiente por rollout (~4 700 en total)', '10 épocas × minibatch 64 (~146 rollouts)']])}
<p>Mismo entorno, red, recompensa y presupuesto (150 000 pasos ≈ 8.3 h simuladas), 2 semillas por algoritmo, 4 Gazebos en paralelo.</p>
{img('parte_a/figs/fig1_recompensa_a2c_vs_ppo.png', caption='Gráfica (1): recompensa acumulada por episodio (media móvil 20) vs episodios y vs pasos, y tasa de éxito.')}
{table(['corrida', 'episodios', 'retorno últimos 50 (media ± desv.)', 'éxito últimos 50', 'choque últimos 50', 'mejor media móvil 20'], train_rows)}
<p><b>Evaluación</b> (20 episodios, política determinista = media del actor):</p>
{table(['actor', 'éxito', 'choque', 'timeout', 'retorno medio', 't medio a meta (s)'], eval_a)}
{img('parte_a/figs/fig2_perfil_v_w.png', width='80%', caption='Gráfica (2): perfil temporal v(t), ω(t) en un episodio de prueba.')}
{img('parte_a/figs/fig3_trayectorias.png', width='62%', caption='Trayectorias de evaluación (complementaria).')}
<h3>Análisis</h3>
<ul>
<li><b>A2C</b> aprende a cruzar el anillo desde ~15k pasos y llega a 40–60 % de éxito estocástico y <b>70 %</b> determinista
(~12 s a casi v<sub>máx</sub>). Es <b>inestable</b>: retorno con desviación ≈ 105–109 y el actor final peor que el mejor —
gradiente de política sin restricción del tamaño de paso y ventajas n-step de alta varianza.</li>
<li><b>PPO</b> es mucho <b>más estable</b> (desv. ≈ 32, curvas casi idénticas entre semillas) pero converge a un
<b>óptimo local conservador</b>: aprende rápido a no chocar (−130 → −15) y luego se queda dentro del anillo, frena
(v → 0) y agota el tiempo. El recorte (ε = 0.2), las pocas actualizaciones (~146) y la normalización de ventajas
diluyen los pocos éxitos de la exploración inicial, y −100 por choque hace que quedarse quieto sea un máximo local seguro.</li>
<li><b>Suavidad</b> (gráfica 2): PPO genera ω(t) mucho más suave (media |Δω| 0.22 vs 0.97 rad/s); A2C corrige el rumbo
con giros bruscos porque −0.02ω² es pequeño frente al +100 de la meta. Esto motiva las Partes B y C.</li>
<li><b><code>actor_base.pth</code></b> = A2C semilla 1 (mejor éxito y retorno en evaluación).</li>
</ul>

<h2 class="pb">4. Parte B – Imitation Learning: BC (línea base) vs DAgger</h2>
<div class="note"><b>Experto / "humano".</b> Para que las rondas sean reproducibles, el rol del teleoperador lo cumple un
<b>experto oráculo</b> (<code>lab4_rl/expert.py</code>): un planificador con información privilegiada (posición y velocidad de
los cilindros, de <code>/model_states</code>) que evalúa 102 arcos (v, ω) a 2.4 s, predice la órbita de los obstáculos y elige
el que más avanza con margen de seguridad y costo por ω² y Δω (97 % de éxito, |Δω| ≈ 0.05 rad/s). El mismo script admite
<code>--expert keyboard</code>: el agente pilotea y el humano toma el control con el teclado
(<code>i</code> tomar/soltar, <code>w/x</code> v, <code>a/d</code> ω), capturando igual los pares (s<sub>peligro</sub>, a<sub>experto</sub>).</div>
<ul>
<li><b>D₀</b>: 30 episodios del experto en condiciones "fáciles" (inicio a ±5 cm del origen, mirando a la meta ±0.3 rad) → 2 022 pares (s, a).</li>
<li><b>π<sub>BC</sub></b>: misma arquitectura del actor; regresión MSE de μ<sub>θ</sub>(s) sobre la acción del experto
(200 épocas, MSE validación 0.0054); <code>bc.pth</code>.</li>
<li><b>Despliegue con perturbaciones iniciales</b>: inicio a ±0.5 m y <b>rumbo aleatorio</b> (estados nunca vistos en D₀).</li>
<li><b>DAgger interactivo</b> (4 rondas): el agente pilotea; el experto interviene cuando mantener la acción del agente lleva a
un contacto en ~1.2 s (o el robot se queda sin avanzar 5 s). Se agregan sólo los pares de intervención
D ← D ∪ D<sub>i</sub> y se hace <i>fine-tuning in situ</i> (simulación en pausa, 80 épocas) → <code>dagger_round{{i}}.pth</code>,
final <code>dagger_sim.pth</code>. Cada política se evalúa además sin ayuda en los mismos 15 inicios perturbados.</li>
</ul>
{table(['política', '|D|', 'pares nuevos', 'tasa de intervención', 'choques con supervisión', 'éxito sin supervisión', 'choque sin supervisión'], round_rows)}
{img('figs_bc/fig3_intervencion_colision.png', width='80%', caption='Gráfica (3): tasa de intervención humana y tasa de colisión por ronda de DAgger (ronda 0 = BC).')}
{img('figs_bc/fig4_trayectorias_bc_vs_dagger.png', width='92%', caption='Gráfica (4): mapa 2D de trayectorias (x, y) de /odom, BC vs DAgger, mismos 15 inicios perturbados.')}
<h3>Análisis</h3>
<ul>
<li><b>Distribution shift:</b> BC imita muy bien al experto en su distribución (MSE 0.005) pero con rumbos iniciales
no vistos sus errores se acumulan: <b>0/15 éxitos y 15/15 choques</b>, saliendo en todas direcciones (gráfica 4) — nunca vio
cómo recuperarse.</li>
<li><b>DAgger</b> corrige exactamente esos estados: con 750 pares de la 1ª ronda la colisión baja de 100 % a 53 % y en la
2ª ronda a 27 % con sólo 4.7 % de intervención. Todas las rondas quedan muy por debajo de BC en intervención y colisión.</li>
<li>La tendencia no es monótona: en las rondas 3–4 la intervención sube (11 % y 22 %) por bloques de ~22 pasos del criterio
"sin avanzar" — la política se vuelve más cautelosa (espera a que pasen los cilindros y orbita cerca de la meta antes de
entrar). Con 15 episodios por ronda la variabilidad es ±12 pp (la fase de los obstáculos cambia en cada episodio).</li>
<li>BC gira poco (|ω| medio 0.16 rad/s) y va recto a máxima velocidad hacia donde apunta; DAgger gira más (0.58 rad/s)
porque aprendió a reorientarse y esquivar — las correcciones son bruscas, lo que motiva la Parte C.</li>
</ul>

<h2 class="pb">5. Parte C – Extensión de DAgger con Bradley-Terry y RLHF</h2>
<div class="note"><b>Preferencias.</b> Se generaron 20 pares de trayectorias cortas (60 pasos = 12 s, suficiente para llegar al anillo de obstáculos) con el agente DAgger desde
el mismo inicio, muestreando con distinto ruido (σ ∈ {{0.05, 0.25, 0.5, 0.8}}) para tener evasiones más o menos fluidas.
Las etiquetas τ<sub>w</sub> ≻ τ<sub>l</sub> las produjo un <b>etiquetador simulado</b> con el criterio explícito "más segura y
fluida" (sin choque, mayor distancia mínima a obstáculos, menor Δω² y ω², más avance);
<code>parte_c.py pairs --labeler interactive</code> permite que un humano elija 1/2 viendo las métricas de cada par.</div>
<div class="eq">ℒ<sub>RM</sub>(ψ) = −ln σ( R̂<sub>ψ</sub>(τ<sub>w</sub>) − R̂<sub>ψ</sub>(τ<sub>l</sub>) ),
&nbsp; R̂<sub>ψ</sub>(τ) = Σ<sub>t</sub> r<sub>ψ</sub>(s<sub>t</sub>, a<sub>t</sub>) &nbsp;&nbsp;&nbsp;&nbsp;
R<sub>RLHF</sub>(s, a) = r<sub>ψ</sub>(s, a) − β · ln[ π<sub>θ</sub>(a|s) / π<sub>DAgger</sub>(a|s) ]</div>
<p>r<sub>ψ</sub>: MLP (28+2)→128→128→1, Adam con weight decay, 16 pares de entrenamiento / 4 de validación, normalizada
(media 0, desv. 1). Ajuste final con PPO (ε = 0.2, rollouts de 512, 5 épocas, lr 10⁻⁴, σ inicial 0.1 para π<sub>θ</sub> y π<sub>DAgger</sub>) partiendo de π<sub>DAgger</sub>
(referencia congelada), 40 000 pasos para β = 0, 0.2 y 1.0. <code>{rm_log}</code></p>
{img('figs_bc/fig5_bradley_terry.png', caption='Gráfica (5): curva de pérdida ℒ<sub>RM</sub> de Bradley-Terry y tabla de éxito con ruido Gaussiano en el LiDAR.')}
<p><b>Prueba de estrés</b> – Success Rate en 10 episodios (mismos inicios perturbados), entre paréntesis la tasa de choque:</p>
{table(['política'] + [f'σ<sub>LiDAR</sub> = {n} m' for n in noises] + ['media |Δω| sin ruido (rad/s)'], stress_rows)}
{img('figs_bc/fig6_rlhf_entrenamiento.png', caption='Gráfica (6, complementaria): ajuste PPO – recompensa aprendida, divergencia con π<sub>DAgger</sub> y éxito para cada β.')}
{img('figs_bc/fig7_suavidad.png', width='85%', caption='Gráfica (7, complementaria): ω(t) en un episodio de prueba, DAgger vs RLHF.')}
{analysis_c}

<h2>6. Reproducibilidad y entregables</h2>
<ul>
<li>Código: <code>lab4_rl/</code> (<code>env.py</code> entorno continuo ROS 2 + gymnasium, <code>agents.py</code> actor/crítico/A2C/PPO,
<code>train.py</code>, <code>expert.py</code>, <code>parte_b.py</code>, <code>parte_c.py</code>, <code>stress.py</code>, gráficas),
<code>launch/lab4_world.launch.py</code>, <code>worlds/lab4_stage3.world.in</code>, <code>scripts/</code>.</li>
<li>Pesos: <code>results/actor_base.pth</code>, <code>results/parte_b/bc.pth</code>, <code>results/parte_b/dagger_sim.pth</code>,
<code>results/parte_c/reward_model.pth</code>, <code>results/parte_c/rlhf_beta*/actor_rlhf.pth</code>.</li>
<li>Comandos: <code>scripts/train_parte_a.sh</code>, <code>scripts/eval_parte_a.sh</code>,
<code>scripts/run_sim.sh 61 log python3 -m lab4_rl.parte_b</code>, <code>scripts/parte_c.sh</code>;
video: <code>scripts/video_fase1.sh</code> y <code>scripts/video_fase2.sh</code> (Gazebo con GUI en tiempo real).</li>
</ul>
</body></html>"""

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print('wrote', OUT)

# Disciplina: Métodos Numéricos II
# Autores: Ana Júlia Gonsalves, Breno Gemelgo
# Tarefa 1 - EDOs
# Problema 3: Oscilação capilar de uma gota - modo n = 2
#
# Para uma gota livre, incompressível, de baixa viscosidade e sob pequenas
# deformações, a teoria de Rayleigh fornece para o modo n = 2:
#
# omega^2 = 8*sigma/(rho*R^3)
#
# Se a(t) é o raio polar instantâneo e R é o raio de equilíbrio:
#
# d^2a/dt^2 + omega^2*(a - R) = 0
#
# A variável de deformação é epsilon(t) = (a(t) - R)/R.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / "dats"
output_dir.mkdir(exist_ok=True)

rho = 1000.0  # kg/m^3
sigma = 0.072  # N/m
R = 1.0e-3  # m

epsilon_t0 = 0.05
a_t0 = R * (1.0 + epsilon_t0)
v_t0 = 0.0

omega = np.sqrt(8.0 * sigma / (rho * R**3))
T = 2.0 * np.pi / omega

t_min = 0.0
t_max = 0.016
h = 5.0e-5


def a_real(t):
    return R + (a_t0 - R) * np.cos(omega * t) + v_t0 / omega * np.sin(omega * t)


def v_real(t):
    return -(a_t0 - R) * omega * np.sin(omega * t) + v_t0 * np.cos(omega * t)


def f(v):
    return v


def g_sistema(a):
    return -(omega**2) * (a - R)


n = int((t_max - t_min) / h)
t = np.linspace(t_min, t_max, n + 1)
t_sol = np.linspace(t_min, t_max, 2000)

a_euler = np.zeros(n + 1)
v_euler = np.zeros(n + 1)
a_rk2 = np.zeros(n + 1)
v_rk2 = np.zeros(n + 1)
a_rk4 = np.zeros(n + 1)
v_rk4 = np.zeros(n + 1)

a_euler[0] = a_t0
v_euler[0] = v_t0
a_rk2[0] = a_t0
v_rk2[0] = v_t0
a_rk4[0] = a_t0
v_rk4[0] = v_t0

for i in range(n):
    da = f(v_euler[i])
    dv = g_sistema(a_euler[i])

    a_euler[i + 1] = a_euler[i] + h * da
    v_euler[i + 1] = v_euler[i] + h * dv

for i in range(n):
    k1_a = f(v_rk2[i])
    k1_v = g_sistema(a_rk2[i])

    a_est = a_rk2[i] + h * k1_a
    v_est = v_rk2[i] + h * k1_v

    k2_a = f(v_est)
    k2_v = g_sistema(a_est)

    a_rk2[i + 1] = a_rk2[i] + 0.5 * h * (k1_a + k2_a)
    v_rk2[i + 1] = v_rk2[i] + 0.5 * h * (k1_v + k2_v)

for i in range(n):
    k1_a = f(v_rk4[i])
    k1_v = g_sistema(a_rk4[i])

    k2_a = f(v_rk4[i] + 0.5 * h * k1_v)
    k2_v = g_sistema(a_rk4[i] + 0.5 * h * k1_a)

    k3_a = f(v_rk4[i] + 0.5 * h * k2_v)
    k3_v = g_sistema(a_rk4[i] + 0.5 * h * k2_a)

    k4_a = f(v_rk4[i] + h * k3_v)
    k4_v = g_sistema(a_rk4[i] + h * k3_a)

    a_rk4[i + 1] = a_rk4[i] + h / 6.0 * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
    v_rk4[i + 1] = v_rk4[i] + h / 6.0 * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

a_true = a_real(t)
Ept_euler = np.abs((a_true - a_euler) / a_true) * 100.0
Ept_rk2 = np.abs((a_true - a_rk2) / a_true) * 100.0
Ept_rk4 = np.abs((a_true - a_rk4) / a_true) * 100.0

print(f"omega = {omega:.6f} rad/s")
print(f"T = {T:.6e} s")
print(f"Passos por período = {T / h:.2f}")
print()

# Dados dos gráficos para TikZ
np.savetxt(
    output_dir / "gota_raio_exato.dat",
    np.column_stack((t_sol * 1.0e3, a_real(t_sol) * 1.0e3)),
    header="t_ms a_exato_mm",
    comments="",
    fmt="%.12e",
)

np.savetxt(
    output_dir / "gota_raio_numerico.dat",
    np.column_stack((t * 1.0e3, a_euler * 1.0e3, a_rk2 * 1.0e3, a_rk4 * 1.0e3)),
    header="t_ms a_euler_mm a_rk2_mm a_rk4_mm",
    comments="",
    fmt="%.12e",
)

np.savetxt(
    output_dir / "gota_erro.dat",
    np.column_stack((t * 1.0e3, Ept_euler, Ept_rk2, Ept_rk4)),
    header="t_ms En_euler En_rk2 En_rk4",
    comments="",
    fmt="%.12e",
)

# Comparação do raio polar
plt.figure()
plt.plot(t_sol * 1.0e3, a_real(t_sol) * 1.0e3, "-k", label="Solução exata")
plt.plot(t * 1.0e3, a_euler * 1.0e3, label=f"Euler, h = {h:.1e} s")
plt.plot(t * 1.0e3, a_rk2 * 1.0e3, label=f"RK2, h = {h:.1e} s")
plt.plot(t * 1.0e3, a_rk4 * 1.0e3, label=f"RK4, h = {h:.1e} s")
plt.xlabel("t (ms)")
plt.ylabel("Raio polar a(t) (mm)")
plt.title("Oscilação capilar da gota - modo n = 2")
plt.legend()
plt.grid()

# Erro percentual verdadeiro
plt.figure()
plt.plot(t * 1.0e3, Ept_euler, label="Euler")
plt.plot(t * 1.0e3, Ept_rk2, label="RK2")
plt.plot(t * 1.0e3, Ept_rk4, label="RK4")
plt.xlabel("t (ms)")
plt.ylabel("Erro percentual verdadeiro (%)")
plt.title("Erro numérico na oscilação da gota")
plt.legend()
plt.grid()


# Visualização da forma da gota pelo segundo polinômio de Legendre.
# r(theta,t) = R*[1 + epsilon(t)*P2(cos(theta))]
def forma_gota(tempo):
    epsilon = (a_real(tempo) - R) / R
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    P2 = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    raio = R * (1.0 + epsilon * P2)

    x = raio * np.sin(theta)
    y = raio * np.cos(theta)
    return x, y


tempos_forma = [0.0, 0.25 * T, 0.5 * T, 0.75 * T]
formas = [forma_gota(tempo) for tempo in tempos_forma]
np.savetxt(
    output_dir / "gota_formas.dat",
    np.column_stack(
        (
            1.0e3 * formas[0][0],
            1.0e3 * formas[0][1],
            1.0e3 * formas[1][0],
            1.0e3 * formas[1][1],
            1.0e3 * formas[2][0],
            1.0e3 * formas[2][1],
            1.0e3 * formas[3][0],
            1.0e3 * formas[3][1],
        )
    ),
    header="x_t0_mm y_t0_mm x_t025_mm y_t025_mm x_t050_mm y_t050_mm x_t075_mm y_t075_mm",
    comments="",
    fmt="%.12e",
)

plt.figure()
for tempo, (x_gota, y_gota) in zip(tempos_forma, formas):
    plt.plot(1.0e3 * x_gota, 1.0e3 * y_gota, label=f"t/T = {tempo / T:.2f}")

plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("Deformação da gota durante um período")
plt.axis("equal")
plt.legend()
plt.grid()

plt.show()

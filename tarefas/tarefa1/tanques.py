# Disciplina: Métodos Numéricos II
# Autores: Ana Júlia Gonsalves, Breno Gemelgo
# Tarefa 1 - EDOs
# Problema 2: Equalização de níveis em dois tanques comunicantes
#
# Dois tanques de mesma área A são conectados por um tubo capilar.
# Para escoamento laminar no tubo, a lei de Hagen-Poiseuille fornece
# Q = pi*r^4*Delta_p/(8*mu*L)
#
# com Delta_p = rho*g*(h1 - h2). Assim,
#
# dh1/dt = -(K/A)*(h1 - h2)
# dh2/dt = (K/A)*(h1 - h2)
#
# onde K = pi*rho*g*r^4/(8*mu*L).

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / "dats"
output_dir.mkdir(exist_ok=True)

rho = 1000.0  # kg/m^3
mu = 1.0e-3  # Pa.s
g = 9.81  # m/s^2
r = 0.5e-3  # m
L = 0.10  # m
A = 1.0e-4  # m^2

h1_t0 = 0.15  # m
h2_t0 = 0.05  # m

t_min = 0.0
t_max = 100.0
h = 10.0

K = np.pi * rho * g * r**4 / (8.0 * mu * L)
alpha = K / A


def h1_real(t):
    h_medio = 0.5 * (h1_t0 + h2_t0)
    delta_h0 = 0.5 * (h1_t0 - h2_t0)
    return h_medio + delta_h0 * np.exp(-2.0 * alpha * t)


def h2_real(t):
    h_medio = 0.5 * (h1_t0 + h2_t0)
    delta_h0 = 0.5 * (h1_t0 - h2_t0)
    return h_medio - delta_h0 * np.exp(-2.0 * alpha * t)


def f(h1, h2):
    return -alpha * (h1 - h2)


def g_sistema(h1, h2):
    return alpha * (h1 - h2)


n = int((t_max - t_min) / h)
t = np.linspace(t_min, t_max, n + 1)
t_sol = np.linspace(t_min, t_max, 1000)

h1_euler = np.zeros(n + 1)
h2_euler = np.zeros(n + 1)
h1_rk2 = np.zeros(n + 1)
h2_rk2 = np.zeros(n + 1)
h1_rk4 = np.zeros(n + 1)
h2_rk4 = np.zeros(n + 1)

h1_euler[0] = h1_t0
h2_euler[0] = h2_t0
h1_rk2[0] = h1_t0
h2_rk2[0] = h2_t0
h1_rk4[0] = h1_t0
h2_rk4[0] = h2_t0

for i in range(n):
    dh1 = f(h1_euler[i], h2_euler[i])
    dh2 = g_sistema(h1_euler[i], h2_euler[i])

    h1_euler[i + 1] = h1_euler[i] + h * dh1
    h2_euler[i + 1] = h2_euler[i] + h * dh2

for i in range(n):
    k1_h1 = f(h1_rk2[i], h2_rk2[i])
    k1_h2 = g_sistema(h1_rk2[i], h2_rk2[i])

    h1_est = h1_rk2[i] + h * k1_h1
    h2_est = h2_rk2[i] + h * k1_h2

    k2_h1 = f(h1_est, h2_est)
    k2_h2 = g_sistema(h1_est, h2_est)

    h1_rk2[i + 1] = h1_rk2[i] + 0.5 * h * (k1_h1 + k2_h1)
    h2_rk2[i + 1] = h2_rk2[i] + 0.5 * h * (k1_h2 + k2_h2)

for i in range(n):
    k1_h1 = f(h1_rk4[i], h2_rk4[i])
    k1_h2 = g_sistema(h1_rk4[i], h2_rk4[i])

    k2_h1 = f(
        h1_rk4[i] + 0.5 * h * k1_h1,
        h2_rk4[i] + 0.5 * h * k1_h2,
    )
    k2_h2 = g_sistema(
        h1_rk4[i] + 0.5 * h * k1_h1,
        h2_rk4[i] + 0.5 * h * k1_h2,
    )

    k3_h1 = f(
        h1_rk4[i] + 0.5 * h * k2_h1,
        h2_rk4[i] + 0.5 * h * k2_h2,
    )
    k3_h2 = g_sistema(
        h1_rk4[i] + 0.5 * h * k2_h1,
        h2_rk4[i] + 0.5 * h * k2_h2,
    )

    k4_h1 = f(
        h1_rk4[i] + h * k3_h1,
        h2_rk4[i] + h * k3_h2,
    )
    k4_h2 = g_sistema(
        h1_rk4[i] + h * k3_h1,
        h2_rk4[i] + h * k3_h2,
    )

    h1_rk4[i + 1] = h1_rk4[i] + h / 6.0 * (k1_h1 + 2.0 * k2_h1 + 2.0 * k3_h1 + k4_h1)
    h2_rk4[i + 1] = h2_rk4[i] + h / 6.0 * (k1_h2 + 2.0 * k2_h2 + 2.0 * k3_h2 + k4_h2)

h1_true = h1_real(t)
h2_true = h2_real(t)

Ept_h1_euler = np.abs((h1_true - h1_euler) / h1_true) * 100.0
Ept_h2_euler = np.abs((h2_true - h2_euler) / h2_true) * 100.0
Ept_h1_rk2 = np.abs((h1_true - h1_rk2) / h1_true) * 100.0
Ept_h2_rk2 = np.abs((h2_true - h2_rk2) / h2_true) * 100.0
Ept_h1_rk4 = np.abs((h1_true - h1_rk4) / h1_true) * 100.0
Ept_h2_rk4 = np.abs((h2_true - h2_rk4) / h2_true) * 100.0

# Diagnósticos físicos
Q_t0 = K * (h1_t0 - h2_t0)
area_tubo = np.pi * r**2
u_media_t0 = Q_t0 / area_tubo
Re_t0 = rho * u_media_t0 * (2.0 * r) / mu
print(f"K = {K:.6e} m^2/s")
print(f"alpha = K/A = {alpha:.6e} 1/s")
print(f"Q(0) = {Q_t0:.6e} m^3/s")
print(f"Re(0) = {Re_t0:.2f}")
print()
V0 = A * (h1_t0 + h2_t0)
erro_volume_euler = np.max(np.abs(A * (h1_euler + h2_euler) - V0))
erro_volume_rk2 = np.max(np.abs(A * (h1_rk2 + h2_rk2) - V0))
erro_volume_rk4 = np.max(np.abs(A * (h1_rk4 + h2_rk4) - V0))
print(f"Erro máximo de conservação de volume - Euler: {erro_volume_euler:.6e} m^3")
print(f"Erro máximo de conservação de volume - RK2: {erro_volume_rk2:.6e} m^3")
print(f"Erro máximo de conservação de volume - RK4: {erro_volume_rk4:.6e} m^3")
print()

# Dados dos gráficos para TikZ
np.savetxt(
    output_dir / "tanques_h1_exata.dat",
    np.column_stack((t_sol, h1_real(t_sol))),
    header="t h1_exato",
    comments="",
    fmt="%.12e",
)

np.savetxt(
    output_dir / "tanques_h1_numerica.dat",
    np.column_stack((t, h1_euler, h1_rk2, h1_rk4)),
    header="t h1_euler h1_rk2 h1_rk4",
    comments="",
    fmt="%.12e",
)

np.savetxt(
    output_dir / "tanques_h2_exata.dat",
    np.column_stack((t_sol, h2_real(t_sol))),
    header="t h2_exato",
    comments="",
    fmt="%.12e",
)

np.savetxt(
    output_dir / "tanques_h2_numerica.dat",
    np.column_stack((t, h2_euler, h2_rk2, h2_rk4)),
    header="t h2_euler h2_rk2 h2_rk4",
    comments="",
    fmt="%.12e",
)

h_equilibrio = 0.5 * (h1_t0 + h2_t0)
np.savetxt(
    output_dir / "tanques_equalizacao.dat",
    np.column_stack(
        (
            t_sol,
            h1_real(t_sol),
            h2_real(t_sol),
            np.full_like(t_sol, h_equilibrio),
        )
    ),
    header="t h1_exato h2_exato h_equilibrio",
    comments="",
    fmt="%.12e",
)

# Gráfico do tanque 1
plt.figure()
plt.plot(t_sol, h1_real(t_sol), "-k", label="Solução exata")
plt.plot(t, h1_euler, "-o", label=f"Euler, h = {h}")
plt.plot(t, h1_rk2, "-s", label=f"RK2, h = {h}")
plt.plot(t, h1_rk4, "-^", label=f"RK4, h = {h}")
plt.xlabel("t (s)")
plt.ylabel(r"$h_1$ (m)")
plt.title("Nível do tanque 1")
plt.legend()
plt.grid()

# Gráfico do tanque 2
plt.figure()
plt.plot(t_sol, h2_real(t_sol), "-k", label="Solução exata")
plt.plot(t, h2_euler, "-o", label=f"Euler, h = {h}")
plt.plot(t, h2_rk2, "-s", label=f"RK2, h = {h}")
plt.plot(t, h2_rk4, "-^", label=f"RK4, h = {h}")
plt.xlabel("t (s)")
plt.ylabel(r"$h_2$ (m)")
plt.title("Nível do tanque 2")
plt.legend()
plt.grid()

# Visualização do processo de equalização pela solução analítica
plt.figure()
plt.plot(t_sol, h1_real(t_sol), label=r"$h_1(t)$")
plt.plot(t_sol, h2_real(t_sol), label=r"$h_2(t)$")
plt.axhline(h_equilibrio, linestyle="--", label="Nível de equilíbrio")
plt.xlabel("t (s)")
plt.ylabel("Nível (m)")
plt.title("Equalização dos níveis nos tanques comunicantes")
plt.legend()
plt.grid()

plt.show()

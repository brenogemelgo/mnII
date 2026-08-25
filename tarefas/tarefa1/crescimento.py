# Disciplina: Métodos Numéricos II
# Autores: Ana Júlia Gonsalves, Breno Gemelgo
# Tarefa 1 - EDOs
# Problema 1: Crescimento populacional limitado
#
# Modelo logístico:
#
# dN/dt = r*N*(1 - N/K)
#
# onde:
# N(t) = população
# r    = taxa intrínseca de crescimento
# K    = capacidade de suporte do meio

import os
import numpy as np
import matplotlib.pyplot as plt

r = 0.8
K = 10000.0
N_t0 = 100.0

t_min = 0.0
t_max = 12.0
h = 1.0


def N_real(t):
    return K / (1.0 + ((K - N_t0) / N_t0) * np.exp(-r * t))


def f(N):
    return r * N * (1.0 - N / K)


base_dir = os.path.dirname(os.path.abspath(__file__))
dat_dir = os.path.join(base_dir, "dats")
os.makedirs(dat_dir, exist_ok=True)

n = int((t_max - t_min) / h)
t = np.linspace(t_min, t_max, n + 1)
t_sol = np.linspace(t_min, t_max, 2000)
N_sol = N_real(t_sol)

N_euler = np.zeros(n + 1)
N_rk2 = np.zeros(n + 1)
N_rk4 = np.zeros(n + 1)

N_euler[0] = N_t0
N_rk2[0] = N_t0
N_rk4[0] = N_t0

for i in range(n):
    N_euler[i + 1] = N_euler[i] + h * f(N_euler[i])

for i in range(n):
    k1 = f(N_rk2[i])
    k2 = f(N_rk2[i] + h * k1)
    N_rk2[i + 1] = N_rk2[i] + 0.5 * h * (k1 + k2)

for i in range(n):
    k1 = f(N_rk4[i])
    k2 = f(N_rk4[i] + 0.5 * h * k1)
    k3 = f(N_rk4[i] + 0.5 * h * k2)
    k4 = f(N_rk4[i] + h * k3)
    N_rk4[i + 1] = N_rk4[i] + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

N_true = N_real(t)
Ept_euler = np.abs((N_true - N_euler) / N_true) * 100.0
Ept_rk2 = np.abs((N_true - N_rk2) / N_true) * 100.0
Ept_rk4 = np.abs((N_true - N_rk4) / N_true) * 100.0

crescimento_exato = f(N_sol)

np.savetxt(
    os.path.join(dat_dir, "crescimento_solucao_exata.dat"),
    np.column_stack((t_sol, N_sol)),
    header="t_h N_exato",
    comments="",
)
np.savetxt(
    os.path.join(dat_dir, "crescimento_solucao_numerica.dat"),
    np.column_stack((t, N_euler, N_rk2, N_rk4)),
    header="t_h N_euler N_rk2 N_rk4",
    comments="",
)
np.savetxt(
    os.path.join(dat_dir, "crescimento_erros.dat"),
    np.column_stack((t, Ept_euler, Ept_rk2, Ept_rk4)),
    header="t_h Ept_euler Ept_rk2 Ept_rk4",
    comments="",
)
np.savetxt(
    os.path.join(dat_dir, "crescimento_taxa_crescimento.dat"),
    np.column_stack((t_sol, N_sol, crescimento_exato)),
    header="t_h N_exato dNdt_exato",
    comments="",
)

# Gráfico da população ao longo do tempo
plt.figure()
plt.plot(t_sol, N_sol, "-k", label="Solução exata")
plt.plot(t, N_euler, "-o", label=f"Euler, h = {h}")
plt.plot(t, N_rk2, "-s", label=f"RK2, h = {h}")
plt.plot(t, N_rk4, "-^", label=f"RK4, h = {h}")
plt.xlabel("t (h)")
plt.ylabel("População N(t)")
plt.title("Crescimento logístico de bactérias")
plt.legend()
plt.grid()

# Gráfico da taxa de crescimento exata
plt.figure()
plt.plot(t_sol, crescimento_exato, "-k")
plt.xlabel("t (h)")
plt.ylabel(r"$dN/dt$")
plt.title("Taxa instantânea de crescimento da população")
plt.grid()

# Erro percentual verdadeiro
plt.figure()
plt.semilogy(t[1:], Ept_euler[1:], "-o", label="Euler")
plt.semilogy(t[1:], Ept_rk2[1:], "-s", label="RK2")
plt.semilogy(t[1:], Ept_rk4[1:], "-^", label="RK4")
plt.xlabel("t (h)")
plt.ylabel("Erro percentual verdadeiro (%)")
plt.title("Erro dos métodos numéricos - crescimento logístico")
plt.legend()
plt.grid()

plt.show()

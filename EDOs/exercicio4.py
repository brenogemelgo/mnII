# Disciplina: Métodos Numéricos II
# Aula 2 - Solução de EDOs, Exercício 4

import numpy as np
import matplotlib.pyplot as plt


def y_real(t):
    return np.cos(np.sqrt(k) * t) + (1 / np.sqrt(k)) * np.sin(np.sqrt(k) * t)


def v_real(t):
    return -np.sqrt(k) * np.sin(np.sqrt(k) * t) + np.cos(np.sqrt(k) * t)


# dy/dt = v
def f(v):
    return v


# dv/dt = -ky
def g(k, y):
    return -k * y


k = 1
t_min = 0
t_max = 2 * np.pi
y_t0 = 1
v_t0 = 1
h = 1

t_sol = np.linspace(t_min, t_max, 1000)
y_sol = y_real(t_sol)
v_sol = v_real(t_sol)

n = int((t_max - t_min) / h)

t = np.zeros(n + 1)
y = np.zeros(n + 1)
v = np.zeros(n + 1)

t[0] = t_min
y[0] = y_t0
v[0] = v_t0

for i in range(n):
    y[i + 1] = y[i] + h * f(v[i])
    v[i + 1] = v[i] + h * g(k, y[i])
    t[i + 1] = t[i] + h

y_true = y_real(t)
Ept_y = np.where(y_true != 0, np.abs((y_true - y) / y_true) * 100, 0)
print(f"Ept máximo: {np.max(Ept_y):.6f}%")

plt.figure()
plt.plot(t_sol, y_sol, "-b", label="Solução exata")
plt.plot(t, y, "-ok", label=f"Euler, h = {h}")
plt.xlabel("t")
plt.ylabel("x")
plt.title(r"$y'' = -ky$")
plt.legend()

plt.show()

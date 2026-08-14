# Disciplina: Métodos Numéricos II
# Aula 2 - Solução de EDOs, Exercício 2 (parte 2)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def y_real(t):
    return 4 / 1.3 * (np.exp(0.8 * t) - np.exp(-0.5 * t)) + 2 * np.exp(-0.5 * t)


def f(t, y):
    return 4 * np.exp(0.8 * t) - 0.5 * y


t_min = 0
t_max = 4
y_t0 = 2
h = 1

t_sol = np.linspace(t_min, t_max, 1000)
y_sol = y_real(t_sol)

n = int((t_max - t_min) / h)

t = np.zeros(n + 1)
y = np.zeros(n + 1)

y[0] = y_t0
t[0] = t_min

for i in range(n):
    k1 = f(t[i], y[i])
    k2 = f(t[i] + 0.5 * h, y[i] + 0.5 * k1 * h)
    k3 = f(t[i] + 0.5 * h, y[i] + 0.5 * k2 * h)
    k4 = f(t[i] + h, y[i] + k3 * h)
    y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t[i + 1] = t[i] + h

y_true = y_real(t)
Ept = np.abs(np.abs(y_true - y) / y_true) * 100
dados = pd.DataFrame({"t": t, "y RK4": y, "y exato": y_true, "Ept (%)": Ept})
print(dados.to_string(index=False))
print(f"Ept máximo: {np.max(Ept):.6f}%")

plt.figure()
plt.plot(t_sol, y_sol, "-b", label="Solução exata")
plt.plot(t, y, "-ok", label=f"RK4, h = {h}")
plt.xlabel("t")
plt.ylabel("y")
plt.title(r"$y' = 4\mathrm{e}^{0{,}8t} - 0{,}5y, \qquad t\,\in\,[0,4]$")
plt.legend()
plt.show()

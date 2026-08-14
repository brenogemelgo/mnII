# Disciplina: Métodos Numéricos II
# Aula 2 - Solução de EDOs, Exercício 1

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
hs = [1, 0.5, 0.1, 0.01]

t_sol = np.linspace(t_min, t_max, 1000)
y_sol = y_real(t_sol)

resultados = []

for h in hs:
    n = int((t_max - t_min) / h)

    t = np.zeros(n + 1)
    y = np.zeros(n + 1)

    t[0] = t_min
    y[0] = y_t0

    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
        t[i + 1] = t[i] + h

    y_true = y_real(t)
    Ept = np.abs(np.abs(y_true - y) / y_true) * 100

    resultados.append(
        {"h": h, "y(4) Euler": y[-1], "y(4) exato": y_true[-1], "Ept (%)": Ept[-1]}
    )

    plt.figure()
    plt.plot(t_sol, y_sol, "-b", label="Solução exata")
    plt.plot(t, y, "-ok", label=f"Forward Euler, h = {h}")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title(r"$y' = 4\mathrm{e}^{0{,}8t} - 0{,}5y, \qquad t\,\in\,[0,4]$")
    plt.legend()


df = pd.DataFrame(resultados)
print(df.to_string(index=False))
plt.show()

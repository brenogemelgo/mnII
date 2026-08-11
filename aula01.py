import numpy as np
import matplotlib.pyplot as plt


def f(t, y):
    return 4 * np.exp(0.8 * t) - 0.5 * y


def sol(t):
    return 4 / 1.3 * (np.exp(0.8 * t) - np.exp(-0.5 * t)) + 2 * np.exp(-0.5 * t)


t0 = 0
y0 = 2
t_max = 4
hs = [1, 0.5, 0.1, 0.01]

t_sol = np.linspace(t0, t_max, 1000)
y_sol = sol(t_sol)

for h in hs:
    n = int((t_max - t0) / h)

    t = np.zeros(n + 1)
    y = np.zeros(n + 1)

    y[0] = y0
    t[0] = t0

    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
        t[i + 1] = t[i] + h

    y_real = sol(t)
    erro = np.abs(y_real - y)
    erro_rel = np.abs(erro / y_real) * 100

    print(f"Valor de h: ", h)
    print(f"y aproximado: ", y[n])
    print(f"y real:", y_real[n])
    print(f"Erro relativo: ", erro_rel[n])
    print("\n")

    plt.figure()
    plt.plot(t, y, "-o")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.show()

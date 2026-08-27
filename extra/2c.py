import numpy as np
import matplotlib.pyplot as plt


def y_real(t):
    return (
        t**4 / 4
        - (2 * np.pi + np.exp(1)) / 3 * t**3
        + 0.5 * (np.pi**2 + 2 * np.exp(1) * np.pi) * t**2
        - np.exp(1) * np.pi**2 * t
        + np.pi
    )


def f(t):
    return (t - np.exp(1)) * (t - np.pi) ** 2


t_min = 0
t_max = 2 * np.pi
y_t0 = np.pi
h = np.pi / 4

t_sol = np.linspace(t_min, t_max, 1000)
y_sol = y_real(t_sol)

n = int((t_max - t_min) / h)

t = np.zeros(n + 1)
y = np.zeros(n + 1)

t[0] = t_min
y[0] = y_t0

for i in range(n):
    k1 = f(t[i])
    k2 = f(t[i] + h)
    y[i + 1] = y[i] + 0.5 * h * (k1 + k2)
    t[i + 1] = t[i] + h

y_true = y_real(t)
Ept = np.abs(np.abs(y_true - y) / y_true) * 100

plt.figure()
plt.plot(t_sol, y_sol, "-b", label="Solução exata")
plt.plot(t, y, "-ok", label=f"RK2, h = {h}")
plt.xlabel("t")
plt.ylabel("y")
plt.title(r"$y' = (t-\mathrm{e})(t-\mathrm{pi})^2, \qquad t\,\in\,[0,2\pi]$")
plt.legend()

plt.figure()

plt.plot(t, Ept, "-or")
plt.xlabel("t")
plt.ylabel("$E_{\mathrm{pt}}$ [%]")
plt.title("Erro percentual verdadeiro")

erro = y - y_true
Ep_rms_rel = 100.0 * np.sqrt(np.sum(erro**2) / np.sum(y_true**2))
print(f"Erro percentual RMS global relativo: {Ep_rms_rel:.16f} %")

plt.show()

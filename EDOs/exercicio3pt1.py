import numpy as np
import matplotlib.pyplot as plt


def x_real(t):
    return 68.1 / 0.25 * np.log(np.cosh(np.sqrt(9.81 * 0.25 / 68.1) * t))


def v_real(t):
    return np.sqrt(9.81 * 68.1 / 0.25) * np.tanh(np.sqrt(9.81 * 0.25 / 68.1) * t)


def f(v):
    return v


def g(v):
    return 9.81 - 0.25 / 68.1 * v**2


t_min = 0
t_max = 10
x_t0 = x_real(0)
v_t0 = v_real(0)
h = 2

t_sol = np.linspace(t_min, t_max, 1000)
x_sol = x_real(t_sol)
v_sol = v_real(t_sol)

n = int((t_max - t_min) / h)

t = np.zeros(n + 1)
x = np.zeros(n + 1)
v = np.zeros(n + 1)

t[0] = t_min
x[0] = x_t0
v[0] = v_t0

for i in range(n):
    x[i + 1] = x[i] + h * f(v[i])
    v[i + 1] = v[i] + h * g(v[i])
    t[i + 1] = t[i] + h

x_true = x_real(t)
Ept_x = np.where(x_true != 0, np.abs((x_true - x) / x_true) * 100, 0)
print(f"Ept_x máximo: {np.max(Ept_x):.6f}%")

v_true = v_real(t)
Ept_v = np.where(v_true != 0, np.abs(np.abs(v_true - v) / v_true) * 100, 0)
print(f"Ept_v máximo: {np.max(Ept_v):.6f}%")

plt.figure()
plt.plot(t_sol, x_sol, "-b", label="Solução exata")
plt.plot(t, x, "-ok", label=f"Euler, h = {h}")
plt.xlabel("t")
plt.ylabel("x")
plt.title(r"$x' = v$")
plt.legend()

plt.figure()
plt.plot(t_sol, v_sol, "-b", label="Solução exata")
plt.plot(t, v, "-ok", label=f"Euler, h = {h}")
plt.xlabel("t")
plt.ylabel("v")
plt.title(r"$v' = 9{,}81 - \dfrac{c_d}{m}v^2$")
plt.legend()

plt.show()

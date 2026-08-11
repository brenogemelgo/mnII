import numpy as np


def f(v):
    return v


def g(v):
    return 9.81 - 0.25 / 68.1 * v**2


def v_real(t):
    return np.sqrt(9.81 * 68.1 / 0.25) * np.tanh(np.sqrt(9.81 * 0.25 / 68.1) * t)


def x_real(t):
    return 68.1 / 0.25 * np.log(np.cosh(np.sqrt(9.81 * 0.25 / 68.1) * t))


t_min = 0
t_max = 10
v_t0 = v_real(0)
x_t0 = x_real(0)
h = 2

t_sol = np.linspace(t_min, t_max, 1000)
v_sol = v_real(t_sol)
x_sol = x_real(t_sol)

n = int((t_max - t_min) / h)

t = np.zeros(n + 1)
v = np.zeros(n + 1)
x = np.zeros(n + 1)

t[0] = t_min
v[0] = v_t0
x[0] = x_t0

for i in range(n):
    v[i + 1] = v[i] + h * f(v[i])
    x[i + 1] = x[i] + h * g(v[i])
    t[i + 1] = t[i] + h

v_real = v_real(t)
x_real = x_real(t)
Ept_v = np.max(np.abs(np.abs(v_real - v) / v_real) * 100)
Ept_x = np.max(np.abs(np.abs(x_real - x) / x_real) * 100)

print(Ept_v)
print(Ept_x)

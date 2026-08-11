import numpy as np
import matplotlib.pyplot as plt

# dy/dt = v
# dv/dt = f(t,y,v)


def f(v):
    return v


def g(k, y):
    return -k * y


t_min = 0
t_max = 10
y_t0 = 1
v_t0 = 1
h = 1

t_sol = np.linspace(t_min, t_max, 1000)
x_sol = x_real(t_sol)
v_sol = v_real(t_sol)

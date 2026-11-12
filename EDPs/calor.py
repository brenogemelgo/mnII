import numpy as np
import matplotlib.pyplot as plt

def tdma(a, b, c, d):
    b = b.copy()
    d = d.copy()
    n = len(b)

    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]

    u = np.zeros(n)
    u[-1] = d[-1] / b[-1]

    for i in range(n-2, -1, -1):
        u[i] = (d[i] - c[i]*u[i+1]) / b[i]

    return u


def sol(x, t):
    u = np.zeros_like(x)

    for m in range(1, 201, 2):
        u += (4*u0 / (m*np.pi)) * np.sin(m*np.pi*x/L) * np.exp(-c2*(m*np.pi/L)**2*t)

    return u

L = 50.0
rho = 8.92
cp = 0.092
k = 0.95
c2 = k / (rho * cp)

nx = 40
nt = 1200
T = 300.0

u0 = 25.0
uw = 0.0
ue = 0.0

dx = L / nx
dt = T / nt
lam = c2 * dt / dx**2

x = (np.arange(nx) + 0.5) * dx

u_sol = sol(x, T)
plt.figure()

for theta in (0, 1):

    u = np.full(nx, u0)

    if theta == 0 and lam > 0.25:
        raise ValueError("FTCS: lambda > 0.25")

    if theta == 1:
        a = np.full(nx-1, -lam)
        b = np.full(nx, 1 + 2*lam)
        c = np.full(nx-1, -lam)

        b[0] = b[-1] = 1 + 4*lam
        c[0] = a[-1] = -4*lam/3

    for n in range(nt):

        if theta == 0:
            u_new = u.copy()

            u_new[1:-1] = u[1:-1] + lam * (u[:-2] - 2*u[1:-1] + u[2:])
            u_new[0] = u[0] + lam * (8*uw/3 - 4*u[0] + 4*u[1]/3)
            u_new[-1] = u[-1] + lam * (4*u[-2]/3 - 4*u[-1] + 8*ue/3)

            u = u_new

        else:
            rhs = u.copy()
            rhs[0] += 8*lam*uw/3
            rhs[-1] += 8*lam*ue/3

            u = tdma(a, b, c, rhs)

    if theta == 0:
        metodo = "FTCS" 
    else:
        metodo = "BTCS"

    Ept = np.mean(np.abs((u_sol - u) / u_sol)) 
    print(f"{metodo}: Ept = {Ept:.6f}")
    plt.plot(x, u, "o--", ms=4, label=metodo)

plt.plot(x, u_sol, "k-", label="Analítica")
plt.xlabel("Posição x (cm)")
plt.ylabel("Temperatura (°C)")
plt.title(f"Equação do calor 1D - t = {T:.0f} s")
plt.legend()
plt.show()

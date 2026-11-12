import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil


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
        u[i] = (d[i] - c[i] * u[i+1]) / b[i]
    return u


def sol(x, t):
    u = np.zeros_like(x, dtype=float)
    for m in range(1, 201, 2):
        u += (4*u0 / (m*np.pi)) * np.sin(m*np.pi*x/L) * np.exp(-c2*(m*np.pi/L)**2*t)
    return u



L = 50.0
rho = 8.92
cp = 0.092
k = 0.95
c2 = k / (rho * cp)
u0 = 25.0
uw = 0.0
ue = 0.0
T = 300.0


nx_base = 40
nt_base = 1200
nx_lista = [20, 40, 80, 160]
nt_lista = [1200, 2400, 4800, 9600]
metodos = {0.0: 'FTCS', 0.5: 'CN', 1.0: 'BTCS'}


casos_espaciais = [(nx, ceil(c2*T / (0.20*(L/nx)**2))) for nx in nx_lista]
casos_temporais = [(nx_base, nt) for nt in nt_lista]
casos = list(dict.fromkeys([(nx_base, nt_base)] + casos_espaciais + casos_temporais))


data = Path(__file__).resolve().parent / 'data' / 'questao01'
data.mkdir(parents=True, exist_ok=True)

resultados = {}
historicos = {}
instantaneos_cn = {}
indices_snap = [0, nt_base//10, nt_base//3, nt_base]


for nx, nt in casos:
    dx = L / nx
    dt = T / nt
    lam = c2 * dt / dx**2
    x = (np.arange(nx) + 0.5) * dx
    exata = sol(x, T)

    for theta, nome in metodos.items():
        if theta == 0.0 and lam > 0.25:
            raise ValueError(f'FTCS instável: nx={nx}, nt={nt}, lambda={lam:.4f}')

        u = np.full(nx, u0)
        if theta > 0.0:
            a = np.full(nx-1, -theta*lam)
            b = np.full(nx, 1 + 2*theta*lam)
            c = np.full(nx-1, -theta*lam)
            b[0] = b[-1] = 1 + 4*theta*lam
            c[0] = a[-1] = -4*theta*lam/3

        base = (nx, nt) == (nx_base, nt_base)
        if base:
            historico = np.empty(nt+1)
            historico[0] = 0.5 * (u[nx//2-1] + u[nx//2])
            if theta == 0.5:
                instantaneos_cn[0] = u.copy()

        for n in range(nt):
            if theta == 0.0:
                u_new = u.copy()
                u_new[1:-1] = u[1:-1] + lam*(u[:-2] - 2*u[1:-1] + u[2:])
                u_new[0] = u[0] + lam*(8*uw/3 - 4*u[0] + 4*u[1]/3)
                u_new[-1] = u[-1] + lam*(4*u[-2]/3 - 4*u[-1] + 8*ue/3)
                u = u_new
            else:
                
                rhs = u.copy()
                rhs[1:-1] += (1-theta)*lam*(u[:-2] - 2*u[1:-1] + u[2:])
                rhs[0] += (1-theta)*lam*(-4*u[0] + 4*u[1]/3) + 8*lam*uw/3
                rhs[-1] += (1-theta)*lam*(4*u[-2]/3 - 4*u[-1]) + 8*lam*ue/3
                u = tdma(a, b, c, rhs)

            if base:
                historico[n+1] = 0.5 * (u[nx//2-1] + u[nx//2])
                if theta == 0.5 and (n+1) in indices_snap:
                    instantaneos_cn[n+1] = u.copy()

        
        Ept = np.mean(np.abs((exata-u)/exata))
        resultados[(nx, nt, theta)] = (u.copy(), Ept)
        if base:
            historicos[theta] = historico
        print(f'{nome:4s} nx={nx:3d} nt={nt:5d} lambda={lam:.5f} Ept={100*Ept:.5f}%')


x = (np.arange(nx_base) + 0.5) * (L/nx_base)
u_exata = sol(x, T)
u_ftcs = resultados[(nx_base, nt_base, 0.0)][0]
u_cn = resultados[(nx_base, nt_base, 0.5)][0]
u_btcs = resultados[(nx_base, nt_base, 1.0)][0]
np.savetxt(data/'perfil_final.dat', np.column_stack((x, u_exata, u_ftcs, u_cn, u_btcs)),
           header='x_cm analitica_C FTCS_C CN_C BTCS_C')

tempo = np.linspace(0, T, nt_base+1)

centro_exato = np.array([sol(np.array([L/2]), t)[0] for t in tempo])
centro_exato[0] = u0
np.savetxt(data/'centro_tempo.dat', np.column_stack((tempo, centro_exato,
           historicos[0.0], historicos[0.5], historicos[1.0])),
           header='t_s analitica_C FTCS_C CN_C BTCS_C')

np.savetxt(data/'evolucao_cn.dat', np.column_stack((x, *[instantaneos_cn[i] for i in indices_snap])),
           header='x_cm CN_t0_C CN_t30_C CN_t100_C CN_t300_C')

linhas_nx = []
for nx, nt in casos_espaciais:
    lam = c2 * (T/nt) / (L/nx)**2
    linhas_nx.append([nx, nt, lam] + [resultados[(nx, nt, th)][1] for th in metodos])
linhas_nx = np.asarray(linhas_nx)
np.savetxt(data/'erro_vs_nx.dat', linhas_nx,
           header='nx nt lambda Ept_FTCS Ept_CN Ept_BTCS')

linhas_nt = []
for nx, nt in casos_temporais:
    lam = c2 * (T/nt) / (L/nx)**2
    linhas_nt.append([nx, nt, lam] + [resultados[(nx, nt, th)][1] for th in metodos])
linhas_nt = np.asarray(linhas_nt)
np.savetxt(data/'erro_vs_nt.dat', linhas_nt,
           header='nx nt lambda Ept_FTCS Ept_CN Ept_BTCS')


plt.figure()
xx = np.linspace(0, L, 500)
plt.plot(xx, sol(xx, T), 'k-', label='Analítica')
for theta, nome, estilo in [(0.0, 'FTCS', 'o--'), (0.5, 'Crank-Nicolson', 's--'), (1.0, 'BTCS', 'x--')]:
    plt.plot(x, resultados[(nx_base, nt_base, theta)][0], estilo, ms=4, label=nome)
plt.xlabel('x (cm)')
plt.ylabel('Temperatura (°C)')
plt.title(f'Perfil térmico em t={T:g} s')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(data/'perfil_final.png', dpi=170)

plt.figure()
plt.plot(tempo, centro_exato, 'k-', label='Analítica')
for theta, nome in metodos.items():
    plt.plot(tempo, historicos[theta], label=nome)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura central (°C)')
plt.title('Evolução da temperatura no centro')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(data/'centro_tempo.png', dpi=170)

plt.figure()
for i in indices_snap:
    plt.plot(x, instantaneos_cn[i], 'o-', ms=3, label=f't={i*T/nt_base:g} s')
plt.xlabel('x (cm)')
plt.ylabel('Temperatura (°C)')
plt.title('Evolução espacial — Crank-Nicolson')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(data/'evolucao_cn.png', dpi=170)

plt.figure()
for j, nome in enumerate(metodos.values(), start=3):
    plt.loglog(linhas_nx[:, 0], linhas_nx[:, j], 'o-', label=nome)
plt.xlabel('nx')
plt.ylabel('Ept (adimensional)')
plt.title('log(Ept) vs. log(nx), lambda ≈ 0,20')
plt.grid(alpha=0.3, which='both')
plt.legend()
plt.tight_layout()
plt.savefig(data/'erro_vs_nx.png', dpi=170)

plt.figure()
for j, nome in enumerate(metodos.values(), start=3):
    plt.loglog(linhas_nt[:, 1], linhas_nt[:, j], 'o-', label=nome)
plt.xlabel('nt')
plt.ylabel('Ept (adimensional)')
plt.title(f'log(Ept) vs. log(nt), nx={nx_base}')
plt.grid(alpha=0.3, which='both')
plt.legend()
plt.tight_layout()
plt.savefig(data/'erro_vs_nt.png', dpi=170)

plt.show()
print(f'Arquivos .dat e .png exportados para: {data}')
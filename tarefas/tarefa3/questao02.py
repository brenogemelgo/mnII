import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil
from time import perf_counter


def tdma(a, b, c, d):
    b = b.copy()
    d = d.copy()
    n = len(b)

    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] -= m*c[i-1]
        d[i] -= m*d[i-1]

    u = np.empty(n)
    u[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        u[i] = (d[i] - c[i]*u[i+1]) / b[i]
    return u


def estacionaria(y):
    z = np.asarray(y) / H
    return Tp + (36*mu*U**2/k)*(z/3 - z**2/2 + 2*z**3/3 - z**4/3) - (q_saida/k)*np.asarray(y)


def analitica(y, t):
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if t == 0:
        return np.full_like(y, T0)
    modos = B * np.exp(-alpha * beta**2 * t)
    return estacionaria(y) + modos @ np.sin(beta[:, None] * y[None, :])


def simular(nx, nt, theta, guardar=False):
    dy = H / nx
    dt = tf / nt
    lam = alpha*dt/dy**2
    y = (np.arange(nx) + 0.5)*dy
    fonte = 36*mu*U**2/H**2 * (1 - 2*y/H)**2  
    s = dt*fonte/(rho*cp)                     

    if theta == 0 and lam > 0.25 + 1e-12:
        raise ValueError(f'FTCS fora do limite conservador: nx={nx}, nt={nt}, lambda={lam:.6f}')

    temp = np.full(nx, T0)

    if theta > 0:
        a = np.full(nx-1, -theta*lam)
        b = np.full(nx, 1 + 2*theta*lam)
        c = np.full(nx-1, -theta*lam)
        b[0] = 1 + 4*theta*lam
        c[0] = -4*theta*lam/3
        b[-1] = 1 + theta*lam
        a[-1] = -theta*lam

    t_hist, temp_hist = [], []
    capturas = {}
    capturar = [0, round(nt*2/tf), round(nt*10/tf), nt]
    if guardar:
        t_hist.append(0.0)
        temp_hist.append(temp[nx//2-1:nx//2+1].mean())
        capturas[0] = temp.copy()

    inicio = perf_counter()
    for n in range(nt):
        
        dif = np.empty(nx)
        dif[0] = 8*Tp/3 - 4*temp[0] + 4*temp[1]/3
        dif[1:-1] = temp[:-2] - 2*temp[1:-1] + temp[2:]
        
        dif[-1] = temp[-2] - temp[-1] - dy*q_saida/k

        if theta == 0:
            temp = temp + lam*dif + s
        else:
            rhs = temp + (1-theta)*lam*dif + s
            rhs[0] += theta*lam*8*Tp/3
            rhs[-1] -= theta*lam*dy*q_saida/k
            temp = tdma(a, b, c, rhs)

        if guardar:
            if (n+1) % max(nt//160, 1) == 0 or n+1 == nt:
                t_hist.append((n+1)*dt)
                temp_hist.append(temp[nx//2-1:nx//2+1].mean())
            if n+1 in capturar:
                capturas[n+1] = temp.copy()

    duracao = perf_counter() - inicio
    exata = analitica(y, tf)
    erro_ept = np.mean(np.abs((exata-temp)/exata))
    return y, temp, lam, erro_ept, duracao, np.asarray(t_hist), np.asarray(temp_hist), capturas



H = 1.0e-3              
rho = 982.6             
mu = 5.12               
k = 0.13                
U = 0.20                
Tp = 60.0               
q_saida = 614.4         


cp = 2000.0             
T0 = 60.0               
alpha = k/(rho*cp)      

tf = 30.0              
nx_base = 32
metodos = {0.0: 'FTCS', 0.5: 'CN', 1.0: 'BTCS'}


beta = (np.arange(100) + 0.5)*np.pi/H
pontos, pesos = np.polynomial.legendre.leggauss(350)
yq = 0.5*H*(pontos+1)
wq = 0.5*H*pesos
B = (2/H)*(
    np.sin(beta[:, None]*yq[None, :]) @ ((T0-estacionaria(yq))*wq)
)


nx_lista = [8, 16, 32, 64]
casos_espaciais = [
    (nx, ceil(alpha*tf/(0.20*(H/nx)**2))) for nx in nx_lista
]


nx_temporal = 16
nt_lista = [2400, 4800, 9600, 19200]
casos_temporais = [(nx_temporal, nt) for nt in nt_lista]

casos = list(dict.fromkeys(casos_espaciais + casos_temporais))
caso_base = next((nx, nt) for nx, nt in casos_espaciais if nx == nx_base)

pasta = Path(__file__).resolve().parent / 'data' / 'questao02'
pasta.mkdir(parents=True, exist_ok=True)
resultados = {}

print(f'alpha={alpha:.9e} m^2/s; cp={cp:.1f} J/(kg*K) [HIPOTESE]')
print('metodo  nx     nt  lambda      Ept[%]    tempoCPU[s]')

for nx, nt in casos:
    for theta, nome in metodos.items():
        dados = simular(nx, nt, theta, guardar=(nx, nt) == caso_base)
        resultados[(nx, nt, theta)] = dados
        print(f'{nome:5s} {nx:4d} {nt:6d} {dados[2]:7.4f} '
              f'{100*dados[3]:11.5e} {dados[4]:11.4f}')


nx, nt = caso_base
y = resultados[(nx, nt, 0.0)][0]
y_mm = y*1e3
T_ex = analitica(y, tf)
T_eq = estacionaria(y)
u_ftcs = resultados[(nx, nt, 0.0)][1]
u_cn = resultados[(nx, nt, 0.5)][1]
u_btcs = resultados[(nx, nt, 1.0)][1]

np.savetxt(pasta/'q2_perfil_final.dat',
           np.column_stack((y_mm, T_ex, T_eq, u_ftcs, u_cn, u_btcs)),
           header='y_mm analitica_transiente_C estacionaria_C FTCS_C CN_C BTCS_C')


plt.figure(figsize=(8, 4.8))
y_fino = np.linspace(0, H, 500)
plt.plot(y_fino*1e3, analitica(y_fino, tf), 'k-', label='Analítica transiente')
plt.plot(y_fino*1e3, estacionaria(y_fino), 'k:', label='Limite estacionário')
for u, nome in [(u_ftcs,'FTCS'), (u_cn,'Crank–Nicolson'), (u_btcs,'BTCS')]:
    plt.plot(y_mm, u, 'o--', ms=3, label=nome)
plt.xlabel('y (mm)')
plt.ylabel('Temperatura (°C)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(pasta/'q2_perfil_final.png', dpi=170)


plt.figure(figsize=(8, 4.8))
t_ref = np.linspace(0, tf, 400)
T_ref = np.array([analitica(np.array([H/2]), t)[0] for t in t_ref])
plt.plot(t_ref, T_ref, 'k-', label='Analítica')
for theta, nome in metodos.items():
    r = resultados[(nx, nt, theta)]
    plt.plot(r[5], r[6], label=nome)
plt.xlabel('t (s)')
plt.ylabel('Temperatura em y = H/2 (°C)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(pasta/'q2_centro_tempo.png', dpi=170)


plt.figure(figsize=(8, 4.8))
instantes = resultados[(nx, nt, 0.5)][7]
for passo, u in sorted(instantes.items()):
    plt.plot(y_mm, u, 'o-', ms=3, label=f't={passo*tf/nt:.1f} s')
plt.plot(y_fino*1e3, estacionaria(y_fino), 'k:', label='Estacionária')
plt.xlabel('y (mm)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(pasta/'q2_evolucao.png', dpi=170)
np.savetxt(pasta/'q2_evolucao.dat',
           np.column_stack((y_mm, *[instantes[i] for i in sorted(instantes)])),
           header='y_mm ' + ' '.join(f't_{i*tf/nt:.3f}s_C' for i in sorted(instantes)))


linhas = []
for nx_i, nt_i in casos_espaciais:
    linhas.append([nx_i, nt_i] + [resultados[(nx_i, nt_i, th)][3]
                                   for th in metodos])
erro_nx = np.asarray(linhas)
np.savetxt(pasta/'q2_erro_nx.dat', erro_nx,
           header='nx nt Ept_FTCS Ept_CN Ept_BTCS')
plt.figure(figsize=(7, 4.8))
for j, nome in enumerate(metodos.values(), start=2):
    plt.loglog(erro_nx[:,0], erro_nx[:,j], 'o-', label=nome)
plt.xlabel(r'$n_x$')
plt.ylabel(r'$E_{pt}$')
plt.grid(alpha=0.3, which='both')
plt.legend()
plt.tight_layout()
plt.savefig(pasta/'q2_erro_nx.png', dpi=170)


linhas = []
for nx_i, nt_i in casos_temporais:
    linhas.append([nx_i, nt_i] + [resultados[(nx_i, nt_i, th)][3]
                                   for th in metodos])
erro_nt = np.asarray(linhas)
np.savetxt(pasta/'q2_erro_nt.dat', erro_nt,
           header='nx nt Ept_FTCS Ept_CN Ept_BTCS')
plt.figure(figsize=(7, 4.8))
for j, nome in enumerate(metodos.values(), start=2):
    plt.loglog(erro_nt[:,1], erro_nt[:,j], 'o-', label=nome)
plt.xlabel(r'$n_t$')
plt.ylabel(r'$E_{pt}$')
plt.grid(alpha=0.3, which='both')
plt.legend()
plt.tight_layout()
plt.savefig(pasta/'q2_erro_nt.png', dpi=170)


cpu_nx = np.array([[nxi] + [resultados[(nxi, nti, th)][4] for th in metodos]
                   for nxi, nti in casos_espaciais])
np.savetxt(pasta/'q2_custo_nx.dat', cpu_nx,
           header='nx CPU_FTCS_s CPU_CN_s CPU_BTCS_s')
plt.figure(figsize=(7, 4.8))
for j, nome in enumerate(metodos.values(), start=1):
    plt.loglog(cpu_nx[:,0], cpu_nx[:,j], 'o-', label=nome)
plt.xlabel(r'$n_x$')
plt.ylabel('Tempo computacional (s)')
plt.grid(alpha=0.3, which='both')
plt.legend()
plt.tight_layout()
plt.savefig(pasta/'q2_custo_nx.png', dpi=170)

plt.close('all')
print('Resultados exportados para', pasta.resolve())
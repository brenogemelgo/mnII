from pathlib import Path
from shutil import which
from time import perf_counter

import numpy as np
from scipy.linalg import expm

H = 1.0e-3
RHO = 982.6
MU = 5.12
K = 0.13
UBAR = 0.20
CP = 2000.0
TP = 60.0
T_INI = 60.0
ALPHA = K / (RHO * CP)
T_FINAL = 30.0
Q_GER = 12.0 * MU * UBAR**2 / H
Q_SAIDA = 614.4
THETAS = (0.0, 0.5, 1.0)
TEMPOS_PERFIS = (0.0, 2.0, 10.0, 30.0)
NY_PERFIS, NT_PERFIS = 40, 12000
NY_ESPACIAIS = (20, 40, 80, 160)
NT_ESPACIAL = 500000
NT_TEMPORAIS = (12000, 24000, 48000, 96000)
NT_CUSTO = 2000
NY_VIDEO, NT_VIDEO = 160, 12000
SALVAR_CADA = 100
FRACOES = (0.0, 0.25, 0.5, 0.75, 1.0)

OUT = Path(__file__).resolve().parent / 'data' / 'questao02'
OUT.mkdir(parents=True, exist_ok=True)

BETA = (np.arange(160, dtype=float) + 0.5) * np.pi / H
PONTOS, PESOS = np.polynomial.legendre.leggauss(400)
YQ = 0.5 * H * (PONTOS + 1.0)
WQ = 0.5 * H * PESOS


def fonte(y):
    z = np.asarray(y, dtype=float) / H
    return 36.0 * MU * UBAR**2 / H**2 * (1.0 - 2.0*z)**2


def estacionaria(y, q_saida=Q_SAIDA):
    y = np.asarray(y, dtype=float)
    z = y / H
    return TP + 36.0*MU*UBAR**2/K * (z/3.0-z**2/2.0+2.0*z**3/3.0-z**4/3.0) - q_saida*y/K


B = (2.0/H) * (np.sin(BETA[:, None]*YQ[None, :]) @ ((T_INI-estacionaria(YQ))*WQ))


def analitica(y, t):
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if t == 0.0:
        return np.full_like(y, T_INI)
    return estacionaria(y) + (B*np.exp(-ALPHA*BETA**2*t)) @ np.sin(BETA[:, None]*y[None, :])


def erro_relativo(aproximacao, referencia):
    return float(np.mean(np.abs((aproximacao-referencia)/referencia)))


def salvar(nome, dados, cabecalho):
    array = np.asarray(dados, dtype=float)
    np.savetxt(OUT / nome, array, fmt='%.12e', header=cabecalho)
    print(f'{nome}: {array.shape}')
    return array


def simular(ny, nt, theta, passos, q_saida=Q_SAIDA, duracao=T_FINAL, alta_precisao=False):
    passos = np.asarray(passos, dtype=int)
    if len(passos) == 0 or passos[0] < 0 or passos[-1] > nt or np.any(np.diff(passos) <= 0):
        raise ValueError('Passos devem estar em ordem crescente entre zero e Nt.')
    tipo = np.longdouble if alta_precisao else np.float64
    dy = tipo(H) / ny
    dt = tipo(duracao) / nt
    lam = tipo(ALPHA)*dt/dy**2

    y = (np.arange(ny, dtype=tipo) + tipo(0.5))*dy
    s = dt*fonte(y)/(RHO*CP)
    fluxo = dy*q_saida/K
    v = np.full(ny, T_INI-TP, dtype=tipo)
    novo = np.empty(ny, dtype=tipo)
    rhs = np.empty(ny, dtype=tipo)
    capturas = np.empty((len(passos), ny), dtype=tipo)
    pos = 0
    if passos[0] == 0:
        capturas[0] = TP + v
        pos = 1

    if theta > 0.0:
        a = np.full(ny-1, -theta*lam, dtype=tipo)
        inv = np.empty(ny, dtype=tipo)
        c_mod = np.empty(ny-1, dtype=tipo)
        inv[0] = 1.0/(1.0+4.0*theta*lam)
        for i in range(ny-1):
            c = -(4.0/3.0)*theta*lam if i == 0 else -theta*lam
            c_mod[i] = c*inv[i]
            diag = 1.0 + (theta*lam if i+1 == ny-1 else 2.0*theta*lam)
            inv[i+1] = 1.0/(diag-a[i]*c_mod[i])
        d_mod = np.empty(ny, dtype=tipo)

    fator = (1.0-theta)*lam
    inicio = perf_counter()
    for n in range(1, nt+1):
        rhs[0] = v[0] + fator*(-4.0*v[0]+4.0*v[1]/3.0) + s[0]
        rhs[1:-1] = v[1:-1] + fator*(v[:-2]-2.0*v[1:-1]+v[2:]) + s[1:-1]
        rhs[-1] = v[-1] + fator*(v[-2]-v[-1]-fluxo) + s[-1] - theta*lam*fluxo

        if theta == 0.0:
            v, rhs = rhs, v
        else:
            d_mod[0] = rhs[0]*inv[0]
            for i in range(1, ny):
                d_mod[i] = (rhs[i]-a[i-1]*d_mod[i-1])*inv[i]
            novo[-1] = d_mod[-1]
            for i in range(ny-2, -1, -1):
                novo[i] = d_mod[i]-c_mod[i]*novo[i+1]
            v, novo = novo, v

        if pos < len(passos) and n == passos[pos]:
            capturas[pos] = TP + v
            pos += 1

    return capturas, perf_counter()-inicio


def referencia_semidiscreta(ny):
    dy = H / ny
    y = (np.arange(ny) + 0.5)*dy
    d = np.diag(np.full(ny, -2.0))
    d[0, 0] = -4.0
    d[-1, -1] = -1.0
    for i in range(ny-1):
        d[i, i+1] = 4.0/3.0 if i == 0 else 1.0
        d[i+1, i] = 1.0
    a = ALPHA*d/dy**2
    f = fonte(y)/(RHO*CP)
    f[-1] -= ALPHA*(dy*Q_SAIDA/K)/dy**2
    v_est = np.linalg.solve(-a, f)
    return TP + v_est + expm(a*T_FINAL) @ (np.full(ny, T_INI-TP)-v_est)


def perfis():
    yf = np.linspace(0.0, H, 1001)
    salvar('q2_perfis_analiticos_1001.dat',
           np.column_stack((yf*1000, *[analitica(yf, t) for t in TEMPOS_PERFIS])),
           'y_mm analitica_t0 analitica_t2 analitica_t10 analitica_t30')

    y = (np.arange(NY_PERFIS)+0.5)*H/NY_PERFIS
    passos = [round(t*NT_PERFIS/T_FINAL) for t in TEMPOS_PERFIS]
    estados = {th: simular(NY_PERFIS, NT_PERFIS, th, passos)[0] for th in THETAS}
    colunas = [y*1000]
    erros = []
    for j, t in enumerate(TEMPOS_PERFIS):
        ref = analitica(y, t)
        colunas.extend([ref, *[estados[th][j] for th in THETAS]])
        erros.append([t, *[100*erro_relativo(estados[th][j], ref) for th in THETAS]])
    salvar('q2_perfis_temporais.dat', np.column_stack(colunas),
           'y_mm [analitica FTCS CN BTCS]_t0,_t2,_t10,_t30')
    salvar('q2_erros_instantes.dat', erros, 't_s Ept_porcento_FTCS Ept_porcento_CN Ept_porcento_BTCS')
    print('Erro por instante [%]:')
    for linha in erros:
        print(' '.join(f'{v:.7f}' for v in linha))


def convergencia():
    esp = []
    for ny in NY_ESPACIAIS:
        y = (np.arange(ny)+0.5)*H/ny
        ref = analitica(y, T_FINAL)
        vals = [simular(ny, NT_ESPACIAL, th, (NT_ESPACIAL,))[0][0] for th in THETAS]
        lam = ALPHA*(T_FINAL/NT_ESPACIAL)/(H/ny)**2
        esp.append([ny, NT_ESPACIAL, lam, *[erro_relativo(v, ref) for v in vals]])
    esp = salvar('q2_erro_nx_indep.dat', esp, 'Ny Nt lambda Ept_FTCS Ept_CN Ept_BTCS')

    semidiscreta = referencia_semidiscreta(NY_PERFIS)
    temporal = []
    for nt in NT_TEMPORAIS:
        vals = [simular(NY_PERFIS, nt, th, (nt,), alta_precisao=True)[0][0] for th in THETAS]
        lam = ALPHA*(T_FINAL/nt)/(H/NY_PERFIS)**2
        temporal.append([NY_PERFIS, nt, lam, *[erro_relativo(v, semidiscreta) for v in vals]])
    temporal = salvar('q2_erro_nt_semidiscreto_ny40.dat', temporal,
                     'Ny Nt lambda Ept_t_FTCS Ept_t_CN Ept_t_BTCS')

    print('Ordens espaciais [FTCS CN BTCS]:',
          [-np.polyfit(np.log(esp[:, 0]), np.log(esp[:, j]), 1)[0] for j in (3, 4, 5)])
    print('Ordens temporais [FTCS CN BTCS]:',
          [-np.polyfit(np.log(temporal[:, 1]), np.log(temporal[:, j]), 1)[0] for j in (3, 4, 5)])


def custo():
    dt = 0.2*(H/max(NY_ESPACIAIS))**2/ALPHA
    linhas = []
    for ny in NY_ESPACIAIS:
        valores = []
        for th in THETAS:
            tempos = [simular(ny, NT_CUSTO, th, (NT_CUSTO,), duracao=NT_CUSTO*dt)[1]
                      for _ in range(2)]
            valores.append(min(tempos)*1000/NT_CUSTO)
        linhas.append([ny, *valores])
    salvar('q2_custo_passo.dat', linhas, 'Ny ms_passo_FTCS ms_passo_CN ms_passo_BTCS')


def superficie_e_video():
    passos = np.arange(0, NT_VIDEO+1, SALVAR_CADA)
    tempos = passos*T_FINAL/NT_VIDEO
    y = (np.arange(NY_VIDEO)+0.5)*H/NY_VIDEO
    conjuntos = np.stack([simular(NY_VIDEO, NT_VIDEO, 0.5, passos, q_saida=f*Q_GER)[0]
                          for f in FRACOES], axis=1)
    y_out = np.linspace(0.0, 1.0, 29)
    tempos_out = np.linspace(0.0, T_FINAL, 21)
    for j, frac in enumerate(FRACOES):
        registros = []
        for p, t in enumerate(tempos_out):
            perfil = conjuntos[6*p, j]
            q_saida = frac*Q_GER
            topo = T_INI if p == 0 else perfil[-1] - 0.5*(H/NY_VIDEO)*q_saida/K
            interpolado = np.interp(y_out, np.r_[0.0, y*1000, H*1000],
                                    np.r_[TP, perfil, topo])
            registros.extend(np.column_stack((y_out, np.full(len(y_out), t), interpolado)))
        nome = f'q2_superficie_frac{round(frac*100):02d}.dat'
        salvar(nome, registros, 'y_mm t_s T_C')

    if which('ffmpeg') is None:
        print('ffmpeg ausente: as superficies foram salvas, mas o video nao.')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.22, top=0.84)
    indices = [4, 3, 2, 1, 0]
    malha = ax.pcolormesh(np.linspace(0.0, 1.0, NY_VIDEO+1),
                          np.arange(len(FRACOES)+1), conjuntos[0, indices],
                          cmap='inferno', shading='flat', vmin=50.0, vmax=70.0)
    ax.set_yticks(np.arange(5)+0.5, ['100%', '75%', '50%', '25%', '0%'])
    ax.set_xlabel('Posição y [mm]')
    ax.set_ylabel('Fração removida')
    ax.set_xlim(0.0, 1.0)
    fig.colorbar(malha, ax=ax, label='Temperatura [°C]')
    titulo = ax.set_title('t = 0.0 s')

    def atualizar(i):
        malha.set_array(conjuntos[i, indices].ravel())
        titulo.set_text(f't = {tempos[i]:.2f} s')
        return malha, titulo

    animacao = FuncAnimation(fig, atualizar, frames=len(tempos), interval=1000/24)
    caminho = OUT / 'animacao_fracoes_fluxo.mp4'
    animacao.save(caminho, writer=FFMpegWriter(fps=24, bitrate=2400), dpi=130)
    plt.close(fig)
    print('Animacao:', caminho)


if __name__ == '__main__':
    perfis()
    convergencia()
    custo()
    superficie_e_video()
    print('Resultados em:', OUT)
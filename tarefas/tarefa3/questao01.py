from pathlib import Path
import numpy as np
from scipy.linalg import expm

L = 50.0
RHO = 8.92
CP = 0.092
K = 0.95
C2 = K / (RHO * CP)
U_INI = 25.0
U_W = U_E = 0.0
T_FINAL = 300.0

THETAS = (0.0, 0.5, 1.0)
NX_PERFIS, NT_PERFIS = 40, 1200
TEMPOS_PERFIS = (0.0, 30.0, 100.0, 300.0)
NX_ESPACIAIS = (20, 40, 80, 160)
NT_ESPACIAL = 500_000
NT_TEMPORAIS = (1200, 2400, 4800, 9600)
NT_ESTENDIDOS = (1200, 2400, 4800, 9600, 19200, 38400)
NX_VIDEO, NT_VIDEO = 160, 12000
SALVAR_CADA = 100

OUT = Path(__file__).resolve().parent / 'data' / 'questao01'
OUT.mkdir(parents=True, exist_ok=True)

def analitica(x, t):
    x = np.asarray(x, dtype=np.float64)
    if t == 0.0:
        return np.where((x > 0.0) & (x < L), U_INI, 0.0)
    m = np.arange(1, 201, 2, dtype=np.float64)[:, None]
    termos = (4.0 * U_INI / (m * np.pi)) * np.exp(-C2 * (m * np.pi / L)**2 * t)
    return np.sum(termos * np.sin(m * np.pi * x[None, :] / L), axis=0)

def ept_relativo(u, referencia):
    return float(np.mean(np.abs((u - referencia) / referencia)))

def avancar_theta(nx, nt, theta, passos_salvos):
    dx = L / nx
    dt = T_FINAL / nt
    lam = C2 * dt / (dx * dx)
    u = np.full(nx, U_INI, dtype=np.float64)
    resultado = np.empty((len(passos_salvos), nx), dtype=np.float64)

    sub = np.empty(nx - 1, dtype=np.float64)
    sup_mod = np.empty(nx - 1, dtype=np.float64)
    inv_diag = np.empty(nx, dtype=np.float64)
    if theta > 0.0:
        for i in range(nx - 1):
            sub[i] = -theta * lam
        sub[nx - 2] = -4.0 * theta * lam / 3.0
        inv_diag[0] = 1.0 / (1.0 + 4.0 * theta * lam)
        for i in range(nx - 1):
            sup = -4.0 * theta * lam / 3.0 if i == 0 else -theta * lam
            sup_mod[i] = sup * inv_diag[i]
            diagonal = 1.0 + (4.0 if i + 1 == nx - 1 else 2.0) * theta * lam
            inv_diag[i + 1] = 1.0 / (diagonal - sub[i] * sup_mod[i])

    rhs = np.empty(nx, dtype=np.float64)
    d_mod = np.empty(nx, dtype=np.float64)
    u_novo = np.empty(nx, dtype=np.float64)
    pos = 0
    if len(passos_salvos) > 0 and passos_salvos[0] == 0:
        resultado[0, :] = u
        pos = 1

    fator = (1.0 - theta) * lam
    for n in range(1, nt + 1):
        rhs[0] = u[0] + fator * (-4.0 * u[0] + 4.0 * u[1] / 3.0)
        rhs[1:-1] = u[1:-1] + fator * (u[:-2] - 2.0 * u[1:-1] + u[2:])
        rhs[nx - 1] = u[nx - 1] + fator * (4.0 * u[nx - 2] / 3.0 - 4.0 * u[nx - 1])

        if theta == 0.0:
            u, u_novo = rhs, u
            rhs = u_novo
        else:
            d_mod[0] = rhs[0] * inv_diag[0]
            for i in range(1, nx):
                d_mod[i] = (rhs[i] - sub[i - 1] * d_mod[i - 1]) * inv_diag[i]
            u_novo[nx - 1] = d_mod[nx - 1]
            for i in range(nx - 2, -1, -1):
                u_novo[i] = d_mod[i] - sup_mod[i] * u_novo[i + 1]
            u, u_novo = u_novo, u

        if pos < len(passos_salvos) and n == passos_salvos[pos]:
            resultado[pos, :] = u
            pos += 1

    return resultado

def simular(nx, nt, theta, passos=(0,)):
    lam = C2 * T_FINAL / (nt * (L / nx)**2)
    if theta == 0.0 and lam > 0.25:
        raise ValueError(f'FTCS: lambda={lam:.6g} excede limite conservador 1/4')
    passos = np.asarray(passos, dtype=np.int64)
    if len(passos) == 0 or passos[0] < 0 or passos[-1] > nt or np.any(np.diff(passos) <= 0):
        raise ValueError('Passos salvos devem estar ordenados, sem repeticoes, em [0,Nt].')
    return avancar_theta(nx, nt, theta, passos)

def referencia_semidiscreta(nx):
    dx = L / nx
    D = np.diag(np.full(nx, -2.0))
    D[0, 0] = D[-1, -1] = -4.0
    for i in range(nx - 1):
        D[i, i + 1] = 4.0 / 3.0 if i == 0 else 1.0
        D[i + 1, i] = 4.0 / 3.0 if i == nx - 2 else 1.0
    return expm((C2 * T_FINAL / dx**2) * D) @ np.full(nx, U_INI)

def salvar_tabela(nome, linhas, cabecalho):
    array = np.asarray(linhas, dtype=np.float64)
    np.savetxt(OUT / nome, array, fmt='%.12e', header=cabecalho)
    return array

def perfis():
    x_ref = np.linspace(0.0, L, 1001)
    salvar_tabela('perfis_analiticos_1001.dat',
                  np.column_stack((x_ref, *[analitica(x_ref, t) for t in TEMPOS_PERFIS])),
                  'x_cm u_an_t0 u_an_t30 u_an_t100 u_an_t300')

    x = (np.arange(NX_PERFIS) + 0.5) * L / NX_PERFIS
    passos = [round(t * NT_PERFIS / T_FINAL) for t in TEMPOS_PERFIS]
    estados = {th: simular(NX_PERFIS, NT_PERFIS, th, passos) for th in THETAS}
    colunas = [x]
    for j, t in enumerate(TEMPOS_PERFIS):
        colunas.extend([analitica(x, t), *[estados[th][j] for th in THETAS]])
    salvar_tabela('perfis_temporais.dat', np.column_stack(colunas),
                  'x_cm [analitica FTCS CN BTCS]_t0,_t30,_t100,_t300')
    print('Perfis (Ept em %, t=0,30,100,300 s):')
    for j, t in enumerate(TEMPOS_PERFIS):
        ref = analitica(x, t)
        erros = [100 * ept_relativo(estados[th][j], ref) for th in THETAS]
        print(f'  t={t:3g}: ' + ' '.join(f'{e:.6f}' for e in erros))

def convergencia():
    linhas_esp = []
    for nx in NX_ESPACIAIS:
        x = (np.arange(nx) + 0.5) * L / nx
        ref = analitica(x, T_FINAL)
        estados = [simular(nx, NT_ESPACIAL, th, (NT_ESPACIAL,))[0] for th in THETAS]
        lam = C2 * T_FINAL / (NT_ESPACIAL * (L/nx)**2)
        linhas_esp.append([nx, NT_ESPACIAL, lam] + [ept_relativo(u, ref) for u in estados])
    esp = salvar_tabela('erro_vs_nx.dat', linhas_esp,
                       'Nx Nt lambda Ept_FTCS Ept_CN Ept_BTCS ; referencia=PDE analitica ; Ept fracao')

    ref_sd = referencia_semidiscreta(NX_PERFIS)
    x = (np.arange(NX_PERFIS) + 0.5) * L / NX_PERFIS
    ref_pde = analitica(x, T_FINAL)
    linhas_temporal = []
    linhas_total = []
    for nt in NT_ESTENDIDOS:
        estados = [simular(NX_PERFIS, nt, th, (nt,))[0] for th in THETAS]
        lam = C2 * T_FINAL / (nt * (L/NX_PERFIS)**2)
        prefixo = [NX_PERFIS, nt, lam]
        if nt in NT_TEMPORAIS:
            linhas_temporal.append(prefixo + [ept_relativo(u, ref_sd) for u in estados])
        linhas_total.append(prefixo + [ept_relativo(u, ref_pde) for u in estados])
    puro = salvar_tabela('erro_vs_nt_temporal_puro.dat', linhas_temporal,
                        'Nx Nt lambda Ept_t_FTCS Ept_t_CN Ept_t_BTCS ; referencia=semidiscreta ; fracao')
    total = salvar_tabela('erro_vs_nt_estendido.dat', linhas_total,
                         'Nx Nt lambda Ept_FTCS Ept_CN Ept_BTCS ; referencia=PDE analitica ; fracao')

    print('Ordens espaciais (ajuste log-log FTCS, CN, BTCS):',
          np.round([-np.polyfit(np.log(esp[:, 0]), np.log(esp[:, j]), 1)[0] for j in (3,4,5)], 4))
    print('Ordens temporais (ajuste log-log FTCS, CN, BTCS):',
          np.round([-np.polyfit(np.log(puro[:, 1]), np.log(puro[:, j]), 1)[0] for j in (3,4,5)], 4))
    print(f'Limite espacial Ept = {100*ept_relativo(ref_sd, ref_pde):.9f}%')
    return esp, puro, total

def superficie_e_video():
    passos = np.arange(0, NT_VIDEO + 1, SALVAR_CADA, dtype=np.int64)
    estados = simular(NX_VIDEO, NT_VIDEO, 0.5, passos)
    tempos = passos * (T_FINAL / NT_VIDEO)
    x = (np.arange(NX_VIDEO) + 0.5) * L / NX_VIDEO

    idx_t = np.arange(0, len(passos), 2)
    idx_x = np.arange(0, NX_VIDEO, 3)
    arquivo = OUT / 'superficie_cn_pgfplots_56_tuniforme5.dat'
    with arquivo.open('w', encoding='utf-8') as f:
        for j in idx_t:
            f.write(f'{0:.8f} {tempos[j]:.8f} {U_W:.12f}\n')
            for i in idx_x:
                f.write(f'{x[i]:.8f} {tempos[j]:.8f} {estados[j,i]:.12f}\n')
            f.write(f'{L:.8f} {tempos[j]:.8f} {U_E:.12f}\n\n')
    assert len(idx_x) + 2 == 56 and len(idx_t) == 61
    print('Superficie 3D: 56 x 61 pontos:', arquivo)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from shutil import which

        if which('ffmpeg') is None:
            print('ffmpeg ausente: dados gerados, animacao nao exportada.')
            return
        fig, ax = plt.subplots(figsize=(11, 2.6))
        fig.subplots_adjust(left=0.085, right=0.86, bottom=0.30, top=0.75)
        malha = ax.pcolormesh(np.linspace(0.0, L, NX_VIDEO + 1), [0.0, 1.0],
                             estados[0][None, :], shading='flat', cmap='inferno',
                             vmin=0, vmax=U_INI)
        ax.set_xlim(0, L)
        ax.set_yticks([])
        ax.set_xlabel('Posicao x [cm]')
        cb = fig.colorbar(malha, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label('Temperatura [°C]')
        titulo = ax.set_title('t = 0 s')

        def atualizar(i):
            malha.set_array(estados[i].reshape(1, NX_VIDEO))
            titulo.set_text(f't = {tempos[i]:.1f} s')
            return malha, titulo

        anim = FuncAnimation(fig, atualizar, frames=len(tempos), interval=1000 / 24)
        video = OUT / 'barra_temperatura_cn_latex.mp4'
        anim.save(video, writer=FFMpegWriter(fps=24, bitrate=2000), dpi=120)
        plt.close(fig)
        print('Animacao:', video)
    except (ImportError, RuntimeError, OSError) as e:
        print('Dados e superficie exportados; nao foi possivel criar video:', e)

if __name__ == '__main__':
    perfis()
    convergencia()
    superficie_e_video()
    print('Arquivos de apresentacao em:', OUT)

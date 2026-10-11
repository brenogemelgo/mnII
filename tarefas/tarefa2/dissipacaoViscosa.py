# Disciplina: Métodos Numéricos II
# Autores: Ana Júlia Gonsalves, Breno Gemelgo
# Tarefa 2 - EDPs
# Problema: aquecimento de petróleo pesado por dissipação viscosa
#
# Petróleo escoa entre duas placas paralelas (Poiseuille plano).
# O cisalhamento gera calor: q(y) = mu * (du/dy)^2.
# Na direção transversal, a equação de energia é:
#
# -k * d2T/dy2 = q(y)
#
# Condições de contorno:
#   T(0) = T_parede                      (temperatura prescrita)
#  -k*T'(H) = fluxo_saida                (fluxo prescrito)

from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

pasta = Path(__file__).resolve().parent
(pasta / "dats").mkdir(exist_ok=True)
# (pasta / "figuras").mkdir(exist_ok=True)

# -------------------------------------------------------------------
# PARÂMETROS FÍSICOS
# -------------------------------------------------------------------
rho = 900.0  # kg/m3
mu = 5.0  # Pa.s
k = 0.13  # W/(m.K)
H = 1.0e-3  # m
U = 0.20  # m/s
T_parede = 40.0  # Celsius

q_total = 12.0 * mu * U**2 / H
fluxo_saida = 0.25 * q_total  # W/m2

# -------------------------------------------------------------------
# PARÂMETROS NUMÉRICOS
# -------------------------------------------------------------------
N = 32
malhas = [8, 16, 32, 64, 128, 256]
Eppara = 0.5 * 10 ** (2 - 6)
maxit = 100000
lambida = 1.8


# -------------------------------------------------------------------
# ESCOAMENTO, DISSIPAÇÃO E SOLUÇÃO ANALÍTICA
# -------------------------------------------------------------------
def velocidade(y):
    s = y / H
    return 6.0 * U * s * (1.0 - s)


def geracao(y):
    dud_y = (6.0 * U / H) * (1.0 - 2.0 * y / H)
    return mu * dud_y**2


def temperatura_exata(y, fluxo):
    s = y / H
    return (
        T_parede
        + 36.0 * mu * U**2 / k * (s / 3 - s**2 / 2 + 2 * s**3 / 3 - s**4 / 3)
        - fluxo * H / k * s
    )


# -------------------------------------------------------------------
# DIFERENÇAS FINITAS: SISTEMA TRIDIAGONAL A*T = d
# -------------------------------------------------------------------
def montar_sistema(n, fluxo):
    dx = H / n
    y = (np.arange(n) + 0.5) * dx

    # Células internas:
    # -T[i-1] + 2*T[i] - T[i+1] = q(y[i])*dx**2/k
    a = np.full(n, -1.0)  # diagonal inferior
    b = np.full(n, 2.0)  # diagonal principal
    c = np.full(n, -1.0)  # diagonal superior
    d = geracao(y) * dx**2 / k  # lado direito

    # Primeira célula: T[-1] = 2*T_parede - T[0]
    # 3*T[0] - T[1] = q(y[0])*dx**2/k + 2*T_parede
    a[0] = 0.0
    b[0] = 3.0
    d[0] += 2.0 * T_parede

    # Última célula: T[n] = T[n-1] - fluxo*dx/k
    # -T[n-2] + T[n-1] = q(y[-1])*dx**2/k - fluxo*dx/k
    b[-1] = 1.0
    c[-1] = 0.0
    d[-1] -= fluxo * dx / k

    return y, dx, a, b, c, d


# -------------------------------------------------------------------
# SOLUÇÃO DO SISTEMA LINEAR
# -------------------------------------------------------------------
def funcEpest(x_new, x_old):
    erros = np.zeros(len(x_new))

    for i in range(len(x_new)):
        if x_new[i] != 0:
            erros[i] = abs((x_new[i] - x_old[i]) / x_new[i]) * 100
        else:
            erros[i] = 100

    return np.max(erros)


def jacobi(A, b, Eppara, maxit):
    n = len(b)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    Epest = 100
    k = 0

    while Epest > Eppara and k < maxit:
        for i in range(n):
            soma = 0

            for j in range(n):
                if j != i:
                    soma += A[i, j] * x_old[j]

            x_new[i] = (b[i] - soma) / A[i, i]

        Epest = funcEpest(x_new, x_old)
        x_old = x_new.copy()
        k += 1

    if Epest > Eppara:
        raise RuntimeError("Jacobi não convergiu")

    return x_new, Epest, k


def gaussSeidel(A, b, Eppara, maxit):
    n = len(b)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    Epest = 100
    k = 0

    while Epest > Eppara and k < maxit:
        for i in range(n):
            soma_A = 0
            soma_B = 0

            for j in range(n):
                if j < i:
                    soma_A += A[i, j] * x_new[j]
                elif j > i:
                    soma_B += A[i, j] * x_old[j]

            x_new[i] = (b[i] - soma_A - soma_B) / A[i, i]

        Epest = funcEpest(x_new, x_old)
        x_old = x_new.copy()
        k += 1

    if Epest > Eppara:
        raise RuntimeError("Gauss-Seidel não convergiu")

    return x_new, Epest, k


def gaussSeidelRelaxamento(A, b, Eppara, maxit, lambida):
    n = len(b)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    Epest = 100
    k = 0

    while Epest > Eppara and k < maxit:
        for i in range(n):
            soma_A = 0
            soma_B = 0

            for j in range(n):
                if j < i:
                    soma_A += A[i, j] * x_new[j]
                elif j > i:
                    soma_B += A[i, j] * x_old[j]

            valor = (b[i] - soma_A - soma_B) / A[i, i]
            x_new[i] = lambida * valor + (1 - lambida) * x_old[i]

        Epest = funcEpest(x_new, x_old)
        x_old = x_new.copy()
        k += 1

    if Epest > Eppara:
        raise RuntimeError("SOR não convergiu")

    return x_new, Epest, k


def tdma(a, b, c, d):
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    Aum = np.hstack((A, d.reshape(-1, 1)))
    n = len(d)

    # Eliminação progressiva na matriz aumentada tridiagonal.
    for i in range(n - 1):
        j = i + 1
        fator = Aum[j, i] / Aum[i, i]
        Aum[j, j] = Aum[j, j] - fator * Aum[i, j]
        Aum[j, n] = Aum[j, n] - fator * Aum[i, n]
        Aum[j, i] = 0.0

    # Substituição regressiva.
    x = np.zeros(n)
    x[n - 1] = Aum[n - 1, n] / Aum[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        soma = 0
        j = i + 1
        soma += Aum[i, j] * x[j]
        x[i] = (Aum[i, n] - soma) / Aum[i, i]

    return x


# -------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# -------------------------------------------------------------------
y, dx, a, b, c, d = montar_sistema(N, fluxo_saida)
T_exata = temperatura_exata(y, fluxo_saida)

A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
b_sistema = d.copy()

metodos = ["Jacobi", "Gauss-Seidel", "SOR", "TDMA"]
iteracoes = []
tempos = []
erros_max = []

for nome in metodos:
    inicio = perf_counter()
    if nome == "Jacobi":
        T, Epest, nit = jacobi(A, b_sistema, Eppara, maxit)
    elif nome == "Gauss-Seidel":
        T, Epest, nit = gaussSeidel(A, b_sistema, Eppara, maxit)
    elif nome == "SOR":
        T, Epest, nit = gaussSeidelRelaxamento(A, b_sistema, Eppara, maxit, lambida)
    else:
        T = tdma(a, b, c, d)
        nit = 1

    tempos.append(perf_counter() - inicio)
    iteracoes.append(nit)
    erros_max.append(np.max(np.abs(T - T_exata)))
    print(f"{nome:13s} | iterações: {nit:6d} | erro máx.: {erros_max[-1]:.6e} C")

T_num = tdma(a, b, c, d)  # usada nos gráficos e na convergência

Ept = 100 * np.mean(np.abs((T_exata - T_num) / T_exata))
print(f"\nEpt (N={N}): {Ept:.8e}")
print(f"Re = {rho * U * H / mu:.6g}")
print(f"Geração integrada exata = {q_total:.6f} W/m2")
print(f"Fluxo prescrito em y=H = {fluxo_saida:.6f} W/m2")

# Balanço energético da malha
fluxo_esquerda = 2.0 * k * (T_num[0] - T_parede) / dx
calor_gerado = np.sum(geracao(y)) * dx
print(f"Balanço: q_esquerda + q_direita = {fluxo_esquerda + fluxo_saida:.6f} W/m2")
print(f"         soma da geração nas células = {calor_gerado:.6f} W/m2")

# -------------------------------------------------------------------
# ESTUDO DE CONVERGÊNCIA DE MALHA (TDMA)
# -------------------------------------------------------------------
dxs = []
erros_L2 = []
erros_Ept = []

for n in malhas:
    yn, h, an, bn, cn, dn = montar_sistema(n, fluxo_saida)
    Tn = tdma(an, bn, cn, dn)
    Tex = temperatura_exata(yn, fluxo_saida)
    L2 = np.sqrt(np.mean((Tn - Tex) ** 2))
    Ept_n = 100 * np.mean(np.abs((Tex - Tn) / Tex))
    dxs.append(h)
    erros_L2.append(L2)
    erros_Ept.append(Ept_n)
    print(f"Malha N={n:3d} | dx={h:.4e} m | L2={L2:.6e} C | Ept={Ept_n:.6e}")

ordem = np.polyfit(np.log(dxs), np.log(erros_L2), 1)[0]
print(f"\nOrdem de convergência observada: {ordem:.4f}")

# -------------------------------------------------------------------
# ARQUIVOS .DAT E GRÁFICOS
# -------------------------------------------------------------------
y_liso = np.linspace(0.0, H, 400)

np.savetxt(
    pasta / "dats/escoamento.dat",
    np.column_stack((y_liso * 1e3, velocidade(y_liso), geracao(y_liso))),
    header="y_mm u_m_s geracao_W_m3",
    comments="",
    fmt="%.12e",
)

np.savetxt(
    pasta / "dats/temperatura_exata.dat",
    np.column_stack((y_liso * 1e3, temperatura_exata(y_liso, fluxo_saida))),
    header="y_mm T_exata_C",
    comments="",
    fmt="%.12e",
)

np.savetxt(
    pasta / "dats/temperatura.dat",
    np.column_stack((y * 1e3, T_exata, T_num, np.abs(T_exata - T_num))),
    header="y_mm T_exata_C T_TDMA_C erro_absoluto_C",
    comments="",
    fmt="%.12e",
)

# Efeito de variar o fluxo retirado pela placa superior
fluxos = q_total * np.array([0.0, 0.25, 0.50, 0.75, 1.0])
perfis = []
for fluxo in fluxos:
    yy, hh, aa, bb, cc, dd = montar_sistema(N, fluxo)
    perfis.append(tdma(aa, bb, cc, dd))
np.savetxt(
    pasta / "dats/resfriamento.dat",
    np.column_stack((y * 1e3, *perfis)),
    header="y_mm q0 q25 q50 q75 q100",
    comments="",
    fmt="%.12e",
)

np.savetxt(
    pasta / "dats/convergencia.dat",
    np.column_stack((malhas, dxs, erros_L2, erros_Ept)),
    header="N dx_m erro_L2_C Ept",
    comments="",
    fmt="%.12e",
)

with open(pasta / "dats/metodos.dat", "w", encoding="utf-8") as arq:
    arq.write("metodo iteracoes tempo_s erro_maximo_C\n")
    for nome, nit, tempo, erro in zip(metodos, iteracoes, tempos, erros_max):
        arq.write(f"{nome} {nit} {tempo:.12e} {erro:.12e}\n")

# Velocidade e geração viscosa
plt.figure()
plt.plot(y_liso * 1e3, velocidade(y_liso))
plt.xlabel(r"$y$ ($\mathrm{mm}$)")
plt.ylabel(r"Velocidade ($\mathrm{m}/\mathrm{s}$)")
plt.title("Perfil de Poiseuille")
plt.grid()
plt.tight_layout()
# plt.savefig(pasta / "figuras/velocidade.png", dpi=200)

plt.figure()
plt.plot(y_liso * 1e3, geracao(y_liso))
plt.xlabel(r"$y$ ($\mathrm{mm}$)")
plt.ylabel(r"Geração de calor ($\mathrm{W}/\mathrm{m}^{3}$)")
plt.title("Dissipação viscosa")
plt.grid()
plt.tight_layout()
# plt.savefig(pasta / "figuras/geracao.png", dpi=200)

plt.figure()
plt.plot(y_liso * 1e3, temperatura_exata(y_liso, fluxo_saida), "-k", label="Analítica")
plt.plot(y * 1e3, T_num, "o", label=f"Diferenças finitas (N={N})")
plt.xlabel(r"$y$ ($\mathrm{mm}$)")
plt.ylabel(r"Temperatura (${}^{\circ}\mathrm{C}$)")
plt.title(r"Solução analítica $\times$ numérica")
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig(pasta / "figuras/temperatura.png", dpi=200)

plt.figure()
for j, perfil in enumerate(perfis):
    plt.plot(y * 1e3, perfil, "-o", markersize=3, label=f"{100*fluxos[j]/q_total:.0f}%")
plt.xlabel(r"$y$ ($\mathrm{mm}$)")
plt.ylabel(r"Temperatura (${}^{\circ}\mathrm{C}$)")
plt.title("Efeito do fluxo de calor na placa superior")
plt.legend(title="Fração de calor retirado")
plt.grid()
plt.tight_layout()
# plt.savefig(pasta / "figuras/resfriamento.png", dpi=200)

plt.figure()
plt.loglog(dxs, erros_L2, "-o", label=r"Erro $L_2$")
plt.loglog(
    dxs, erros_L2[0] * (np.array(dxs) / dxs[0]) ** 2, "--", label="Referência: ordem 2"
)
plt.xlabel(r"Tamanho da célula $\Delta x$ ($\mathrm{m}$)")
plt.ylabel(r"Erro $L_2$ (${}^{\circ}\mathrm{C}$)")
plt.title(f"Convergência de malha (ordem = {ordem:.3f})")
plt.legend()
plt.grid(which="both")
plt.tight_layout()
# plt.savefig(pasta / "figuras/convergencia.png", dpi=200)

plt.figure()
plt.bar(metodos, iteracoes)
plt.ylabel("Iterações")
plt.title(f"Comparação dos solucionadores (N={N})")
plt.grid(axis="y")
plt.tight_layout()
# plt.savefig(pasta / "figuras/metodos.png", dpi=200)

plt.show()

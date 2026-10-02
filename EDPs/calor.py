import numpy as np

k = 1.0
L = 10.0
q = 0.0

uw = 10.0
ue = 5.0

n = 5
dx = L / n

A = np.zeros((n, n))
d = np.zeros(n)

A[0, 0] = 4 * k / dx**2
A[0, 1] = -4 * k / (3 * dx**2)
d[0] = q + 8 * k * uw / (3 * dx**2)

for i in range(1, n - 1):
    A[i, i - 1] = -k / dx**2
    A[i, i] = 2 * k / dx**2
    A[i, i + 1] = -k / dx**2

    d[i] = q

A[n - 1, n - 2] = -4 * k / (3 * dx**2)
A[n - 1, n - 1] = 4 * k / dx**2
d[n - 1] = q + 8 * k * ue / (3 * dx**2)

u = np.linalg.solve(A, d)

print("Matriz A:")
print(A)

print("\nVetor d:")
print(d)

print("\nSolução:")
for i in range(n):
    print(f"u{i} = {u[i]:.4f}")

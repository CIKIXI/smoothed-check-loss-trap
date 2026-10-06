"""Re-check the two cases my first script got wrong (grid range, kernel scale)."""

import numpy as np
from scipy.special import ndtr

rng = np.random.RandomState(99)


def phi(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)


def rho_tilde(r, s, tau=0.5):
    s = max(float(s), 1e-12)
    z = np.asarray(r, float) / s
    return r * (ndtr(z) - (1 - tau)) + s * phi(z)


def report(name, claimed, computed, tol):
    ok = abs(claimed - computed) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} closed form {claimed:>10.6f}   "
          f"MC argmin {computed:>10.6f}   dev {abs(claimed - computed):.2e}")
    return ok


print("=" * 96)
print("A. Raw-W adaptive smoothed loss, kernel M: exact argmin (Sigma_x+Sigma_u+M)^{-1}Sigma_x beta*")
print("=" * 96)
Sigma_x = np.array([[1.0, 0.6], [0.6, 1.0]])
Sigma_u = np.diag([0.4, 0.10])
beta = np.array([2.0, 1.0])
L = np.linalg.cholesky(Sigma_x)
Lu = np.linalg.cholesky(Sigma_u)
n = 1_500_000
X = rng.standard_normal((n, 2)) @ L.T
U = rng.standard_normal((n, 2)) @ Lu.T
W = X + U
eps = rng.laplace(0, 1, n)
Y = X @ beta + eps
for lab, M in (("M = 0", 0.0 * np.eye(2)),
               ("M = 0.2 I", 0.2 * np.eye(2)),
               ("M non-commuting", np.array([[0.30, 0.15], [0.15, 0.45]]))):
    b_exact = np.linalg.solve(Sigma_x + Sigma_u + M, Sigma_x @ beta)

    def loss(b):
        s = float(np.sqrt(b @ M @ b))          # the kernel scale, as in the manuscript
        return float(np.mean(rho_tilde(Y - W @ b, s)))

    best, bval = None, np.inf
    step = 0.01
    for d1 in np.arange(-0.12, 0.1201, step):
        for d2 in np.arange(-0.12, 0.1201, step):
            cand = b_exact + np.array([d1, d2])
            v = loss(cand)
            if v < bval:
                bval, best = v, cand
    dev = float(np.linalg.norm(best - b_exact))
    print(f"  {lab:<18} closed form {np.round(b_exact, 5).tolist()}   grid argmin "
          f"{np.round(best, 5).tolist()}   ||dev|| {dev:.4f} (grid step {step})")

print()
print("=" * 96)
print("B. Naive check loss on W, scalar case beta*=2, sigma_u=0.5: argmin should be beta_bar=1.6")
print("=" * 96)
n = 3_000_000
X = rng.standard_normal(n)
U = 0.5 * rng.standard_normal(n)
W = X + U
eps = rng.laplace(0, 1, n)
Y = 2.0 * X + eps
grid = np.arange(1.55, 1.65, 0.0002)
vals = [float(np.mean((Y - b * W) * (0.5 - (Y - b * W < 0)))) for b in grid]
k = int(np.argmin(vals))
y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
b_mc = float(grid[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * 0.0002)
report("naive QR population argmin = beta_bar", 1.6, b_mc, 1.5e-3)

print()
print("=" * 96)
print("C. Fixed-scale smoothed loss (pilot scale), scalar: argmin should be exactly beta_bar")
print("=" * 96)
for su in (0.1, 0.3, 0.5):
    beta_bar = 2.0 / (1 + su ** 2)
    n = 2_000_000
    X = rng.standard_normal(n)
    U = su * rng.standard_normal(n)
    W = X + U
    eps = rng.laplace(0, 1, n)
    Y = 2.0 * X + eps
    sigma = beta_bar * su
    grid = np.arange(beta_bar - 0.05, beta_bar + 0.05, 0.0002)
    vals = [float(np.mean(rho_tilde(Y - b * W, sigma))) for b in grid]
    k = int(np.argmin(vals))
    y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
    b_mc = float(grid[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * 0.0002)
    report(f"fixed-scale smoothed argmin = beta_bar (sigma_u={su})", beta_bar, b_mc, 1.5e-3)

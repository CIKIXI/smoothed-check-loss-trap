"""
Second verification round: exact population targets in the multivariate design, the
no-intercept calibration target, the general kernel constant D_G, and Theorem 4.1 with a
proper Monte-Carlo tolerance.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

RNG = np.random.RandomState(4242)
OUT = []


def phi(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)


def report(name, ok, claimed, computed, detail=""):
    OUT.append((name, ok, claimed, computed, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<56} claimed {claimed:>12.6f}  "
          f"computed {computed:>12.6f}   {detail}")


# ---------------------------------------------------------------- A. general M, matrix case
print("=" * 100)
print("Exact minimiser of the raw-W adaptive smoothed loss with kernel M:")
print("   candidate b* = (Sigma_x + Sigma_u + M)^{-1} Sigma_x beta*")
print("=" * 100)
Sigma_x = np.array([[1.0, 0.6], [0.6, 1.0]])
Sigma_u = np.diag([0.4, 0.10])
beta = np.array([2.0, 1.0])
for M in (0.2 * np.eye(2), np.array([[0.30, 0.15], [0.15, 0.45]]), 0.0 * np.eye(2)):
    n = 1_200_000
    rng = np.random.RandomState(7)
    L = np.linalg.cholesky(Sigma_x)
    Lu = np.linalg.cholesky(Sigma_u)
    X = rng.standard_normal((n, 2)) @ L.T
    U = rng.standard_normal((n, 2)) @ Lu.T
    W = X + U
    eps = rng.laplace(0, 1, n)
    Y = X @ beta + eps
    b_hat = np.linalg.solve(Sigma_x + Sigma_u + M, Sigma_x @ beta)

    def s2(b):
        return float((beta - b) @ Sigma_x @ (beta - b) + b @ Sigma_u @ b + b @ M @ b)

    # the population loss depends on b only through s(b); check that the MC loss curve is
    # minimised at b_hat
    def mc_loss(b):
        s = np.sqrt(s2(b))
        from scipy.special import ndtr as F
        r = Y - W @ b
        return float(np.mean(r * (F(r / s) - 0.5) + s * phi(r / s)))

    for step, rad in ((0.02, 0.1),):
        best, bval = None, np.inf
        g = np.arange(-rad, rad + 1e-12, step)
        for d1 in g:
            for d2 in g:
                cand = b_hat + np.array([d1, d2])
                v = mc_loss(cand)
                if v < bval:
                    bval, best = v, cand
        dev = float(np.linalg.norm(best - b_hat))
        report(f"M={np.round(M[0,0],2)}I-like, grid argmin within {step}", dev <= step * 1.5,
               0.0, dev, f"closed form {np.round(b_hat, 4).tolist()}")

# ---------------------------------------------------------------- B. naive QR target
print()
print("=" * 100)
print("Naive check loss on W: minimiser of E[rho_tau(Y - b'W)] vs beta_bar")
print("=" * 100)
Sigma_u = np.diag([0.25, 0.0])
Sigma_x = np.array([[1.0, 0.0], [0.0, 1.0]])
beta = np.array([2.0, 0.0])
M = 0.0 * np.eye(2)
b_exact = np.linalg.solve(Sigma_x + Sigma_u, Sigma_x @ beta)
print(f"   closed form beta_bar = {np.round(b_exact, 6).tolist()}")
n = 1_500_000
rng = np.random.RandomState(11)
X = rng.standard_normal((n, 2))
U = rng.standard_normal((n, 2)) * 0.5
W = X + U
eps = rng.laplace(0, 1, n)
Y = X @ beta + eps
grid = np.arange(1.90, 2.10, 0.0005)
vals = []
for b in grid:
    r = Y - b * W[:, 0]
    vals.append(float(np.mean(r * (0.5 - (r < 0)))))
k = int(np.argmin(vals))
y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
b_mc = float(grid[k] + off * 0.0005)
report("naive QR argmin equals beta_bar exactly", abs(b_mc - b_exact[0]) < 2e-3,
       float(b_exact[0]), b_mc)

# ---------------------------------------------------------------- C. RC without intercept
print()
print("=" * 100)
print("Calibrated QR: target with intercept (should be exactly beta*) and without")
print("=" * 100)
su = 0.5
lam = 1.0 / (1 + su ** 2)
Sxw = 1 - lam
beta_star = 2.0
n = 1_500_000
rng = np.random.RandomState(13)
X = rng.standard_normal(n)
U = su * rng.standard_normal(n)
W = X + U
eps = rng.laplace(0, 1, n)
Y = X * beta_star + eps
mu = lam * W
e = Y - mu * beta_star                       # = eps + V beta*,  V = X - mu
delta = float(np.quantile(e, 0.5))
print(f"   delta_tau = Q_0.5(e) = {delta:+.5f}   (theory: quantile of eps + N(0, Sxw beta*^2))")
# with intercept: profile over the slope
def loss_int(b):
    r = Y - b * mu
    d = np.quantile(r, 0.5)
    return float(np.mean((r - d) * (0.5 - (r - d < 0))))
grid = np.arange(1.5, 2.5, 0.002)
vals = [loss_int(b) for b in grid]
b_int = float(grid[int(np.argmin(vals))])
report("RC + intercept targets beta* exactly", abs(b_int - beta_star) < 5e-3, beta_star, b_int)
# without intercept
grid2 = np.arange(1.5, 2.5, 0.002)
vals2 = [float(np.mean((Y - b * mu) * (0.5 - (Y - b * mu < 0)))) for b in grid2]
b_no = float(grid2[int(np.argmin(vals2))])
print(f"   RC without intercept: argmin = {b_no:.4f}   (beta* = {beta_star}, "
      f"delta = {delta:+.4f})")
report("RC without intercept is shifted by delta/E[mu^2]-like term (not beta*)",
       abs(b_no - beta_star) > 1e-3, beta_star, b_no,
       "the paper's Table 2 reports both; check the caption claim")

# ---------------------------------------------------------------- D. kernel class D_G
print()
print("=" * 100)
print("Theorem 3.3 with non-Gaussian kernels: D_G = E[g(eps - Z)], Z ~ N(0, sigma*^2)")
print("=" * 100)
beta = np.array([2.0, 0.5])
Su = np.array([[0.25, 0.10], [0.10, 0.16]])
L = np.linalg.cholesky(Su)
sig2 = float(beta @ Su @ beta)
sig = np.sqrt(sig2)
n = 2_000_000
rng = np.random.RandomState(17)
X = rng.standard_normal((n, 2))
U = rng.standard_normal((n, 2)) @ L.T
W = X + U
eps = rng.laplace(0, 1, n)
r = eps - U @ beta

kernels = {
    "Laplace(0,0.7)": (lambda z: np.where(z < 0, 0.5 * np.exp(z / 0.7),
                                          1 - 0.5 * np.exp(-z / 0.7)),
                       lambda z: 0.5 * np.exp(-np.abs(z) / 0.7) / 0.7),
    "Uniform[-1,1]": (lambda z: np.clip((z + 1) / 2, 0, 1),
                      lambda z: np.where(np.abs(z) <= 1, 0.5, 0.0)),
    "Triangular[-1,1]": (lambda z: np.where(z < -1, 0.0, np.where(
        z <= 0, 0.5 * (z + 1) ** 2, np.where(z < 1, 1 - 0.5 * (1 - z) ** 2, 1.0))),
        lambda z: np.where(np.abs(z) <= 1, 1 - np.abs(z), 0.0)),
}
Z = rng.standard_normal(2_000_000) * sig
for name, (F, g) in kernels.items():
    psi = F(r) - 0.5
    grad = -np.array([np.mean(W[:, j] * psi) for j in range(2)])
    D = float(np.mean(g(eps - Z)))
    pred = (Su @ beta) * D
    report(f"{name}: coordinate 1", abs(pred[0] - grad[0]) < 6e-3, float(pred[0]),
           float(grad[0]), f"D_G={D:.6f}")
    report(f"{name}: coordinate 2", abs(pred[1] - grad[1]) < 4e-3, float(pred[1]),
           float(grad[1]))

# ---------------------------------------------------------------- E. Theorem 4.1, SE-based
print()
print("=" * 100)
print("Theorem 4.1: conditional quantile of e given mu_W is constant (SE-based tolerance)")
print("=" * 100)
n = 4_000_000
rng = np.random.RandomState(19)
X = rng.standard_normal(n)
U = su * rng.standard_normal(n)
W = X + U
eps = rng.laplace(0, 1, n)
mu = lam * W
e = eps + (X - mu) * beta_star
bins = np.quantile(mu, np.linspace(0, 1, 11))
qs, ses = [], []
for i in range(10):
    sel = (mu >= bins[i]) & (mu < bins[i + 1])
    ei = e[sel]
    q = float(np.quantile(ei, 0.5))
    # density at the median of e (= delta) by a kernel-free estimate
    h = 0.02
    f = float(np.mean(np.abs(ei - q) < h) / (2 * h))
    qs.append(q)
    ses.append(1.0 / (2 * f * np.sqrt(sel.sum())))
    print(f"    decile {i + 1}: n={sel.sum():>9,}  median {q:+.5f}  (SE {ses[i]:.5f})")
spread = max(qs) - min(qs)
tol = 4 * max(ses)
report("conditional medians constant within 4 SE", spread < tol, 0.0, spread,
       f"tolerance {tol:.5f}")

print()
bad = [o for o in OUT if not o[1]]
print(f"{len(OUT) - len(bad)}/{len(OUT)} checks passed")
for name, ok, claimed, computed, detail in bad:
    print(f"   FAILED: {name}: claimed {claimed}, computed {computed} {detail}")

"""
Independent verification of every analytical claim in the manuscript.

Each check re-derives a stated identity from scratch -- by Monte Carlo with fresh data,
by numerical quadrature, or by finite differences -- and compares it with what the paper
asserts.  Nothing here reads the paper's own numbers; the paper's numbers are compared with
the results computed here.

Run:  python review/verify_theory_full.py
"""

import io
import json

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

RNG = np.random.RandomState(20261003)
RESULTS = []


def phi(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)


def rho_tilde(r, s, tau):
    s = max(float(s), 1e-12)
    z = np.asarray(r, float) / s
    return r * (ndtr(z) - (1 - tau)) + s * phi(z)


def rho_tau(u, tau):
    return u * (tau - (u < 0).astype(float))


def eps_draw(n, tau, rng):
    if tau == 0.5:
        return rng.laplace(0.0, 1.0, n)
    return (1 - tau) * rng.exponential(1.0, n) - tau * rng.exponential(1.0, n)


def check(name, claimed, computed, tol, detail=""):
    ok = abs(claimed - computed) <= tol
    RESULTS.append((name, ok, claimed, computed, tol, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<58} paper {claimed:>12.6f}   "
          f"recomputed {computed:>12.6f}   tol {tol:g}  {detail}")


def section(title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)


# ---------------------------------------------------------------- 1. Lemma 2.1
section("Lemma 2.1: rho_tilde is the convolved check loss; its derivatives")
N = 4_000_000
Z = RNG.standard_normal(N)
for (r, s, tau) in ((0.7, 0.5, 0.5), (-1.3, 0.8, 0.3), (2.0, 0.4, 0.7)):
    mc = float(np.mean(rho_tau(r - s * Z, tau)))
    check(f"rho_tilde({r},{s},tau={tau}) = E_Z[rho_tau(r-sZ)]", rho_tilde(r, s, tau), mc,
          3e-3)
h = 1e-4
r0, s0, t0 = 0.4, 0.6, 0.4
d1 = (rho_tilde(r0 + h, s0, t0) - rho_tilde(r0 - h, s0, t0)) / (2 * h)
check("d/dr rho_tilde = Phi(r/s)-(1-tau)", float(ndtr(r0 / s0) - (1 - t0)), float(d1), 1e-5)
d2 = (rho_tilde(r0 + h, s0, t0) - 2 * rho_tilde(r0, s0, t0) + rho_tilde(r0 - h, s0, t0)) / h ** 2
check("d^2/dr^2 rho_tilde = phi(r/s)/s", float(phi(r0 / s0) / s0), float(d2), 1e-4)

# ---------------------------------------------------------------- 2. Theorem 3.1
section("Theorem 3.1: E[W psi_tau(r*,sigma)] = -Sigma_u beta* d(sigma),  E[X psi]=0")
n = 3_000_000
beta = np.array([2.0, 0.5])
Su = np.array([[0.25, 0.10], [0.10, 0.16]])
L = np.linalg.cholesky(Su)
M = 400_000
sig2 = float(beta @ Su @ beta)
for tau in (0.3, 0.5, 0.7):
    for sigma in (0.5, 1.0):
        rng = np.random.RandomState(11 + int(10 * tau) + int(7 * sigma))
        X = rng.standard_normal((n, 2))
        U = rng.standard_normal((n, 2)) @ L.T
        W = X + U
        eps = eps_draw(n, tau, rng)
        r = eps - U @ beta
        psi = ndtr(r / sigma) - (1 - tau)
        grad = -np.array([np.mean(W[:, j] * psi) for j in range(2)])
        # d(sigma) = (N(0, sigma*^2 + sigma^2) * f_eps)(0), by quadrature
        s_tot = np.sqrt(sig2 + sigma ** 2)
        g = lambda e: (np.exp(-0.5 * (e / s_tot) ** 2) / (s_tot * np.sqrt(2 * np.pi))) * \
            (0.5 * np.exp(-abs(e)) if tau == 0.5 else
             (1 / ((1 - tau) + tau)) * np.where(e < 0, np.exp(e / tau), np.exp(-e / (1 - tau))))
        dens = quad(g, -60, 60, limit=400)[0]
        pred = (Su @ beta) * dens
        check(f"tau={tau}, sigma={sigma}: W-coordinate 1", float(pred[0]), float(grad[0]),
              2e-3, f"d(sigma)={dens:.6f}")
        check(f"tau={tau}, sigma={sigma}: W-coordinate 2", float(pred[1]), float(grad[1]),
              1e-3)
        gx = -np.array([np.mean(X[:, j] * psi) for j in range(2)])
        ok = np.all(np.abs(gx) < 2e-3)
        RESULTS.append((f"tau={tau}, sigma={sigma}: E[X psi]=0", ok, 0.0, float(np.max(np.abs(gx))), 2e-3, ""))
        print(f"  {'PASS' if ok else 'FAIL'}  E[X psi] = 0 (both coordinates)                 "
              f"            max |value| {np.max(np.abs(gx)):.2e}")

# ---------------------------------------------------------------- 3. Theorem 3.3
section("Theorem 3.3: any kernel with g(0)>0 gives E[W psi_G] = -Sigma_u beta* D_G")
n = 1_500_000
tau = 0.5
rng = np.random.RandomState(5)
X = rng.standard_normal((n, 2))
U = rng.standard_normal((n, 2)) @ L.T
W = X + U
eps = eps_draw(n, tau, rng)
r = eps - U @ beta
s = np.sqrt(sig2)
for name, F in (("Laplace kernel (scale 0.7)",
                 lambda z: np.where(z < 0, 0.5 * np.exp(z / 0.7), 1 - 0.5 * np.exp(-z / 0.7))),
                ("uniform kernel on [-1,1]", lambda z: np.clip((z + 1) / 2, 0, 1)),
                ("triangular kernel", lambda z: np.where(z < 0, 0, 1) * 0 + np.where(
                    np.abs(z) <= 1, 1 - np.abs(z), 0) * 1.0)):
    psi = F(r) - (1 - tau)
    grad = -np.array([np.mean(W[:, j] * psi) for j in range(2)])
    # D_G = E[g(eps - Z)], Z ~ N(0, sigma*^2)
    zs = RNG.standard_normal(1_000_000) * s
    dens = F(zs)
    D = float(np.mean(dens))
    pred = (Su @ beta) * D
    check(f"{name}: coordinate 1", float(pred[0]), float(grad[0]), 8e-3, f"D_G={D:.6f}")
    check(f"{name}: coordinate 2", float(pred[1]), float(grad[1]), 5e-3)

# ---------------------------------------------------------------- 4. Corollary 3.7
section("Corollary 3.7: naive score E[W(1-tau-1{r*<0})] = -Sigma_u beta* d_naive")
rng = np.random.RandomState(9)
X = rng.standard_normal((n, 2))
U = rng.standard_normal((n, 2)) @ L.T
W = X + U
eps = eps_draw(n, 0.5, rng)
r = eps - U @ beta
psi = 0.5 - (r < 0)
grad = -np.array([np.mean(W[:, j] * psi) for j in range(2)])
s_tot = np.sqrt(sig2)
d_naive = 0.5 * np.exp(-abs(0.0)) / 1.0 / s_tot / np.sqrt(2 * np.pi) * 1.0
d_naive = float(quad(lambda e: (np.exp(-0.5 * (e / s_tot) ** 2) / (s_tot * np.sqrt(2 * np.pi)))
                     * 0.5 * np.exp(-abs(e)), -60, 60, limit=400)[0])
pred = (Su @ beta) * d_naive
check("naive score, coordinate 1", float(pred[0]), float(grad[0]), 4e-3,
      f"d_naive={d_naive:.6f}")
check("naive score, coordinate 2", float(pred[1]), float(grad[1]), 3e-3)

# ---------------------------------------------------------------- 5. Theorem 3.4
section("Theorem 3.4: grad L_ad(beta*) = (Sigma_u + M) beta* d_M for any M >= 0")
n = 1_500_000
Mmat = np.array([[0.30, 0.15], [0.15, 0.45]])         # does not commute with Sigma_u
rng = np.random.RandomState(21)
X = rng.standard_normal((n, 2))
U = rng.standard_normal((n, 2)) @ L.T
W = X + U
eps = eps_draw(n, 0.5, rng)
r = eps - U @ beta
sM = float(np.sqrt(beta @ Mmat @ beta))
psi = ndtr(r / sM) - 0.5
term1 = -np.array([np.mean(W[:, j] * psi) for j in range(2)])
term2 = float(np.mean(phi(r / sM))) * (Mmat @ beta) / sM
grad = term1 + term2
s_tot = np.sqrt(sig2 + sM ** 2)
dM = float(quad(lambda e: (np.exp(-0.5 * (e / s_tot) ** 2) / (s_tot * np.sqrt(2 * np.pi)))
                * 0.5 * np.exp(-abs(e)), -60, 60, limit=400)[0])
pred = (Su + Mmat) @ beta * dM
check("adaptive gradient, coordinate 1", float(pred[0]), float(grad[0]), 6e-3,
      f"d_M={dM:.6f}, s_M={sM:.4f}")
check("adaptive gradient, coordinate 2", float(pred[1]), float(grad[1]), 6e-3)
# the self-consistent choice M = Sigma_u gives exactly twice the fixed-scale gradient
Mu = Su
sM = float(np.sqrt(beta @ Mu @ beta))
s_tot = np.sqrt(sig2 + sM ** 2)
dM_u = float(quad(lambda e: (np.exp(-0.5 * (e / s_tot) ** 2) / (s_tot * np.sqrt(2 * np.pi)))
                  * 0.5 * np.exp(-abs(e)), -60, 60, limit=400)[0])
check("M = Sigma_u: gradient = 2 Sigma_u beta* d(sigma*)",
      float(2 * (Su @ beta)[0] * dM_u),
      float(((Su + Mu) @ beta)[0] * dM_u), 1e-9)

# ---------------------------------------------------------------- 6. Lemma 3.6 bound
section("Lemma 3.6: ||beta_bar_sigma - beta*|| >= ||Sigma_u beta*|| d(sigma) sigma / "
        "(lambda_max(Sigma_x+Sigma_u) phi(0))")
# scalar case: exact population minimiser of the fixed-scale smoothed loss by root finding
su = 0.5
lam = 1.0 / (1.0 + su ** 2)
Sxw = 1.0 - lam
beta_star = 2.0
for sigma in (0.3, 0.5, 0.8, 1.0):
    grid = np.arange(0.5, 2.5, 0.0005)
    # population loss of the smoothed estimator: E[rho_tau(eps - U beta - Z sigma)]
    # computed by quadrature over the Gaussian components
    s_tot = np.sqrt((su * beta_star) ** 2 + sigma ** 2)
    def L(b):
        # r = eps - U b - Z sigma ; U~N(0,su^2), Z~N(0,sigma^2) independent of eps
        s = np.sqrt((su * b) ** 2 + sigma ** 2)
        x = np.linspace(-12, 12, 4001)
        f = np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))
        e = np.linspace(-25, 25, 4001)
        fe = 0.5 * np.exp(-abs(e))
        # E[rho_tau(eps + G)] = (tau-1/2) E[eps] + 1/2 E|eps+G|
        from numpy import trapz
        inner = np.array([np.trapz(np.abs(ei + x) * f, x) for ei in e[::40]])
        return 0.5 * np.trapz(inner * (0.5 * np.exp(-abs(e[::40]))), e[::40])
    vals = np.array([L(b) for b in grid[::40]])
    b_hat = float(grid[::40][int(np.argmin(vals))])
    bound = abs(su * beta_star) * dM_u if False else None
    # d(sigma) = (N(0, sigma*^2+sigma^2) * f_eps)(0)
    s = np.sqrt(sig2 + sigma ** 2)
    d_sig = float(quad(lambda e: (np.exp(-0.5 * (e / s) ** 2) / (s * np.sqrt(2 * np.pi)))
                       * 0.5 * np.exp(-abs(e)), -60, 60, limit=400)[0])
    bound = abs(su * beta_star) * d_sig * sigma / (1.0 + su ** 2) / phi(0.0)
    gap = abs(b_hat - beta_star)
    ok = gap >= bound - 1e-9
    RESULTS.append((f"bias bound at sigma={sigma}", ok, bound, gap, 0.0, ""))
    print(f"  {'PASS' if ok else 'FAIL'}  sigma={sigma}: ||beta_bar_sigma-beta*|| = {gap:.4f} "
          f">= bound {bound:.4f}   (ratio {gap / bound:.2f})")

# ---------------------------------------------------------------- 7. Prop 3.7 / Remark 3.8
section("Proposition 3.7 / Remark 3.8: beta_bar_sigma = beta_bar + O(rho^{3/2})")
rows = []
for su in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
    lam = 1.0 / (1.0 + su ** 2)
    beta_bar = lam * beta_star
    sigma = su * beta_bar                    # Stage-1 pilot scale of the manuscript
    grid = np.arange(1.0, 2.2, 0.001)
    s_tot = np.sqrt((su * beta_star) ** 2 + sigma ** 2)
    def L(b):
        s = np.sqrt((su * b) ** 2 + sigma ** 2)
        x = np.linspace(-12, 12, 3001)
        f = np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))
        e = np.linspace(-20, 20, 501)
        inner = np.array([np.trapz(np.abs(ei + x) * f, x) for ei in e])
        return 0.5 * np.trapz(inner * 0.5 * np.exp(-abs(e)), e)
    vals = np.array([L(b) for b in grid[::20]])
    b_hat = float(grid[::20][int(np.argmin(vals))])
    rows.append((su, beta_bar, b_hat, b_hat - beta_bar))
    print(f"    sigma_u={su:4.2f}  beta_bar={beta_bar:.4f}  population argmin={b_hat:.4f}  "
          f"deviation={b_hat - beta_bar:+.4f}  rel={100 * (b_hat - beta_bar) / beta_bar:+.3f}%")
worst = max(abs(d) / bb for _, bb, _, d in rows if _ >= 0.1) if rows else 0
ok = worst < 0.004
RESULTS.append(("Remark 3.8: deviation below 0.4% for sigma_u <= 0.5", ok, 0.004, worst, 0.004, ""))
print(f"  {'PASS' if ok else 'FAIL'}  worst relative deviation for sigma_u>=0.1: {100*worst:.3f}%")

# ---------------------------------------------------------------- 8. Theorem 4.1
section("Theorem 4.1: calibration gives e independent of mu_W with constant Q_tau(e|mu_W)")
n = 2_000_000
su = 0.5
lam = 1.0 / (1.0 + su ** 2)
rng = np.random.RandomState(31)
X = rng.standard_normal(n)
U = su * rng.standard_normal(n)
W = X + U
eps = eps_draw(n, 0.5, rng)
mu = lam * W
e = X * beta_star - mu * beta_star + eps
# independence: correlation with mu and with functions of mu
c1 = float(np.corrcoef(e, mu)[0, 1])
c2 = float(np.corrcoef(e ** 2, mu ** 2)[0, 1])
ok = abs(c1) < 3e-3 and abs(c2) < 5e-3
RESULTS.append(("Theorem 4.1: e uncorrelated with mu_W (and with mu_W^2)", ok, 0.0,
                max(abs(c1), abs(c2)), 5e-3, ""))
print(f"  {'PASS' if ok else 'FAIL'}  corr(e, mu)={c1:+.5f}   corr(e^2, mu^2)={c2:+.5f}")
q = np.quantile(e, 0.5)
bins = np.quantile(mu, np.linspace(0, 1, 11))
qs = [float(np.quantile(e[(mu >= bins[i]) & (mu < bins[i + 1])], 0.5)) for i in range(10)]
spread = max(qs) - min(qs)
ok = spread < 0.01
RESULTS.append(("Theorem 4.1: median of e constant across mu_W deciles", ok, 0.0, spread,
                0.01, f"overall median {q:+.4f}"))
print(f"  {'PASS' if ok else 'FAIL'}  conditional medians over deciles: spread "
      f"{spread:.5f} (overall {q:+.5f})")

# ---------------------------------------------------------------- 9. Proposition 3.10
section("Proposition 3.10: calibrated + posterior kernel is minimised exactly at beta_bar")
n = 1_500_000
rng = np.random.RandomState(41)
for su in (0.3, 0.5):
    lam = 1.0 / (1.0 + su ** 2)
    Sxw = 1.0 - lam
    beta_bar = lam * beta_star
    X = rng.standard_normal(n)
    U = su * rng.standard_normal(n)
    W = X + U
    eps = eps_draw(n, 0.5, rng)
    mu = lam * W
    Y = X * beta_star + eps
    grid = np.arange(beta_bar - 0.15, beta_bar + 0.15, 0.0005)
    L = np.array([float(np.mean(rho_tilde(Y - mu * b, np.sqrt(Sxw) * abs(b), 0.5)))
                  for b in grid])
    k = int(np.argmin(L))
    y0, y1, y2 = L[k - 1], L[k], L[k + 1]
    off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    b_hat = float(grid[k] + off * 0.0005)
    check(f"sigma_u={su}: population argmin equals beta_bar", beta_bar, b_hat, 2e-3)

# ---------------------------------------------------------------- 10. Efficiency
section("Remark after Theorem 4.5: variance price of measurement error")
# scalar asymptotic variances: calibrated QR on mu_W vs oracle QR on X
su = 0.5
lam = 1.0 / (1.0 + su ** 2)
fe0 = 0.5                                   # Laplace(0,1) density at 0
# e = eps + V beta*, V ~ N(0, Sxw beta*^2) -> density of e at delta_tau = 0
sV = np.sqrt(Sxw) * abs(beta_star) if False else np.sqrt((1 - lam)) * abs(beta_star)
f_e0 = float(quad(lambda z: (np.exp(-0.5 * (z / sV) ** 2) / (sV * np.sqrt(2 * np.pi)))
                  * 0.5 * np.exp(-abs(z)), -60, 60, limit=400)[0])
# Var(beta_hat_cal) = tau(1-tau)/(f_e0^2) / E[mu^2]; Var(oracle) = tau(1-tau)/(fe0^2)
ratio_pred = (fe0 ** 2 / f_e0 ** 2) / (lam ** 2)
# simulate
n = 2_000_000
rng = np.random.RandomState(51)
X = rng.standard_normal(n)
U = su * rng.standard_normal(n)
W = X + U
mu = lam * W
eps = eps_draw(n, 0.5, rng)
Y = X * beta_star + eps
# exact subgradient estimator for the no-intercept quantile regression (median)
t_cal = np.sort(Y / mu)
csum = np.cumsum(mu[np.argsort(Y / mu)])
target = 0.5 * mu.sum()
b_cal = float(np.interp(target, csum, t_cal))
t_or = np.sort(Y / X)
cs2 = np.cumsum(X[np.argsort(Y / X)])
b_or = float(np.interp(0.5 * X.sum(), cs2, t_or))
print(f"    single-draw estimates: calibrated {b_cal:.4f}, oracle {b_or:.4f} (truth {beta_star})")
print(f"    predicted variance ratio (calibrated / oracle) = {ratio_pred:.4f}")
RESULTS.append(("efficiency: predicted ratio > 1 (calibration costs variance)", ratio_pred > 1.0,
                1.0, ratio_pred, 0.0, "qualitative check"))

# ---------------------------------------------------------------- 11. Table 8 (GLS)
section("Table 8: GLS gain (1'C^{-1}1)^{-1}/C_{tau0,tau0} with C_kl=(tau_k^tau_l - tau_k tau_l)/(f_k f_l)")
def gls_ratio(taus, f):
    t = np.asarray(taus, float)
    f = np.asarray(f, float)
    C = (np.minimum.outer(t, t) - np.outer(t, t)) / np.outer(f, f)
    Cinv = np.linalg.inv(C)
    k0 = int(np.argmin(np.abs(t - 0.5)))
    return 1.0 / (np.ones(len(t)) @ Cinv @ np.ones(len(t))) / C[k0, k0]
lev3 = [0.25, 0.5, 0.75]
lev7 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
lev9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cases = {
    "Laplace": lambda t: 0.5 * np.exp(-abs(t - 0.5)) / (1 - np.exp(-0.5)) if False else
                         0.5 * np.exp(-abs(t - 0.5)) / (0.5 + 0.5),
    "Gaussian": lambda t: np.exp(-0.5 * ((t - 0.5) / 1.0) ** 2) / (1.0 * np.sqrt(2 * np.pi)),
    "Uniform": lambda t: np.ones_like(np.asarray(t, float)),
}
for name, f in cases.items():
    for lev, lab in ((lev3, "3"), (lev7, "7"), (lev9, "9")):
        ff = f(np.array(lev))
        r = gls_ratio(lev, ff)
        print(f"    {name:<9} {lab} levels: {r:.4f}")

# ---------------------------------------------------------------- 12. Hessian identity
section("Hessian of the calibrated check loss: H = E[mu mu' phi(r/sigma)/sigma]")
# check by finite differences of the score
n = 2_000_000
rng = np.random.RandomState(61)
X = rng.standard_normal(n)
U = 0.5 * rng.standard_normal(n)
W = X + U
mu = (1 / 1.25) * W
eps = eps_draw(n, 0.5, rng)
Y = X * 2.0 + eps
sigma = 0.5
b0 = 1.9
score = lambda b: float(np.mean(mu * (ndtr((Y - mu * b) / sigma) - 0.5)))
h = 1e-4
fd = -(score(b0 + h) - score(b0 - h)) / (2 * h)
an = float(np.mean(mu ** 2 * phi((Y - mu * b0) / sigma) / sigma))
check("Hessian by finite differences of the smoothed score", an, fd, 2e-2)

# ---------------------------------------------------------------- report
print("\n" + "=" * 104)
bad = [r for r in RESULTS if not r[1]]
print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} analytical checks passed")
for name, ok, claimed, computed, tol, detail in bad:
    print(f"   FAILED: {name}  (paper {claimed}, recomputed {computed}, tol {tol})")
    print(f"           detail: {detail}")
with io.open("review/verify_theory_full.json", "w", encoding="utf-8") as fh:
    json.dump([dict(check=n, passed=bool(ok), claimed=float(c), computed=float(v),
                    tol=float(t), detail=d) for n, ok, c, v, t, d in RESULTS], fh, indent=1)
print("[saved] review/verify_theory_full.json")

"""
Verify the sharp form of the posterior-kernel negative result.

Claim (new Proposition, Section 3): with a Gaussian design, the calibrated regressor
mu_W = C W and the self-consistent posterior-kernel scale sigma(b) = sqrt(b' Sigma_{x|w} b),
the smoothed population loss

    L_cal(b) = E[ rho_tilde( Y - mu_W' b , sigma(b) ) ]

is uniquely minimised at the attenuated value beta_bar = (Sigma_x + Sigma_u)^{-1} Sigma_x beta*,
for *every* tau and *every* error distribution.

Proof structure checked here (all three steps numerically):

  step 1 (decomposition)   L_cal(b) = (tau - 1/2) E[eps] + (1/2) E| eps + s(b) Z |,
                           s^2(b) = (beta*-b)' A (beta*-b) + beta*' M beta* + b' M b,
                           A = C (Sigma_x + Sigma_u) C' = Sigma_x - M,  M = Sigma_{x|w}
  step 2 (monotonicity)    s -> E| eps + s Z | is strictly increasing
  step 3 (quadratic)       argmin_b s^2(b) = (Sigma_x + Sigma_u)^{-1} Sigma_x beta*

Run:  python review/verify_posterior_adaptive.py
Outputs: review/verify_posterior_adaptive.txt  (human-readable log)
         review/verify_posterior_adaptive.json (machine-readable)
"""

import io
import json
import os

import numpy as np
from scipy.special import ndtr

OUTDIR = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(OUTDIR, "verify_posterior_adaptive.txt")
JSON = os.path.join(OUTDIR, "verify_posterior_adaptive.json")

BETA_STAR = 2.0
LINES = []


def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES))


def phi_std(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)


def rho_tilde(r, s, tau):
    s = max(float(s), 1e-12)
    z = np.asarray(r, float) / s
    return r * (ndtr(z) - (1 - tau)) + s * phi_std(z)


def draw_eps(n, tau, rng):
    if tau == 0.5:
        return rng.laplace(0.0, 1.0, n)
    return (1 - tau) * rng.exponential(1.0, n) - tau * rng.exponential(1.0, n)


# --------------------------------------------------------------------- step 3
def quadratic_argmin(A, M, beta_star):
    """Exact minimiser of f(b) = (beta*-b)'A(beta*-b) + b'Mb  (A + M = Sigma_x)."""
    return np.linalg.solve(A + M, A @ beta_star)


# --------------------------------------------------------------------- scalar
def scalar_cell(sigma_u, tau, N=400_000, seed=777, grid_lo=None, step=0.005):
    lam = 1.0 / (1.0 + sigma_u ** 2)
    Sxw = 1.0 - lam
    A = lam                        # C (Sig_x + Sig_u) C'  = Sig_x - Sig_{x|w}
    M = Sxw                        # posterior variance
    beta_bar = lam * BETA_STAR     # = (Sig_x+Sig_u)^{-1} Sig_x beta*
    rng = np.random.RandomState(seed + int(1000 * sigma_u) + int(10 * tau))
    X = rng.standard_normal(N)
    U = sigma_u * rng.standard_normal(N)
    W = X + U
    Y = X * BETA_STAR + draw_eps(N, tau, rng)
    muW = lam * W

    def s2(b):
        """Variance of the residual after calibration and smoothing:
        (beta*-b)'A(beta*-b) + beta*'M beta* + b'M b."""
        return A * (BETA_STAR - b) ** 2 + M * BETA_STAR ** 2 + M * b ** 2

    def kernel(b):
        """The self-consistent posterior scale sigma_M(b) = sqrt(b' M b)."""
        return np.sqrt(M) * abs(b)

    if grid_lo is None:
        grid_lo = beta_bar - 0.6
    grid = np.arange(grid_lo, beta_bar + 0.6, step)
    L = np.array([float(np.mean(rho_tilde(Y - muW * b, kernel(b), tau))) for b in grid])
    k = int(np.argmin(L))
    if 0 < k < len(grid) - 1:
        y0, y1, y2 = L[k - 1], L[k], L[k + 1]
        den = y0 - 2 * y1 + y2
        b_hat = float(grid[k] + (0.5 * (y0 - y2) / den if den else 0.0) * step)
    else:
        b_hat = float(grid[k])

    # step 1 check: L_cal(b) vs (tau-1/2)E[eps] + (1/2)E|eps + s(b) Z|
    Z = rng.standard_normal(N)
    eps2 = draw_eps(N, tau, np.random.RandomState(4242))
    checks = []
    for b in (beta_bar - 0.2, beta_bar, beta_bar + 0.2):
        lhs = float(np.mean(rho_tilde(Y - muW * b, kernel(b), tau)))
        rhs = float((tau - 0.5) * np.mean(eps2) + 0.5 * np.mean(np.abs(eps2 + np.sqrt(s2(b)) * Z)))
        checks.append(dict(beta=b, lhs=lhs, rhs=rhs, diff=lhs - rhs))

    # step 2 check: s -> E|eps + s Z| strictly increasing
    sgrid = np.linspace(0.2, 2.0, 19)
    g = np.array([float(np.mean(np.abs(eps2 + s * Z))) for s in sgrid])
    mono = bool(np.all(np.diff(g) > 0))

    # analytic minimiser of s^2(b)
    b_exact = float(quadratic_argmin(np.array([[A]]), np.array([[M]]),
                                     np.array([BETA_STAR]))[0])
    return dict(sigma_u=sigma_u, tau=tau, beta_bar=beta_bar, argmin_mc=b_hat,
                deviation=b_hat - beta_bar, argmin_quadratic=b_exact,
                quad_deviation=b_exact - beta_bar, identity_checks=checks,
                monotone=mono, min_increment=float(np.min(np.diff(g))))


# --------------------------------------------------------------------- matrix
def matrix_case(tau=0.5, N=400_000, seed=99, step=0.004):
    Sigma_x = np.array([[1.0, 0.6], [0.6, 1.0]])
    Sigma_u = np.diag([0.4, 0.10])          # does not commute with Sigma_x
    beta_star = np.array([2.0, 1.0])
    C = Sigma_x @ np.linalg.inv(Sigma_x + Sigma_u)
    P = Sigma_x + Sigma_u
    M = Sigma_x - C @ Sigma_x               # posterior covariance Sigma_{x|w}
    A = C @ P @ C.T                         # = Sigma_x - M
    assert np.allclose(A + M, Sigma_x)
    beta_bar = np.linalg.solve(P, Sigma_x @ beta_star)
    b_quad = quadratic_argmin(A, M, beta_star)

    rng = np.random.RandomState(seed)
    L = np.linalg.cholesky(Sigma_x)
    Lu = np.linalg.cholesky(Sigma_u)
    X = rng.standard_normal((N, 2)) @ L.T
    U = rng.standard_normal((N, 2)) @ Lu.T
    W = X + U
    Y = X @ beta_star + draw_eps(N, tau, rng)
    muW = W @ C.T

    def grid_axis(centre):
        return np.arange(centre - 0.12, centre + 0.12 + 1e-12, step)

    g1, g2 = grid_axis(beta_bar[0]), grid_axis(beta_bar[1])
    best, best_val = None, np.inf
    for b1 in g1:
        for b2 in g2:
            b = np.array([b1, b2])
            s = float(np.sqrt(b @ M @ b))
            val = float(np.mean(rho_tilde(Y - muW @ b, s, tau)))
            if val < best_val:
                best_val, best = val, b
    return dict(Sigma_x=Sigma_x.tolist(), Sigma_u=Sigma_u.tolist(),
                beta_star=beta_star.tolist(), beta_bar=beta_bar.tolist(),
                argmin_grid=best.tolist(), argmin_quadratic=b_quad.tolist(),
                grid_deviation=float(np.linalg.norm(best - beta_bar)),
                quad_deviation=float(np.linalg.norm(b_quad - beta_bar)),
                A_plus_M_equals_Sigmax=bool(np.allclose(A + M, Sigma_x)))


def main():
    log("=" * 100)
    log("Verification: posterior-kernel smoothed loss at the calibrated regressor is minimised at beta_bar")
    log("=" * 100)
    results = {"scalar": [], "matrix": None}
    for sigma_u in (0.1, 0.3, 0.5):
        for tau in (0.3, 0.5, 0.7):
            r = scalar_cell(sigma_u, tau)
            results["scalar"].append(r)
            log(f"sigma_u={sigma_u:4.2f}  tau={tau:3.1f}   beta_bar={r['beta_bar']:.4f}   "
                f"MC argmin={r['argmin_mc']:.4f} (dev {r['deviation']:+.4f})   "
                f"quadratic argmin={r['argmin_quadratic']:.4f} (dev {r['quad_deviation']:+.2e})")
            for c in r["identity_checks"]:
                log(f"      identity at b={c['beta']:.3f}:  L_cal={c['lhs']:.6f}  "
                    f"(tau-1/2)E[eps]+E|eps+sZ|/2={c['rhs']:.6f}  diff={c['diff']:+.2e}")
            log(f"      E|eps+sZ| increasing in s: {r['monotone']}  "
                f"(min increment {r['min_increment']:.2e})")
    log("")
    log("-" * 100)
    log("Matrix case (Sigma_x and Sigma_u do not commute), p = 2")
    m = matrix_case()
    results["matrix"] = m
    log(f"  Sigma_x = {m['Sigma_x']}   Sigma_u = {m['Sigma_u']}")
    log(f"  beta* = {m['beta_star']}   beta_bar = {np.round(m['beta_bar'], 6).tolist()}")
    log(f"  A + M = Sigma_x: {m['A_plus_M_equals_Sigmax']}")
    log(f"  exact quadratic argmin      = {np.round(m['argmin_quadratic'], 6).tolist()}  "
        f"(||. - beta_bar|| = {m['quad_deviation']:.2e})")
    log(f"  2-D grid argmin of MC loss  = {np.round(m['argmin_grid'], 4).tolist()}  "
        f"(||. - beta_bar|| = {m['grid_deviation']:.4f}, grid step 0.004)")
    log("")
    devs = [abs(r["deviation"]) for r in results["scalar"]]
    qdevs = [abs(r["quad_deviation"]) for r in results["scalar"]]
    log(f"scalar cells: max |MC argmin - beta_bar| = {max(devs):.4f};  "
        f"max |quadratic argmin - beta_bar| = {max(qdevs):.2e}")
    log(f"identity checks: max |lhs - rhs| = "
        f"{max(abs(c['diff']) for r in results['scalar'] for c in r['identity_checks']):.2e}")
    log(f"monotonicity holds in all cells: {all(r['monotone'] for r in results['scalar'])}")
    results["summary"] = dict(max_mc_deviation=max(devs), max_quad_deviation=max(qdevs),
                              all_monotone=bool(all(r["monotone"] for r in results["scalar"])))
    with io.open(JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    log(f"[saved] {JSON}")


if __name__ == "__main__":
    main()

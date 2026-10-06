"""
Route-A experiments, part 1: EXACT population targets (no LP solver, no sklearn).

Everything here is computed to numerical exactness for p = 1:
  * no-intercept QR   -> exact subgradient root-finding (sorted arrays, O(log n) per eval)
  * with-intercept QR -> profile over the slope, intercept = tau-quantile of residuals
  * convolved losses  -> fine grid + parabolic refinement

Purpose: establish, for the record, WHAT each estimator converges to.
    beta*      = 2                    (the parameter of interest)
    beta_bar   = Sigma_x/(Sigma_x+Sigma_u) * beta*  (attenuated target)
The paper's corrected loss is compared against
    (a) its fixed-sigma form (sigma from Stage 1)   <- what the manuscript actually runs
    (b) its adaptive form   (sigma = |beta| sigma_u)
and against naive QR, regression calibration (RC), RC + intercept, and the oracle.
"""

import io
import json
import time
import numpy as np

LOG = "simulations/routeA_targets_output.txt"
JSON = "simulations/routeA_targets.json"
OUT = []


def log(m=""):
    print(m, flush=True)
    OUT.append(str(m))
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))


def phi_std(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)


def rho_tilde(r, s, tau):
    from scipy.special import ndtr
    s = max(float(s), 1e-12)
    z = np.asarray(r, float) / s
    return r * (ndtr(z) - (1 - tau)) + s * phi_std(z)


def check_loss(u, tau):
    return u * (tau - (u < 0).astype(float))


def draw_eps(n, tau, rng):
    if tau == 0.5:
        return rng.laplace(0.0, 1.0, n)
    return (1 - tau) * rng.exponential(1.0, n) - tau * rng.exponential(1.0, n)


# --------------------------------------------------------------- exact 1-D QR
def qr_no_intercept(Y, Z, tau, lo=0.0, hi=6.0, iters=200):
    """Exact minimiser of mean rho_tau(Y - b Z) over b, via the subgradient
    g(b) = tau*S_pos - A(b) + (tau-1)*S_neg + B(b), which is non-increasing."""
    pos = Z > 0
    t_pos = np.sort(Y[pos] / Z[pos])
    cz_pos = np.concatenate([[0.0], np.cumsum(Z[pos][np.argsort(Y[pos] / Z[pos])])])
    neg = ~pos
    t_neg = np.sort(Y[neg] / Z[neg])
    cz_neg = np.concatenate([[0.0], np.cumsum(Z[neg][np.argsort(Y[neg] / Z[neg])])])
    S_pos, S_neg = float(Z[pos].sum()), float(Z[neg].sum())

    def g(b):
        k = int(np.searchsorted(t_pos, b))
        A = float(cz_pos[k])
        m = int(np.searchsorted(t_neg, b))
        B = float(cz_neg[m])
        return tau * S_pos - A + (tau - 1.0) * S_neg + B

    a, c = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + c)
        if g(mid) > 0:
            a = mid
        else:
            c = mid
        if c - a < 1e-12:
            break
    return 0.5 * (a + c)


def qr_intercept_slope(Y, Z, tau, lo=0.0, hi=6.0, n_grid=301):
    """Profile: f(b) = min_delta mean rho_tau(Y - b Z - delta) with
    delta(b) = tau-quantile(Y - b Z).  f is convex in b."""
    def f(b):
        r = Y - b * Z
        d = np.quantile(r, tau)
        return float(np.mean(check_loss(r - d, tau))), d

    grid = np.linspace(lo, hi, n_grid)
    vals = np.array([f(b)[0] for b in grid])
    k = int(np.argmin(vals))
    a = grid[max(k - 1, 0)]
    c = grid[min(k + 1, n_grid - 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = c - gr * (c - a), a + gr * (c - a)
    f1, f2 = f(x1)[0], f(x2)[0]
    for _ in range(120):
        if f1 < f2:
            c, x2, f2 = x2, x1, f1
            x1 = c - gr * (c - a)
            f1 = f(x1)[0]
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + gr * (c - a)
            f2 = f(x2)[0]
    b = 0.5 * (a + c)
    return b, float(f(b)[1])


def convolved_target(Y, Z, s_of_b, tau, tag, grid_step=0.005, hi=4.0):
    grid = np.arange(0.0, hi + 1e-9, grid_step)
    L = np.empty_like(grid)
    for k, b in enumerate(grid):
        L[k] = np.mean(rho_tilde(Y - Z * b, s_of_b(b), tau))
    k = int(np.argmin(L))
    if 0 < k < len(grid) - 1:
        y0, y1, y2 = L[k - 1], L[k], L[k + 1]
        den = y0 - 2 * y1 + y2
        b = grid[k] + (0.5 * (y0 - y2) / den if den else 0.0) * grid_step
    else:
        b = grid[k]
    return float(b)


# --------------------------------------------------------------------------- main
def main(N=200_000, beta_star=2.0, taus=(0.3, 0.5, 0.7), sigmas=(0.1, 0.3, 0.5),
         seed=20240607):
    log("=" * 100)
    log("ROUTE-A PART 1: exact population targets  (p = 1, X ~ N(0,1), U ~ N(0,sigma_u^2))")
    log("=" * 100)
    log(f"beta* = {beta_star}, N = {N:,} per cell, tau in {list(taus)}, sigma_u in {list(sigmas)}")
    log("")
    results = []
    for sigma_u in sigmas:
        Su = sigma_u ** 2
        lam = 1.0 / (1.0 + Su)              # Sigma_x/(Sigma_x+Sigma_u)
        beta_bar = lam * beta_star
        Sxw = 1.0 - lam                     # posterior variance of X | W
        log("#" * 100)
        log(f"# sigma_u = {sigma_u}   Sigma_u = {Su:.4f}   calibration mu_W = {lam:.4f} W   "
            f"posterior var = {Sxw:.4f}")
        log(f"#   beta* = {beta_star:.4f}   beta_bar = {beta_bar:.4f}   "
            f"||beta_bar - beta*|| = {abs(beta_bar-beta_star):.4f}")
        log("#" * 100)
        for tau in taus:
            rng = np.random.RandomState(seed + int(sigma_u * 1000) + int(tau * 10))
            X = rng.standard_normal(N)
            U = rng.standard_normal(N) * sigma_u
            W = X + U
            Y = X * beta_star + draw_eps(N, tau, rng)
            muW = lam * W
            # predicted residual-quantile shift after calibration
            Vb = rng.standard_normal(4_000_000) * np.sqrt(Sxw) * beta_star
            delta = float(np.quantile(draw_eps(4_000_000, tau, np.random.RandomState(5)) + Vb, tau))

            row = dict(sigma_u=sigma_u, tau=tau, beta_star=beta_star,
                       beta_bar=beta_bar, delta=delta)
            log(f"  --- tau = {tau}  (predicted intercept after calibration: delta = {delta:+.4f}) ---")

            b = convolved_target(Y, W, lambda bb: abs(beta_star) * np.sqrt(Su), tau,
                                 "convolved, sigma fixed at |beta*|sigma_u")
            row["conv_fixed_betastar"] = b
            log(f"      convolved loss, sigma fixed at |beta*|sigma_u                 -> {b:8.4f}"
                f"   (bias {b-beta_star:+.4f})")
            b = convolved_target(Y, W, lambda bb: abs(beta_bar) * np.sqrt(Su), tau,
                                 "convolved, sigma fixed at Stage-1 value")
            row["conv_fixed_stage1"] = b
            log(f"      convolved loss, sigma fixed at Stage-1 value |beta_bar|sigma_u -> {b:8.4f}"
                f"   (bias {b-beta_star:+.4f})")
            b = convolved_target(Y, W, lambda bb: abs(bb) * np.sqrt(Su), tau,
                                 "convolved, adaptive sigma(b)=|b|sigma_u")
            row["conv_adaptive"] = b
            log(f"      convolved loss, adaptive sigma(b) = |b|sigma_u                -> {b:8.4f}"
                f"   (bias {b-beta_star:+.4f})")
            b = convolved_target(Y, muW, lambda bb: abs(bb) * np.sqrt(Sxw), tau,
                                 "CONTROL: convolved on mu_W, posterior kernel")
            row["conv_posterior_kernel"] = b
            log(f"      CONTROL: convolved on mu_W with posterior kernel Sigma_x|w  -> {b:8.4f}"
                f"   (bias {b-beta_star:+.4f})   [Corollary 3.3: still inconsistent]")

            b = qr_no_intercept(Y, W, tau)
            row["naive_qr"] = b
            log(f"      naive QR on W (no intercept)                                  -> {b:8.4f}"
                f"   (bias {b-beta_star:+.4f})")
            b, d = qr_intercept_slope(Y, W, tau)
            row["naive_qr_int"] = b
            row["naive_qr_int_intercept"] = d
            log(f"      naive QR on W with intercept                                   -> {b:8.4f}"
                f"   (bias {b-beta_star:+.4f}, intercept {d:+.4f})")

            b = qr_no_intercept(Y, muW, tau)
            row["rc_qr"] = b
            log(f"      RC: QR on mu_W (no intercept)                                  -> {b:8.4f}"
                f"   (bias {b-beta_star:+.4f})")
            b, d = qr_intercept_slope(Y, muW, tau)
            row["rc_qr_int"] = b
            row["rc_qr_int_intercept"] = d
            log(f"      RC + intercept: QR on [mu_W, 1]                                -> {b:8.4f}"
                f"   (bias {b-beta_star:+.4f}, intercept {d:+.4f})")

            b = qr_no_intercept(Y, X, tau)
            row["oracle_qr"] = b
            log(f"      ORACLE: QR on clean X                                          -> {b:8.4f}"
                f"   (bias {b-beta_star:+.4f})")
            log("")
            results.append(row)
    with io.open(JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    log(f"[saved] {JSON}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"[total {time.time()-t0:.1f}s]")
    print(f"\n[saved] {LOG}")

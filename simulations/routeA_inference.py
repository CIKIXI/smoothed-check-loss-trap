"""
Route-A experiment 5: does the repair actually deliver valid inference?

Theorem 4.2 (lowdim) claims that the calibrated estimator is asymptotically normal with
the standard quantile-regression sandwich covariance.  The paper contained no numerical
validation of that claim, which is the constructive half's payoff.  This script checks it.

E5a  coverage of nominal 90% and 95% Wald intervals for beta*_j using the theorem's
     covariance, for several n and sigma_u, at tau = 0.3, 0.5, 0.7.
E5b  the same for the SMOOTHED estimator, whose target is beta_bar: we report coverage of
     beta*_j (expected to under-cover) and coverage of beta_bar_j (expected to be close to
     nominal), which is the numerical form of the "target shift" statement.
E5c  numeric check of Corollary 3.4: the naive-score mean E[W(tau - 1{r*<0})] equals
     -(Sigma_u beta*) * d_naive with d_naive = (N(0,sigma*^2) * f_eps)(0).

Model: scalar, X ~ N(0,1), U ~ N(0,sigma_u^2), Q_tau(eps)=0, Sigma_u known.
"""

import io
import json
import time

import numpy as np
from scipy.special import ndtr

LOG = "simulations/routeA_inference_output.txt"
OUT = []


def log(m=""):
    print(m, flush=True)
    OUT.append(str(m))
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))


def phi(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)


def draw_eps(n, tau, rng):
    if tau == 0.5:
        return rng.laplace(0.0, 1.0, n)
    return (1 - tau) * rng.exponential(1.0, n) - tau * rng.exponential(1.0, n)


# ---------------------------------------------------------------- fitting
def fit_calibrated(Y, muW, tau, delta0=None):
    """Quantile regression of Y on [muW, 1] with intercept delta.  Returns (b, delta, se_b).

    Solved by exact 1-D profiling: for fixed b, the optimal delta is the tau-quantile of
    Y - b*muW, so the objective is a convex function of b alone (midpoint of the minimising
    interval found by bisection on the subgradient)."""
    n = len(Y)

    def loss(b):
        r = Y - b * muW
        d = np.quantile(r, tau)
        u = r - d
        return float(np.mean(u * (tau - (u < 0).astype(float)))), d

    lo, hi = -8.0, 8.0
    for _ in range(80):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if loss(m1)[0] < loss(m2)[0]:
            hi = m2
        else:
            lo = m1
    b = 0.5 * (lo + hi)
    _, delta = loss(b)
    # standard error: sandwich with sparsity estimator of the residual density at 0
    u = Y - b * muW - delta
    h = 1.06 * np.std(u) * n ** (-0.2)
    f_hat = float(np.mean(np.abs(u) < h) / (2 * h))
    v = np.column_stack([muW - muW.mean(), np.ones(n)])
    A = (v.T @ v) / n
    se = np.sqrt(tau * (1 - tau) / (n * f_hat ** 2) * np.diag(np.linalg.inv(A)))
    return b, delta, float(se[0]), f_hat


def fit_smoothed(Y, Z, sigma, tau):
    """Smoothed (convolved) estimator with fixed scale sigma, exact ndtr; returns (b, se_b)
    with the same sandwich form based on the smoothed score."""
    n = len(Y)

    def fg(b):
        r = Y - b * Z
        g = ndtr(r / sigma) - (1 - tau)
        val = float(np.mean(r * g + sigma * phi(r / sigma)))
        grad = -float(Z @ g) / n
        return val, grad, r

    lo, hi = -8.0, 8.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        # f is convex, so f' is increasing: f'(m) > 0 puts the minimum to the LEFT
        if fg(m)[1] > 0:
            hi = m
        else:
            lo = m
    b = 0.5 * (lo + hi)
    r = Y - b * Z
    w = phi(r / sigma) / sigma
    A = float(np.mean(Z ** 2 * w))
    B = float(np.mean((Z * (ndtr(r / sigma) - (1 - tau))) ** 2))
    se = np.sqrt(B / max(A, 1e-12) ** 2 / n)
    return b, float(se)


# ---------------------------------------------------------------- E5a/b
def exp_E5(nrep=400, ns=(500, 1000, 2000, 4000), sigma_u=0.5, taus=(0.3, 0.5, 0.7),
           beta_star=2.0, seed=555, n0=200000):
    lam = 1.0 / (1.0 + sigma_u ** 2)
    beta_bar = lam * beta_star
    log("=" * 100)
    log("E5  inference: coverage of nominal 90% / 95% intervals")
    log("=" * 100)
    log(f"scalar model, X~N(0,1), U~N(0,{sigma_u}^2), beta* = {beta_star}, "
        f"beta_bar = {beta_bar:.4f}")
    log(f"{nrep} replications per cell; intervals from the sandwich covariance of "
        f"Theorem 4.2 (calibrated) and of the smoothed score (smoothed)")
    log("")
    res = {}
    for tau in taus:
        # a large pilot sample to fix sigma* for the smoothed estimator
        rng0 = np.random.RandomState(seed + int(tau * 100))
        X0 = rng0.standard_normal(n0)
        W0 = X0 + rng0.standard_normal(n0) * sigma_u
        sig_star = abs(beta_star) * sigma_u
        log(f"--- tau = {tau} (sigma* = {sig_star:.3f}) ---")
        log(f"{'n':>6} | {'calibrated: cov90':>17} {'cov95':>7} {'se':>8} | "
            f"{'smoothed -> beta*':>18} {'cov95':>7} | {'smoothed -> beta_bar':>20} {'cov95':>7}")
        for n in ns:
            c90 = c95 = 0
            s95_star = s95_bar = 0
            ses = []
            z95 = 1.959963985
            z90 = 1.644853627
            for rep in range(nrep):
                rng = np.random.RandomState(seed + rep * 977 + n + int(tau * 10))
                X = rng.standard_normal(n)
                U = rng.standard_normal(n) * sigma_u
                W = X + U
                Y = X * beta_star + draw_eps(n, tau, rng)
                muW = lam * W
                b, d, se, _ = fit_calibrated(Y, muW, tau)
                ses.append(se)
                c90 += abs(b - beta_star) <= z90 * se
                c95 += abs(b - beta_star) <= z95 * se
                bs, ses_s = fit_smoothed(Y, W, sig_star, tau)
                s95_star += abs(bs - beta_star) <= z95 * ses_s
                s95_bar += abs(bs - beta_bar) <= z95 * ses_s
            row = dict(n=n, tau=tau, cov90_cal=c90 / nrep, cov95_cal=c95 / nrep,
                       mean_se=float(np.mean(ses)), cov95_smoothed_to_star=s95_star / nrep,
                       cov95_smoothed_to_bar=s95_bar / nrep)
            res[f"tau{tau}_n{n}"] = row
            log(f"{n:>6} | {row['cov90_cal']:>17.3f} {row['cov95_cal']:>7.3f} "
                f"{row['mean_se']:>8.4f} | {row['cov95_smoothed_to_star']:>18.3f} "
                f"{'':>7} | {row['cov95_smoothed_to_bar']:>20.3f} {s95_bar/nrep:>7.3f}")
        log("")
    res["_meta"] = dict(sigma_u=sigma_u, beta_star=beta_star, beta_bar=beta_bar,
                        nrep=nrep, ns=list(ns), taus=list(taus))
    with io.open("simulations/routeA_results_E5.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


# ---------------------------------------------------------------- E5c
def exp_E5c(beta_val=2.0, sigma_u_list=(0.1, 0.3, 0.5), taus=(0.3, 0.5, 0.7),
            N=400_000, seed=606):
    log("=" * 100)
    log("E5c  numeric check of Corollary 3.4 (naive score)")
    log("     E[W(tau - 1{r*<0})] =? -(Sigma_u beta*) d_naive, "
        "d_naive = (N(0,sigma*^2)*f_eps)(0)")
    log("=" * 100)
    res = {}
    for sigma_u in sigma_u_list:
        sigma_star = abs(beta_val) * sigma_u
        for tau in taus:
            rng = np.random.RandomState(seed + int(sigma_u * 100) + int(tau * 10))
            X = rng.standard_normal(N)
            U = rng.standard_normal(N) * sigma_u
            W = X + U
            Y = X * beta_val + draw_eps(N, tau, rng)
            r = Y - W * beta_val
            s = W * (tau - (r < 0).astype(float))
            m, se = float(np.mean(s)), float(np.std(s) / np.sqrt(N))
            e = draw_eps(4_000_000, tau, np.random.RandomState(3))
            d_naive = float(np.mean(phi(e / sigma_star) / sigma_star))
            pred = -sigma_u ** 2 * beta_val * d_naive
            res[f"su{sigma_u}_tau{tau}"] = dict(sigma_u=sigma_u, tau=tau, E_Wpsi=m,
                                                se=se, t=m / se, pred=pred)
            log(f"  sigma_u={sigma_u} tau={tau}: E[W psi] = {m:+.5f} +- {se:.5f} "
                f"(t = {m/se:+8.1f});  analytic {pred:+.5f}")
    with io.open("simulations/routeA_results_E5c.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log("")
    return res


if __name__ == "__main__":
    t0 = time.time()
    exp_E5c()
    exp_E5()
    log(f"[total {time.time()-t0:.1f}s]")
    print(f"\n[saved] {LOG}")

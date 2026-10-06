"""
Debiased inference for the calibrated estimator -- final experiment.

What the diagnostics (hdc_debias_diag.py, hdc_debias_fix.py) established:

1. One-step debiasing needs an initial estimator whose bias is small relative to its
   standard error.  With l1-QR at lambda = sqrt(log p / n) the bias on a coefficient of
   size 2 is about -0.93 against a standard error of 0.32, and one-step debiasing then
   over-corrects to +0.86: coverage 0.26.
2. REFITTING the initial estimator by unpenalised QR on the l1-selected support removes
   that bias (+0.006 at the same lambda), after which one-step debiasing covers.
3. Smoothing the SCORE biases the centre by O(h^2); using the exact indicator score
   tau - 1{e < 0} with a kernel-smoothed Hessian avoids it.

This script produces the table: for each sigma_u and lambda,
    l1            l1-QR, sandwich interval (ignores the penalty bias)
    debiased      one-step debiasing from the l1 initial estimator
    refit         unpenalised QR on the selected support
    refit+deb     one-step debiasing from the refitted estimator
with coverage on the support and on its complement, interval lengths, and selection
quality.  Design: X ~ N(0,I), mu_W = C W with C = (1 + sigma_u^2)^{-1} I (known).
"""

import io
import json
import time
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdc_debias import fit_l1, debias, Z95
from routeA_lib import calibration_matrix, draw_eps, phi_std

from sklearn.linear_model import QuantileRegressor

LOG = "simulations/hdc_debias_output.txt"
OUT = []


def log(m=""):
    print(m, flush=True)
    OUT.append(str(m))
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))


def refit(Y, D, support, tau):
    idx = np.where(support)[0]
    b = np.zeros(D.shape[1])
    if len(idx) == 0:
        return b, float(np.quantile(Y, tau))
    q = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")
    q.fit(D[:, idx], Y)
    b[idx] = q.coef_
    return b, float(q.intercept_)


def run(n=400, p=40, s=5, beta_val=2.0, sigma_u_list=(0.3, 0.5), tau=0.5, R=200,
        lam_factors=(1.0, 0.5), seed0=24680):
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    lam0 = float(np.sqrt(np.log(p) / n))
    log("=" * 104)
    log("DEB  debiased inference for the calibrated estimator (p < n: exact plug-in Hessian)")
    log("=" * 104)
    log(f"n = {n}, p = {p}, s = {s}, tau = {tau}, R = {R}, nominal level 95%; "
        f"lambda = factor x sqrt(log p / n) = factor x {lam0:.4f}")
    log("")
    res = {}
    for sigma_u in sigma_u_list:
        C = calibration_matrix(np.eye(p), sigma_u ** 2 * np.eye(p))
        for fac in lam_factors:
            lam = fac * lam0
            acc = {k: [] for k in ("cov_sup_l1", "cov_sup_deb", "cov_sup_ref",
                                   "cov_sup_rd", "cov_null_rd", "cov_null_deb",
                                   "len_rd", "len_ref", "len_deb", "len_l1",
                                   "nsup", "exact_support")}
            for rep in range(R):
                rng = np.random.RandomState(seed0 + rep * 7919)
                X = rng.standard_normal((n, p))
                W = X + rng.standard_normal((n, p)) * sigma_u
                Y = X @ beta_star + draw_eps(n, tau, rng)
                muW = W @ C.T
                sup = np.arange(s)
                null = np.arange(s, p)

                b_l1, d_l1 = fit_l1(Y, muW, tau, lam)
                b_db, _, se_db = debias(Y, muW, b_l1, d_l1, tau, score="exact")
                support = np.abs(b_l1) > 1e-8
                b_rf, d_rf = refit(Y, muW, support, tau)
                b_rd, _, se_rd = debias(Y, muW, b_rf, d_rf, tau, score="exact")
                # sandwich intervals for the penalised and refitted estimators
                se_l1 = sandwich(Y, muW, b_l1, d_l1, tau)[:p]
                se_rf = sandwich(Y, muW, b_rf, d_rf, tau)[:p]

                acc["cov_sup_l1"].append(np.mean(np.abs(b_l1[sup] - beta_star[sup]) <= Z95 * se_l1[sup]))
                acc["cov_sup_deb"].append(np.mean(np.abs(b_db[sup] - beta_star[sup]) <= Z95 * se_db[sup]))
                acc["cov_sup_ref"].append(np.mean(np.abs(b_rf[sup] - beta_star[sup]) <= Z95 * se_rf[sup]))
                acc["cov_sup_rd"].append(np.mean(np.abs(b_rd[sup] - beta_star[sup]) <= Z95 * se_rd[sup]))
                acc["cov_null_rd"].append(np.mean(np.abs(b_rd[null] - beta_star[null]) <= Z95 * se_rd[null]))
                acc["cov_null_deb"].append(np.mean(np.abs(b_db[null] - beta_star[null]) <= Z95 * se_db[null]))
                acc["len_rd"].append(float(np.mean(2 * Z95 * se_rd[sup])))
                acc["len_ref"].append(float(np.mean(2 * Z95 * se_rf[sup])))
                acc["len_deb"].append(float(np.mean(2 * Z95 * se_db[sup])))
                acc["len_l1"].append(float(np.mean(2 * Z95 * se_l1[sup])))
                acc["nsup"].append(int(support.sum()))
                acc["exact_support"].append(float(set(np.where(support)[0]) == set(sup)))
            row = {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(R)))
                   for k, v in acc.items()}
            res[f"su{sigma_u}_lam{fac}"] = {k: dict(mean=v[0], se=v[1]) for k, v in row.items()}
            log(f"  --- sigma_u = {sigma_u}, lambda = {lam:.4f} (factor {fac}) ---")
            log(f"    coverage on the support:  l1 {row['cov_sup_l1'][0]:.3f} | "
                f"debiased(l1) {row['cov_sup_deb'][0]:.3f} | "
                f"refit {row['cov_sup_ref'][0]:.3f} | "
                f"refit+debiased {row['cov_sup_rd'][0]:.3f}+-{row['cov_sup_rd'][1]:.3f}")
            log(f"    coverage on null coordinates: debiased(l1) {row['cov_null_deb'][0]:.3f} | "
                f"refit+debiased {row['cov_null_rd'][0]:.3f}")
            log(f"    mean interval length: l1 {row['len_l1'][0]:.3f}, "
                f"debiased(l1) {row['len_deb'][0]:.3f}, "
                f"refit {row['len_ref'][0]:.3f}, refit+debiased {row['len_rd'][0]:.3f}")
            log(f"    selected support size {row['nsup'][0]:.2f} (true {s}); "
                f"exact recovery {row['exact_support'][0]:.3f}")
            log("")
    res["_meta"] = dict(n=n, p=p, s=s, tau=tau, R=R, beta_val=beta_val,
                        sigma_u_list=list(sigma_u_list), lam_factors=list(lam_factors),
                        lam0=lam0)
    with io.open("simulations/hdc_debias.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


def sandwich(Y, D, beta, delta, tau, h_scale=1.06):
    """Usual sandwich SE for a fitted QR (used for the non-debiased estimators)."""
    n, p = D.shape
    e = Y - D @ beta - delta
    h = max(h_scale * np.std(e) * n ** (-0.2), 1e-3)
    X = np.column_stack([D, np.ones(n)])
    z = e / h
    H = (X.T * (phi_std(z) / h)) @ X / n
    J = (X.T * (tau - (e < 0)) ** 2) @ X / n
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Hi = np.linalg.pinv(H)
    return np.sqrt(np.maximum(np.diag(Hi @ J @ Hi / n), 0.0))


if __name__ == "__main__":
    t0 = time.time()
    run()
    log(f"[total {time.time()-t0:.1f}s]")
    print(f"\n[saved] {LOG}")

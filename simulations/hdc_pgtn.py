"""
Debiased inference for the calibrated estimator when p > n.

When p > n the plug-in Hessian Hhat = (1/n) sum_i w_i x_i x_i' has rank at most n and is
singular, so Hhat^{-1} does not exist.  The standard remedy (Meinshausen & Buehlmann;
van de Geer et al. 2014) approximates it row by row: for each coordinate j,

    gamma_j = argmin_gamma  (1/2) gamma' M_{-j,-j} gamma - gamma' M_{-j,j}
                            + lambda_j ||gamma||_1,      M = Hhat,
    row_j(Theta) = [ -gamma_j , 1 ] / tau_j^2,
    tau_j^2 = M_jj - 2 gamma_j'M_{-j,j} + gamma_j'M_{-j,-j}gamma_j.

Crucially, the debiased estimator b_j = beta_j + Theta_{j.} S(beta) needs only ROW j of
Theta, and the nodewise problems decouple, so we only solve them for the coordinates we
report.  That makes the p > n case cheap.

The quadratic nodewise problem is turned into a Lasso regression by a Cholesky factor:
with M_{-j,-j} = L L',

    min_gamma (1/2)||L'gamma - L^{-1}M_{-j,j}||^2 + lambda_j ||gamma||_1.

Diagnostics reported: max_j |(Theta Hhat)_jj - 1| (the quality of the inverse
approximation) and the coverage of nominal 95% intervals.
"""

import io
import json
import time
import os
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routeA_lib import calibration_matrix, draw_eps, phi_std
from hdc_debias import fit_l1, Z95

from sklearn.linear_model import Lasso, QuantileRegressor

LOG = "simulations/hdc_pgtn_output.txt"
OUT = []


def log(m=""):
    print(m, flush=True)
    OUT.append(str(m))
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))


def hessian(Y, D, beta, delta, tau, h_scale=1.06):
    n, p = D.shape
    e = Y - D @ beta - delta
    h = max(h_scale * np.std(e) * n ** (-0.2), 1e-3)
    X = np.column_stack([D, np.ones(n)])
    w = phi_std(e / h) / h
    return (X.T * w) @ X / n, e, h


def nodewise_rows(H, idx, lam_factor=0.7, n_obs=None):
    """Rows idx of Theta with Theta H ~ I, by nodewise l1 regression."""
    p1 = H.shape[0]
    lam_base = lam_factor * np.sqrt(np.log(p1) / (n_obs or p1))
    rows = {}
    for j in idx:
        keep = np.array([k for k in range(p1) if k != j])
        Mjj = H[j, j]
        M_rest = H[np.ix_(keep, keep)]
        m_j = H[keep, j]
        try:
            L = np.linalg.cholesky(M_rest + 1e-10 * np.eye(len(keep)))
        except np.linalg.LinAlgError:
            w_, V_ = np.linalg.eigh(M_rest)
            w_ = np.maximum(w_, 1e-10)
            L = V_ @ np.diag(np.sqrt(w_))
        y = np.linalg.solve(L, m_j)              # L^{-1} M_{-j,j}
        lam = lam_base * np.sqrt(max(Mjj, 1e-12))
        la = Lasso(alpha=lam, fit_intercept=False, max_iter=3000, tol=1e-6)
        la.fit(L.T, y)
        gamma = la.coef_
        tau2 = Mjj - 2 * gamma @ m_j + gamma @ (M_rest @ gamma)
        tau2 = max(tau2, 1e-10)
        row = np.zeros(p1)
        row[keep] = -gamma / tau2
        row[j] = 1.0 / tau2
        rows[j] = row
    return rows


def run(n=200, p=250, s=5, beta_val=2.0, sigma_u=0.5, tau=0.5, R=50,
        lam_factors=(0.6, 0.4), n_null=15, seed0=6060):
    Sigma_x = np.eye(p)
    C = calibration_matrix(Sigma_x, sigma_u ** 2 * Sigma_x)
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    lam0 = float(np.sqrt(np.log(p) / n))
    log("=" * 100)
    log("PGTN  debiased inference with p > n (nodewise inverse of the plug-in Hessian)")
    log("=" * 100)
    log(f"n = {n}, p = {p}, s = {s}, sigma_u = {sigma_u}, tau = {tau}, R = {R}")
    log(f"lambda = factor x sqrt(log p / n) = factor x {lam0:.4f}; nodewise rows computed "
        f"for the {s} signal and {n_null} null coordinates")
    log("")
    out = {"_meta": dict(n=n, p=p, s=s, sigma_u=sigma_u, tau=tau, R=R, lam0=lam0,
                         n_null=n_null, lam_factors=list(lam_factors))}
    for lam_factor in lam_factors:
        lam = lam_factor * lam0
        acc = {k: [] for k in ("cov_sup", "cov_null", "len_sup", "nsup", "exact_support",
                               "inv_err", "cov_sup_l1", "n_under")}
        for rep in range(R):
            rng = np.random.RandomState(seed0 + rep * 7919)
            X = rng.standard_normal((n, p))
            W = X + rng.standard_normal((n, p)) * sigma_u
            Y = X @ beta_star + draw_eps(n, tau, rng)
            muW = W @ C.T
            b_l1, d_l1 = fit_l1(Y, muW, tau, lam)
            support = np.abs(b_l1) > 1e-8
            idx = np.where(support)[0]
            if len(idx) == 0:
                idx = np.array([0])
            q = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")
            q.fit(muW[:, idx], Y)
            b_rf = np.zeros(p)
            b_rf[idx] = q.coef_
            d_rf = float(q.intercept_)
            H, e, h = hessian(Y, muW, b_rf, d_rf, tau)
            sup = np.arange(s)
            null_pool = np.arange(s, p)
            null = rng.choice(null_pool, size=min(n_null, len(null_pool)), replace=False)
            want = np.concatenate([sup, null])
            rows = nodewise_rows(H, want, n_obs=n)
            Xa = np.column_stack([muW, np.ones(n)])
            psi = tau - (e < 0).astype(float)
            S = Xa.T @ psi / n
            J = (Xa.T * (psi ** 2)) @ Xa / n
            b = np.concatenate([b_rf, [d_rf]])
            bd, sed = {}, {}
            for j in want:
                row = rows[j]
                bd[j] = b[j] + row @ S
                sed[j] = np.sqrt(max(row @ J @ row, 0.0) / n)
            acc["cov_sup"].append(float(np.mean(
                [abs(bd[j] - beta_star[j]) <= Z95 * sed[j] for j in sup])))
            acc["cov_null"].append(float(np.mean(
                [abs(bd[j] - beta_star[j]) <= Z95 * sed[j] for j in null])))
            acc["len_sup"].append(float(np.mean([2 * Z95 * sed[j] for j in sup])))
            acc["nsup"].append(int(support.sum()))
            acc["exact_support"].append(float(set(idx.tolist()) == set(sup.tolist())))
            acc["n_under"].append(int(len(set(sup.tolist()) - set(idx.tolist()))))
            acc["inv_err"].append(float(max(abs(rows[j] @ H[:, j] - 1.0) for j in want)))
            Hl, el, hl = hessian(Y, muW, b_l1, d_l1, tau)
            Xl = np.column_stack([muW, np.ones(n)])
            Jl = (Xl.T * ((tau - (el < 0)) ** 2)) @ Xl / n
            Hi = np.linalg.pinv(Hl + 1e-8 * np.eye(p + 1))
            se_l1 = np.sqrt(np.maximum(np.diag(Hi @ Jl @ Hi / n), 0.0))
            acc["cov_sup_l1"].append(float(np.mean(
                [abs(b_l1[j] - beta_star[j]) <= Z95 * se_l1[j] for j in sup])))
        row = {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(R)))
               for k, v in acc.items()}
        out[f"lam{lam_factor}"] = {k: dict(mean=v[0], se=v[1]) for k, v in row.items()}
        log(f"  --- lambda = {lam:.4f} (factor {lam_factor}) ---")
        log(f"    coverage on the support (nodewise + refit + debias): "
            f"{row['cov_sup'][0]:.3f}+-{row['cov_sup'][1]:.3f}")
        log(f"    coverage on {n_null} null coordinates:                  "
            f"{row['cov_null'][0]:.3f}+-{row['cov_null'][1]:.3f}")
        log(f"    coverage of the penalised estimator's own intervals: "
            f"{row['cov_sup_l1'][0]:.3f}")
        log(f"    mean interval length on the support: {row['len_sup'][0]:.3f}")
        log(f"    selected support size {row['nsup'][0]:.2f} (true {s}); exact recovery "
            f"{row['exact_support'][0]:.3f}; missed signals {row['n_under'][0]:.2f}")
        log(f"    inverse diagnostic max_j |(Theta H)_jj - 1|: {row['inv_err'][0]:.4f}")
        log("")
    with io.open("simulations/hdc_pgtn.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    t0 = time.time()
    run()
    log(f"[total {time.time()-t0:.1f}s]")
    print(f"\n[saved] {LOG}")

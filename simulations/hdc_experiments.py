"""
High-dimensional calibration: does the plug-in calibration matrix C_hat work in p >> n?

Theory being tested (new Section 5):
    C = Sigma_x (Sigma_x + Sigma_u)^{-1}   (oracle calibration)
    C_hat = Sigma_hat_x (Sigma_hat_x + Sigma_u)^{-1},  Sigma_hat_x = thresholded estimate
Claim chain:  ||Sigma_hat_x - Sigma_x||_2 = O_P(sqrt(log p / n))   [Bickel-Levina]
           => ||C_hat - C||_2 = O_P(sqrt(log p / n))              [perturbation lemma]
           => l1-QR on (C_hat W, 1) obeys the same oracle inequality as on (C W, 1).

Also reports the score diagnostic that the proof rests on:
    || (1/n) sum_i Xhat_i psi_i ||_inf   vs   || (1/n) sum_i mu_Wi psi_i ||_inf
where psi_i = tau - 1{Y_i - mu_Wi'beta* - delta_tau < 0}.

Design: X ~ N(0, Sigma_x) with Sigma_x banded Toeplitz (row sparsity 11), Sigma_u = sigma_u^2 I.
Estimators are exact linear-programming l1-quantile regressions.
"""

import io
import json
import time
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routeA_lib import draw_eps, calibration_matrix

from sklearn.linear_model import QuantileRegressor
from sklearn.covariance import LedoitWolf

LOG = "simulations/hdc_experiments_output.txt"
OUT = []


def log(m=""):
    print(m, flush=True)
    OUT.append(str(m))
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))


def banded_sigma_x(p, rho=0.5, half_band=5, ridge=0.5):
    idx = np.arange(p)
    D = np.abs(idx[:, None] - idx[None, :])
    S = np.where(D <= half_band, 0.5 * rho ** D, 0.0) + ridge * np.eye(p)
    return S


def gen(n, p, s, sigma_u, tau, Sigma_x, seed, beta_fixed=None):
    rng = np.random.RandomState(seed)
    L = np.linalg.cholesky(Sigma_x)
    X = rng.standard_normal((n, p)) @ L.T
    U = rng.standard_normal((n, p)) * sigma_u
    if beta_fixed is None:
        beta = np.zeros(p)
        beta[:s] = rng.randn(s) * 2.0
    else:
        beta = beta_fixed
    Y = X @ beta + draw_eps(n, tau, rng)
    return X, X + U, Y, beta


def threshold_estimate(W, Sigma_u, c_thresh=2.0, floor=1e-2):
    """Ledoit-Wolf shrinkage, subtract Sigma_u, hard-threshold off-diagonals
    (Bickel-Levina), floor the eigenvalues -> Sigma_hat_x."""
    p = W.shape[1]
    lw = LedoitWolf(assume_centered=False).fit(W)
    S = lw.covariance_ - Sigma_u
    n = W.shape[0]
    t = c_thresh * np.sqrt(np.log(p) / n)
    off = ~np.eye(p, dtype=bool)
    S[off] = np.where(np.abs(S[off]) > t, S[off], 0.0)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, floor)
    return V @ np.diag(w) @ V.T


def fit_l1(Y, D, tau, lam):
    q = QuantileRegressor(quantile=tau, alpha=lam, solver="highs")
    q.fit(D, Y)
    return q.coef_


def exp_hd(n=400, p=500, s=10, sigma_u_list=(0.1, 0.3, 0.5), tau=0.5, R=30,
           lam=0.02, seed0=9191):
    Sigma_x = banded_sigma_x(p)
    beta_star = np.zeros(p)
    beta_star[:s] = np.random.RandomState(seed0).randn(s) * 2.0
    log("=" * 100)
    log("HDC  high-dimensional calibration with an ESTIMATED Sigma_x")
    log("=" * 100)
    log(f"n = {n}, p = {p}, s = {s}, tau = {tau}, R = {R}, lambda = {lam}")
    log(f"Sigma_x: banded Toeplitz, half-band 5 (row sparsity 11), "
        f"lambda_min = {np.linalg.eigvalsh(Sigma_x).min():.3f}")
    log(f"||beta*||_2 = {np.linalg.norm(beta_star):.3f}")
    log("")
    res = {}
    for sigma_u in sigma_u_list:
        Sigma_u = sigma_u ** 2 * np.eye(p)
        C = calibration_matrix(Sigma_x, Sigma_u)
        beta_bar = C @ beta_star
        rows = {k: [] for k in ("naive", "oracle_cal", "plugin_cal", "norm_C_err",
                                "score_plugin", "score_oracle", "tpr", "fdr",
                                "pred_naive", "pred_oracle", "pred_plugin")}
        for rep in range(R):
            X, W, Y, bt = gen(n, p, s, sigma_u, tau, Sigma_x, seed0 + rep,
                              beta_fixed=beta_star)
            muW = W @ C.T
            # --- calibration matrix estimated from the data
            Sx_hat = threshold_estimate(W, Sigma_u)
            C_hat = Sx_hat @ np.linalg.inv(Sx_hat + Sigma_u)
            Xhat = W @ C_hat.T
            rows["norm_C_err"].append(float(np.linalg.norm(C_hat - C, 2)))

            b_naive = fit_l1(Y, W, tau, lam)
            b_or = fit_l1(Y, muW, tau, lam)
            b_pl = fit_l1(Y, Xhat, tau, lam)
            rows["naive"].append(np.linalg.norm(b_naive - bt))
            rows["oracle_cal"].append(np.linalg.norm(b_or - bt))
            rows["plugin_cal"].append(np.linalg.norm(b_pl - bt))
            rows["pred_naive"].append(float(np.mean((X @ (b_naive - bt)) ** 2)))
            rows["pred_oracle"].append(float(np.mean((X @ (b_or - bt)) ** 2)))
            rows["pred_plugin"].append(float(np.mean((X @ (b_pl - bt)) ** 2)))
            # support recovery on the penalised coordinates only
            def tpr_fdr(b):
                supp = np.abs(b) > 1e-6
                true = np.abs(bt) > 0
                tp = np.sum(supp & true)
                fp = np.sum(supp & ~true)
                return (tp / max(true.sum(), 1), fp / max(supp.sum(), 1))
            t, f = tpr_fdr(b_pl)
            rows["tpr"].append(t)
            rows["fdr"].append(f)
            # --- score diagnostic at the truth (calibrated model, delta_tau)
            e = Y - muW @ bt
            d_hat = float(np.quantile(e, tau))
            psi = tau - (e < d_hat).astype(float)
            rows["score_oracle"].append(float(np.max(np.abs(muW.T @ psi)) / n))
            rows["score_plugin"].append(float(np.max(np.abs(Xhat.T @ psi)) / n))
        row = {}
        for k, v in rows.items():
            v = np.array(v, float)
            row[k + "_mean"] = float(np.mean(v))
            row[k + "_median"] = float(np.median(v))
            row[k + "_se"] = float(np.std(v, ddof=1) / np.sqrt(R))
        row["beta_bar_dist"] = float(np.linalg.norm(beta_bar - beta_star))
        res[f"sigma_u{sigma_u}"] = row
        log(f"  --- sigma_u = {sigma_u}  (||beta_bar-beta*|| = {row['beta_bar_dist']:.3f}) ---")
        log(f"    ||C_hat - C||_2                     : {row['norm_C_err_mean']:.4f} "
            f"(median {row['norm_C_err_median']:.4f})")
        log(f"    ||beta_hat-beta*||_2  naive         : {row['naive_mean']:.4f} "
            f"+- {row['naive_se']:.4f}")
        log(f"    ||beta_hat-beta*||_2  oracle cal.   : {row['oracle_cal_mean']:.4f} "
            f"+- {row['oracle_cal_se']:.4f}")
        log(f"    ||beta_hat-beta*||_2  plug-in cal.  : {row['plugin_cal_mean']:.4f} "
            f"+- {row['plugin_cal_se']:.4f}")
        log(f"    prediction error      naive/oracle/plugin: "
            f"{row['pred_naive_mean']:.4f} / {row['pred_oracle_mean']:.4f} / "
            f"{row['pred_plugin_mean']:.4f}")
        log(f"    TPR {row['tpr_mean']:.3f}   FDR {row['fdr_mean']:.3f}  (plug-in)")
        log(f"    sup-norm score  oracle design       : {row['score_oracle_mean']:.5f}"
            f"  (theory: O(sqrt(log p/n)) = {np.sqrt(np.log(p)/n):.5f})")
        log(f"    sup-norm score  plug-in design      : {row['score_plugin_mean']:.5f}")
        log("")
    res["_meta"] = dict(n=n, p=p, s=s, tau=tau, R=R, lam=lam,
                        beta_star=beta_star.tolist(), sqrt_logp_over_n=float(np.sqrt(np.log(p) / n)),
                        sigma_u_list=list(sigma_u_list))
    with io.open("simulations/hdc_results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    t0 = time.time()
    exp_hd()
    log(f"[total {time.time()-t0:.1f}s]")
    print(f"\n[saved] {LOG}")

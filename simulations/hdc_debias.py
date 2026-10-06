"""
Debiased inference for the calibrated estimator: does it cover?

Setting (moderate dimension, p < n so that the plug-in Hessian is invertible and the
one-step debiasing is exact -- the nodewise approximation needed when p > n is discussed
in Section 6 of the paper and is not implemented here):

    Y_i = mu_Wi'beta* + e_i,   mu_W = C W,   C = Sigma_x (Sigma_x + Sigma_u)^{-1}

Estimators compared:
    l1            l1-penalised QR on (mu_W, 1); intervals from the usual sandwich, which
                  ignores the penalty bias
    debiased      bhat + H^{-1} S(bhat) with the smoothed score (He-Pan-Tan-Zhou 2023
                  construction, applied to the calibrated design)
    debiased-naive  the same construction on the RAW design W: debiasing removes the
                  l1 bias but not the attenuation, so its intervals should miss
Nominal level 95%; coverage is reported separately for the support and its complement.
"""

import io
import json
import time
import os
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routeA_lib import phi_std, calibration_matrix, draw_eps

from sklearn.linear_model import QuantileRegressor

LOG = "simulations/hdc_debias_output.txt"
OUT = []
Z95 = 1.959963985


def log(m=""):
    print(m, flush=True)
    OUT.append(str(m))
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))


def fit_l1(Y, D, tau, lam):
    q = QuantileRegressor(quantile=tau, alpha=lam, solver="highs")
    q.fit(D, Y)
    return q.coef_, float(q.intercept_)


def debias(Y, D, beta, delta, tau, h_scale=1.06, score="exact", trunc=0.0):
    """One-step debiasing: bhat = (beta, delta) + H^{-1} S(beta, delta).

    score = "exact"  : S uses the indicator score tau - 1{e < 0} (mean-zero at the truth,
                       no smoothing bias), while H uses the Gaussian kernel.
    score = "smooth" : S uses the smoothed score Phi(e/h) - (1-tau); this has an O(h^2)
                       bias which shifts the centre and, as our pilot showed, destroys
                       coverage unless the bias is corrected.

    trunc > 0 trims observations with |e| > trunc * sd(e) when forming H, which is the
    usual device for keeping the kernel weights away from zero.
    """
    n, p = D.shape
    e = Y - D @ beta - delta
    sd = float(np.std(e))
    h = max(h_scale * sd * n ** (-0.2), 1e-3)
    X = np.column_stack([D, np.ones(n)])
    z = e / h
    w = phi_std(z) / h                       # kernel weights for the Hessian
    if trunc > 0:
        keep = np.abs(e) <= trunc * sd
        w = w * keep
    if score == "exact":
        psi = tau - (e < 0).astype(float)
    else:
        psi = ndtr(z) - (1 - tau)
    H = (X.T * w) @ X / n
    S = X.T @ psi / n
    J = (X.T * psi ** 2) @ X / n
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Hinv = np.linalg.pinv(H)
    b = np.concatenate([beta, [delta]]) + Hinv @ S
    cov = Hinv @ J @ Hinv / n
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return b[:p], b[p], se[:p]


def run(n=400, p=40, s=5, beta_val=2.0, sigma_u_list=(0.3, 0.5), tau=0.5, R=200,
        lam=None, seed0=777):
    Sigma_x = np.eye(p)
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    if lam is None:
        lam = float(np.sqrt(np.log(p) / n))
    log("=" * 100)
    log("DEB  coverage of debiased calibrated intervals (p < n, exact one-step debiasing)")
    log("=" * 100)
    log(f"n = {n}, p = {p}, s = {s}, tau = {tau}, R = {R}, lambda = {lam:.4f}, "
        f"nominal level 95%")
    log("")
    res = {}
    for sigma_u in sigma_u_list:
        Sigma_u = sigma_u ** 2 * np.eye(p)
        C = calibration_matrix(Sigma_x, Sigma_u)
        rows = {k: [] for k in ("cov_sup_l1", "cov_sup_deb", "cov_sup_debsm",
                                "cov_sup_debnv", "cov_null_deb", "len_deb", "len_l1")}
        for rep in range(R):
            rng = np.random.RandomState(seed0 + rep * 7919)
            X = rng.standard_normal((n, p))
            W = X + rng.standard_normal((n, p)) * sigma_u
            Y = X @ beta_star + draw_eps(n, tau, rng)
            muW = W @ C.T
            b_l1, d_l1 = fit_l1(Y, muW, tau, lam)
            b_db, _, se_db = debias(Y, muW, b_l1, d_l1, tau, score="exact")
            _, _, se_sm = debias(Y, muW, b_l1, d_l1, tau, score="smooth")
            b_nv, d_nv = fit_l1(Y, W, tau, lam)
            b_dn, _, se_dn = debias(Y, W, b_nv, d_nv, tau, score="exact")
            # sandwich SE for the plain l1 fit (ignores the penalty bias)
            e = Y - muW @ b_l1 - d_l1
            h = max(1.06 * np.std(e) * n ** (-0.2), 1e-3)
            Xa = np.column_stack([muW, np.ones(n)])
            zz = e / h
            H = (Xa.T * (phi_std(zz) / h)) @ Xa / n
            J = (Xa.T * (ndtr(zz) - (1 - tau)) ** 2) @ Xa / n
            try:
                Hi = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                Hi = np.linalg.pinv(H)
            se_l1 = np.sqrt(np.maximum(np.diag(Hi @ J @ Hi / n), 0.0))[:p]
            sup = np.arange(s)
            null = np.arange(s, p)
            rows["cov_sup_l1"].append(np.mean(
                np.abs(b_l1[sup] - beta_star[sup]) <= Z95 * se_l1[sup]))
            rows["cov_sup_deb"].append(np.mean(
                np.abs(b_db[sup] - beta_star[sup]) <= Z95 * se_db[sup]))
            rows["cov_sup_debsm"].append(np.mean(
                np.abs(b_db[sup] - beta_star[sup]) <= Z95 * se_sm[sup]))
            rows["cov_sup_debnv"].append(np.mean(
                np.abs(b_dn[sup] - beta_star[sup]) <= Z95 * se_dn[sup]))
            rows["cov_null_deb"].append(np.mean(
                np.abs(b_db[null] - beta_star[null]) <= Z95 * se_db[null]))
            rows["len_deb"].append(float(np.mean(2 * Z95 * se_db[sup])))
            rows["len_l1"].append(float(np.mean(2 * Z95 * se_l1[sup])))
        row = {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(R)))
               for k, v in rows.items()}
        res[f"sigma_u{sigma_u}"] = {k: dict(mean=v[0], se=v[1]) for k, v in row.items()}
        log(f"  --- sigma_u = {sigma_u} ---")
        log(f"    coverage, support:  l1 {row['cov_sup_l1'][0]:.3f}  |  "
            f"debiased (calibrated) {row['cov_sup_deb'][0]:.3f}+-{row['cov_sup_deb'][1]:.3f}"
            f"  |  debiased (raw W) {row['cov_sup_debnv'][0]:.3f}")
        log(f"    coverage, null coordinates (debiased): {row['cov_null_deb'][0]:.3f}")
        log(f"    mean interval length (support): debiased {row['len_deb'][0]:.3f}, "
            f"l1 {row['len_l1'][0]:.3f}")
        log("")
    res["_meta"] = dict(n=n, p=p, s=s, tau=tau, R=R, lam=lam, beta_val=beta_val,
                        sigma_u_list=list(sigma_u_list))
    with io.open("simulations/hdc_debias.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    t0 = time.time()
    run()
    log(f"[total {time.time()-t0:.1f}s]")
    print(f"\n[saved] {LOG}")

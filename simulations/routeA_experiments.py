"""
Route-A experiments, part 2: finite-sample behaviour, high-dimensional comparison,
and the score check that pins down WHY the corrected loss fails.

E2  low-dimensional scaling (p=10 < n): does the convolved estimator's error floor at
    the attenuated target while the calibrated one keeps converging?
E3  high-dimensional comparison (n=400, p=500, s=10) with 50 reps, medians + IQR,
    each method penalised, plus the manuscript's own two-stage code as published.
E4  population score check at beta*: the convolved score has a non-zero mean of order
    Sigma_u beta*, the calibrated score has mean exactly zero.

Outputs JSON into simulations/routeA_results_*.json
"""

import io
import json
import sys
import time
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from routeA_lib import (fit_smooth_unpenalized, fit_smooth_l1,
                        fit_smooth_l1_intercept, gen_lowdim, gen_highdim,
                        draw_eps, phi_std, psi_h, rho_h,
                        calibration_matrix, posterior_cov)

H = 0.01                      # computational smoothing bandwidth (bias O(H^2) ~ 1e-4)
LOG = "simulations/routeA_experiments_output.txt"
OUT = []


def log(m=""):
    print(m, flush=True)
    OUT.append(str(m))
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))


# =============================================================== E2
def exp_E2(p=10, s=5, beta_val=2.0, sigma_u=0.5, taus=(0.3, 0.5, 0.7),
           ns=(500, 1000, 2000, 4000, 8000), R=50, seed0=1234):
    log("=" * 100)
    log("E2  low-dimensional scaling: p = %d, s = %d, sigma_u = %.2f, R = %d reps"
        % (p, s, sigma_u, R))
    log("    unpenalised fits (p < n), computational bandwidth h = %.3f" % H)
    log("    Stage 1 for the convolved method = calibrated fit on mu_W  (IDEALISED:")
    log("    the true Sigma_x is used, which FAVOURS the manuscript's method)")
    log("=" * 100)
    Sx = np.eye(p)
    Su = sigma_u ** 2 * np.eye(p)
    C = calibration_matrix(Sx, Su)
    Sxw = posterior_cov(Sx, Su)
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    sigma_star = float(np.sqrt(beta_star @ Su @ beta_star))
    beta_bar = C @ beta_star
    log(f"    ||beta*|| = {np.linalg.norm(beta_star):.4f},  sigma* = {sigma_star:.4f},  "
        f"||beta_bar - beta*|| = {np.linalg.norm(beta_bar - beta_star):.4f}")
    log("")
    hdr = (f"{'tau':>5} {'n':>6} | " + " | ".join(
        f"{m:>19}" for m in ("convolved(paper)", "convolved(||.||",
                             "naive", "RC", "RC+int", "oracle")))
    log(hdr)
    log("-" * len(hdr))
    res = {}
    for tau in taus:
        for n in ns:
            errs = {k: [] for k in ("conv", "conv_to_bar", "naive", "rc", "rc_int", "oracle")}
            sigmas = []
            for rep in range(R):
                X, W, Y, bt = gen_lowdim(n, p, s, sigma_u, tau, seed0 + rep * 7919 + n,
                                         beta_val)
                muW = W @ C.T
                D_int = np.column_stack([muW, np.ones(n)])
                # Stage 1 (idealised calibration) -> sigma_hat
                b_init, _ = fit_smooth_unpenalized(Y, muW, H, tau)
                sig_hat = float(np.sqrt(max(b_init @ Su @ b_init, 1e-12)))
                sigmas.append(sig_hat)
                b_conv, _ = fit_smooth_unpenalized(Y, W, sig_hat, tau)
                b_naive, _ = fit_smooth_unpenalized(Y, W, H, tau)
                b_rc, _ = fit_smooth_unpenalized(Y, muW, H, tau)
                b_rci, _ = fit_smooth_unpenalized(Y, D_int, H, tau)
                b_or, _ = fit_smooth_unpenalized(Y, X, H, tau)
                errs["conv"].append(np.linalg.norm(b_conv - bt))
                errs["conv_to_bar"].append(np.linalg.norm(b_conv - beta_bar))
                errs["naive"].append(np.linalg.norm(b_naive - bt))
                errs["rc"].append(np.linalg.norm(b_rc - bt))
                errs["rc_int"].append(np.linalg.norm(b_rci[:p] - bt))
                errs["oracle"].append(np.linalg.norm(b_or - bt))
            row = {}
            for k, v in errs.items():
                row[k] = float(np.mean(v))
                row[k + "_se"] = float(np.std(v, ddof=1) / np.sqrt(R))
            row["sigma_hat_mean"] = float(np.mean(sigmas))
            row["sigma_star"] = sigma_star
            res[f"tau{tau}_n{n}"] = row
            log(f"{tau:>5} {n:>6} | {row['conv']:>8.4f}+-{row['conv_se']:.4f} "
                f"{row['conv_to_bar']:>8.4f} | {row['naive']:>8.4f} | {row['rc']:>8.4f} | "
                f"{row['rc_int']:>8.4f} | {row['oracle']:>8.4f}")
    res["_meta"] = dict(p=p, s=s, sigma_u=sigma_u, R=R, ns=list(ns), taus=list(taus),
                        beta_star=beta_star.tolist(), beta_bar=beta_bar.tolist(),
                        sigma_star=sigma_star, h=H,
                        beta_bar_dist=float(np.linalg.norm(beta_bar - beta_star)))
    with io.open("simulations/routeA_results_E2.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log("")
    return res


# =============================================================== E3
def exp_E3(n=400, p=500, s=10, sigma_u_list=(0.1, 0.3, 0.5), tau=0.5, R=30,
           lam=0.02, seed0=4242, use_paper_stage1=True):
    """High-dimensional comparison.

    The three reference methods (naive, RC, oracle) are fitted with the EXACT
    linear-programming quantile-regression solver, so no conclusion can be attributed to
    optimisation error.  The published two-stage smoothed estimator is run through its own
    published code, unchanged.  All methods use the same penalty lambda = 0.02 and all
    include an intercept (sklearn's `fit_intercept=True`, as in the original scripts).
    """
    from sklearn.linear_model import QuantileRegressor
    log("=" * 100)
    log(f"E3  high-dimensional comparison: n = {n}, p = {p}, s = {s}, tau = {tau}, "
        f"R = {R}, lambda = {lam}")
    log("    exact LP fits for naive / RC / oracle; the smoothed method runs its own")
    log("    published two-stage code.  Fixed beta* across replications.")
    log("=" * 100)
    res = {}
    beta_star_used = np.zeros(p)
    beta_star_used[:s] = np.random.RandomState(seed0).randn(s) * 2.0
    for sigma_u in sigma_u_list:
        Su_mat = sigma_u ** 2 * np.eye(p)
        Su_vec = np.full(p, sigma_u ** 2)
        C = calibration_matrix(np.eye(p), Su_mat)
        beta_bar = C @ beta_star_used
        rows = {k: [] for k in ("conv_paper", "naive", "rc_int", "oracle",
                                "conv_paper_to_bar")}
        for rep in range(R):
            X, W, Y, bt = gen_highdim(n, p, s, sigma_u, tau, seed0 + rep,
                                      beta_fixed=beta_star_used)
            muW = W @ C.T

            def lp(D):
                q = QuantileRegressor(quantile=tau, alpha=lam, solver="highs")
                q.fit(D, Y)
                return q.coef_

            rows["naive"].append(np.linalg.norm(lp(W) - bt))
            rows["rc_int"].append(np.linalg.norm(lp(muW) - bt))
            rows["oracle"].append(np.linalg.norm(lp(X) - bt))

            if use_paper_stage1:
                from corrected_loss import two_stage_estimator
                r2 = two_stage_estimator(Y, W, Su_vec, tau=tau, lambda_reg=lam,
                                         step_size=0.005, max_iter=1000)
                b_cp = r2["beta"]
                rows["conv_paper"].append(np.linalg.norm(b_cp - bt))
                rows["conv_paper_to_bar"].append(np.linalg.norm(b_cp - beta_bar))
        row = {}
        for k, v in rows.items():
            v = np.array(v, float)
            row[k + "_median"] = float(np.median(v))
            row[k + "_mean"] = float(np.mean(v))
            row[k + "_q25"] = float(np.percentile(v, 25))
            row[k + "_q75"] = float(np.percentile(v, 75))
        row["beta_star"] = beta_star_used.tolist()
        row["beta_bar_dist"] = float(np.linalg.norm(beta_bar - beta_star_used))
        res[f"sigma_u{sigma_u}"] = row
        log(f"  sigma_u = {sigma_u}:  ||beta_bar-beta*|| = {row['beta_bar_dist']:.4f}")
        log(f"    smoothed (published two-stage)  median {row['conv_paper_median']:.4f}"
            f"  [distance to beta_bar: {row['conv_paper_to_bar_median']:.4f}]")
        log(f"    naive l1-QR on W (exact LP)     median {row['naive_median']:.4f}")
        log(f"    RC (calibrated) l1-QR (exact LP) median {row['rc_int_median']:.4f}")
        log(f"    oracle l1-QR on X (exact LP)    median {row['oracle_median']:.4f}")
        log("")
    res["_meta"] = dict(n=n, p=p, s=s, tau=tau, R=R, lam=lam, h=H,
                        sigma_u_list=list(sigma_u_list))
    with io.open("simulations/routeA_results_E3.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


# =============================================================== E4
def exp_E4(p=2, beta_val=2.0, sigma_u_list=(0.1, 0.3, 0.5), taus=(0.3, 0.5, 0.7),
           N=400_000, seed=808):
    log("=" * 100)
    log("E4  population score check at beta*  (exact ndtr, no smoothing)")
    log("    the convolved score psi = Phi((Y-W'b)/s) - (1-tau) is evaluated at the TRUE")
    log("    sigma*; the calibrated score uses mu_W and the check-loss step function")
    log("=" * 100)
    from scipy.special import ndtr
    beta_star = np.zeros(p)
    beta_star[0] = beta_val
    res = {}
    for sigma_u in sigma_u_list:
        Su = sigma_u ** 2 * np.eye(p)
        C = calibration_matrix(np.eye(p), Su)
        sigma_star = float(np.sqrt(beta_star @ Su @ beta_star))
        for tau in taus:
            rng = np.random.RandomState(seed + int(sigma_u * 100) + int(tau * 10))
            X = rng.standard_normal((N, p))
            U = rng.standard_normal((N, p)) * sigma_u
            W = X + U
            Y = X @ beta_star + draw_eps(N, tau, rng)
            muW = W @ C.T
            # convolved score at beta*
            g = ndtr((Y - W @ beta_star) / sigma_star) - (1 - tau)
            sW = W[:, 0] * g
            # calibrated score at beta* (check-loss indicator)
            gc = tau - ((Y - muW @ beta_star) < 0).astype(float)
            sMu = muW[:, 0] * gc
            # analytic prediction: E[W_j psi] = -(Sigma_u beta*)_j * d, where
            # d = density at 0 of eps + U'beta* + N(0,sigma*^2) = E_eps[phi(eps/(sqrt2
            # sigma*))]/(sqrt2 sigma*)   (smooth integrand -> ~0.1% accurate)
            tau_s = np.sqrt(2.0) * sigma_star
            ee = draw_eps(4_000_000, tau, np.random.RandomState(3))
            d = float(np.mean(phi_std(ee / tau_s) / tau_s))
            pred_W = -float((Su @ beta_star)[0]) * d
            mW, seW = float(np.mean(sW)), float(np.std(sW) / np.sqrt(N))
            mM, seM = float(np.mean(sMu)), float(np.std(sMu) / np.sqrt(N))
            res[f"su{sigma_u}_tau{tau}"] = dict(
                sigma_u=sigma_u, tau=tau, sigma_star=sigma_star,
                E_Wpsi=mW, se_Wpsi=seW, t_Wpsi=mW / seW, pred_E_Wpsi=pred_W,
                E_muW_psi=mM, se_muW_psi=seM, t_muW_psi=mM / seM)
            log(f"  sigma_u={sigma_u} tau={tau}:  E[W psi]    = {mW:+.5f} +- {seW:.5f} "
                f"(t = {mW/seW:+8.1f};  analytic {pred_W:+.5f})")
            log(f"                            E[mu_W psi] = {mM:+.5f} +- {seM:.5f} "
                f"(t = {mM/seM:+8.1f})")
    with io.open("simulations/routeA_results_E4.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log("")
    return res


if __name__ == "__main__":
    t0 = time.time()
    exp_E4()
    exp_E2()
    exp_E3()
    log(f"[total {time.time()-t0:.1f}s]")
    print(f"\n[saved] {LOG}")

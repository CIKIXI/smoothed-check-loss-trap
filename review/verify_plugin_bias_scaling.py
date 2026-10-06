r"""
The plug-in calibration bias with n/p held fixed: does |bias| / s.e. grow like sqrt(log p)?

Proposition 6.2 gives r_n <= 2 sigma_max(Sigma_u) b^2 s_0 eps_n with
eps_n = O_P(sqrt(log p / n)), so the violation of the debiasing condition is of order
s_0 sqrt(log p) ||beta*||_1, and the bias it produces relative to the standard error should
grow like sqrt(log p) as p grows at fixed n/p.  The earlier scan at fixed n = 400 broke down
for p >~ 100 because the l1 fit itself becomes unstable when n/p is small, so here n = 4p.

Run:  python review/verify_plugin_bias_scaling.py [R]
"""

import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations"))

from hdc_debias import fit_l1, debias, Z95                 # noqa: E402
from routeA_lib import calibration_matrix, draw_eps        # noqa: E402
from sklearn.linear_model import QuantileRegressor         # noqa: E402


def refit(Y, D, support, tau):
    idx = np.where(support)[0]
    b = np.zeros(D.shape[1])
    if len(idx) == 0:
        return b, float(np.quantile(Y, tau))
    q = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")
    q.fit(D[:, idx], Y)
    b[idx] = q.coef_
    return b, float(q.intercept_)


def run(R, p, ratio=4, s=5, beta_val=2.0, sigma_u=0.5, tau=0.5, seed0=24680):
    n = ratio * p
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    lam = float(np.sqrt(np.log(p) / n))
    Su = sigma_u ** 2 * np.eye(p)
    C = calibration_matrix(np.eye(p), Su)
    sup = np.arange(s)
    out = {t: {k: [] for k in ("bias", "se", "cov")} for t in ("known", "plug")}
    rn = []
    for rep in range(R):
        rng = np.random.RandomState(seed0 + rep * 7919)
        X = rng.standard_normal((n, p))
        W = X + rng.standard_normal((n, p)) * sigma_u
        Y = X @ beta_star + draw_eps(n, tau, rng)
        SW = W.T @ W / n
        C_hat = np.eye(p) - Su @ np.linalg.inv(SW)
        rn.append(float(np.max(np.sum(np.abs(C_hat - C), axis=1))))
        for tag, Cu in (("known", C), ("plug", C_hat)):
            muW = W @ Cu.T
            b_l1, d_l1 = fit_l1(Y, muW, tau, lam)
            support = np.abs(b_l1) > 1e-8
            b_rf, d_rf = refit(Y, muW, support, tau)
            b_rd, _, se_rd = debias(Y, muW, b_rf, d_rf, tau, score="exact")
            out[tag]["bias"].append((b_rd - beta_star)[sup])
            out[tag]["se"].append(se_rd[sup])
            out[tag]["cov"].append(float(np.mean(np.abs((b_rd - beta_star)[sup])
                                                 <= Z95 * se_rd[sup])))
    res = {"n": n, "p": p, "R": R, "r_n": float(np.mean(rn)),
           "sqrt_logp_over_n": float(np.sqrt(np.log(p) / n))}
    for tag in out:
        b = np.array(out[tag]["bias"]).ravel()
        se = np.array(out[tag]["se"]).ravel()
        res[tag] = dict(bias=float(b.mean()), se=float(se.mean()),
                        cov=float(np.mean(out[tag]["cov"])),
                        bias_se=float(abs(b.mean()) / se.mean()),
                        bias_se_mc=float(b.std(ddof=1) / np.sqrt(len(b)) / se.mean()))
    return res


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    t0 = time.time()
    print(f"n = 4p throughout, s = 5, sigma_u = 0.5, tau = 1/2, R = {R}")
    print(f"{'p':>5} {'n':>6} {'sqrt(log p)':>11} {'r_n':>7} | {'known cov':>9} "
          f"{'plug cov':>9} {'plug bias':>10} {'|bias|/se':>10}")
    rows = []
    for p in (25, 50, 100, 200):
        r = run(R, p)
        rows.append(r)
        print(f"{r['p']:5d} {r['n']:6d} {np.sqrt(np.log(r['p'])):11.3f} {r['r_n']:7.3f} | "
              f"{r['known']['cov']:9.3f} {r['plug']['cov']:9.3f} "
              f"{r['plug']['bias']:+10.4f} {r['plug']['bias_se']:10.3f}")
    with io.open("review/verify_plugin_bias_scaling.json", "w", encoding="utf-8") as fh:
        json.dump(dict(R=R, rows=rows), fh, indent=1)
    print(f"\n[total {time.time()-t0:.1f}s]  [saved] review/verify_plugin_bias_scaling.json")


if __name__ == "__main__":
    main()

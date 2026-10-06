r"""
Does the *plug-in* calibration matrix break the debiased intervals?

Section 6 of the paper validates the debiased intervals with the *true* calibration matrix
C = Sigma_x (Sigma_x + Sigma_u)^{-1} (simulations/hdc_debias_final.py and hdc_pgtn.py both
call calibration_matrix(...) and use it to build mu_W).  The theorem that Section 6 states
is about an *estimated* calibration, so this experiment asks what changes when C is replaced
by the natural method-of-moments plug-in

        C_hat = I - Sigma_u S_W^{-1},      S_W = (1/n) sum_i W_i W_i',

which uses only the known Sigma_u.  Two things are recorded:

  (1) coverage of the debiased intervals on the signal coordinates, known C versus C_hat;
  (2) the realised bias of the debiased estimator, and the leading bias predicted by the
      expansion of the score at the estimated design,
          E[S_j(beta*)] = -f_e(delta_tau) e_j' Sigma_x (C_hat - C)' beta* + o(...),
      together with its ratio to the standard error.

Run:  python review/verify_plugin_calibration_bias.py [R]
"""

import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations"))

from hdc_debias import fit_l1, debias, Z95                     # noqa: E402
from routeA_lib import calibration_matrix, draw_eps, phi_std   # noqa: E402
from sklearn.linear_model import QuantileRegressor             # noqa: E402


def refit(Y, D, support, tau):
    idx = np.where(support)[0]
    b = np.zeros(D.shape[1])
    if len(idx) == 0:
        return b, float(np.quantile(Y, tau))
    q = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")
    q.fit(D[:, idx], Y)
    b[idx] = q.coef_
    return b, float(q.intercept_)


def one_rep(rng, n, p, s, beta_val, sigma_u, tau=0.5):
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    Su = sigma_u ** 2 * np.eye(p)
    C = calibration_matrix(np.eye(p), Su)
    X = rng.standard_normal((n, p))
    W = X + rng.standard_normal((n, p)) * sigma_u
    Y = X @ beta_star + draw_eps(n, tau, rng)
    SW = W.T @ W / n
    C_hat = np.eye(p) - Su @ np.linalg.inv(SW)
    return dict(Y=Y, W=W, beta_star=beta_star, C=C, C_hat=C_hat, Su=Su)


def run(R=200, n=400, p=40, s=5, beta_val=2.0, sigma_u=0.5, tau=0.5, seed0=24680):
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    lam = float(np.sqrt(np.log(p) / n))
    acc = {k: [] for k in ("cov_known", "cov_plug", "bias_known", "bias_plug",
                           "se_known", "se_plug", "pred_bias", "r_n", "pred_rn",
                           "est_err_known", "est_err_plug")}
    for rep in range(R):
        rng = np.random.RandomState(seed0 + rep * 7919)
        d = one_rep(rng, n, p, s, beta_val, sigma_u, tau)
        Y, W, C, C_hat = d["Y"], d["W"], d["C"], d["C_hat"]
        sup = np.arange(s)

        # --- row-wise calibration error and the predicted score bias
        D = C_hat - C
        r_n = float(np.max(np.sum(np.abs(D), axis=1)))
        w = np.linalg.solve(np.eye(p) + d["Su"], d["Su"] @ beta_star)
        pred = np.abs(C @ D @ w)                      # C (C_hat - C)' beta* approximately
        acc["r_n"].append(r_n)
        acc["pred_rn"].append(np.max(np.abs(D)) * np.sqrt(p))

        for tag, Cuse in (("known", C), ("plug", C_hat)):
            muW = W @ Cuse.T
            b_l1, d_l1 = fit_l1(Y, muW, tau, lam)
            support = np.abs(b_l1) > 1e-8
            b_rf, d_rf = refit(Y, muW, support, tau)
            b_rd, _, se_rd = debias(Y, muW, b_rf, d_rf, tau, score="exact")
            acc[f"cov_{tag}"].append(float(np.mean(
                np.abs(b_rd[sup] - beta_star[sup]) <= Z95 * se_rd[sup])))
            acc[f"bias_{tag}"].append(float(np.mean(b_rd[sup] - beta_star[sup])))
            acc[f"se_{tag}"].append(float(np.mean(se_rd[sup])))
            acc[f"est_err_{tag}"].append(float(np.max(np.abs(b_rd[sup] - beta_star[sup]))))
        acc["pred_bias"].append(float(np.max(pred)))
    out = {}
    for k, v in acc.items():
        v = np.array(v)
        out[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                      se=float(v.std(ddof=1) / np.sqrt(R)))
    return out


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t0 = time.time()
    print("=" * 96)
    print("Debiased intervals: known calibration vs method-of-moments plug-in")
    print("=" * 96)
    res = {}
    for sigma_u in (0.3, 0.5):
        r = run(R=R, sigma_u=sigma_u)
        res[f"sigma_u={sigma_u}"] = r
        print(f"\n--- sigma_u = {sigma_u}, n = 400, p = 40, s = 5, R = {R} ---")
        print(f"  coverage on the signal:  known C {r['cov_known']['mean']:.3f} "
              f"({r['cov_known']['se']:.3f})   plug-in {r['cov_plug']['mean']:.3f} "
              f"({r['cov_plug']['se']:.3f})")
        print(f"  mean debiased bias:      known C {r['bias_known']['mean']:+.4f}   "
              f"plug-in {r['bias_plug']['mean']:+.4f}")
        print(f"  mean standard error:     known C {r['se_known']['mean']:.4f}   "
              f"plug-in {r['se_plug']['mean']:.4f}")
        print(f"  bias / s.e.:             known C "
              f"{abs(r['bias_known']['mean'])/r['se_known']['mean']:.3f}   plug-in "
              f"{abs(r['bias_plug']['mean'])/r['se_plug']['mean']:.3f}")
        print(f"  |bias|/se of the mean bias estimate: "
              f"{r['bias_plug']['se']/r['se_plug']['mean']:.4f}")
        print(f"  row-wise calibration error r_n: {r['r_n']['mean']:.4f} "
              f"(sqrt(log p / n) = {np.sqrt(np.log(40)/400):.4f})")
        print(f"  predicted sup-norm score bias |C(C_hat-C)'beta*|_inf: "
              f"{r['pred_bias']['mean']:.4f}")
    with io.open("review/verify_plugin_calibration_bias.json", "w", encoding="utf-8") as fh:
        json.dump(dict(R=R, res=res), fh, indent=1)
    print(f"\n[total {time.time()-t0:.1f}s]  [saved] review/verify_plugin_calibration_bias.json")


if __name__ == "__main__":
    main()

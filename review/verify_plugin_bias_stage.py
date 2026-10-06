r"""
Localise the plug-in calibration bias: is it in the penalty stage, the refit, or the
debiasing step?

Run:  python review/verify_plugin_bias_stage.py [R]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations"))

from hdc_debias import fit_l1, debias, Z95, phi_std   # noqa: E402
from routeA_lib import calibration_matrix, draw_eps   # noqa: E402
from sklearn.linear_model import QuantileRegressor    # noqa: E402


def refit(Y, D, support, tau):
    idx = np.where(support)[0]
    b = np.zeros(D.shape[1])
    if len(idx) == 0:
        return b, float(np.quantile(Y, tau))
    q = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")
    q.fit(D[:, idx], Y)
    b[idx] = q.coef_
    return b, float(q.intercept_)


def run(R=48, n=400, p=40, s=5, beta_val=2.0, sigma_u=0.5, tau=0.5, seed0=24680,
        lam_factor=1.0):
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    lam = lam_factor * float(np.sqrt(np.log(p) / n))
    Su = sigma_u ** 2 * np.eye(p)
    C = calibration_matrix(np.eye(p), Su)
    out = {}
    for tag in ("known", "plug"):
        out[tag] = {k: [] for k in ("l1", "rf", "deb", "se")}
    for rep in range(R):
        rng = np.random.RandomState(seed0 + rep * 7919)
        X = rng.standard_normal((n, p))
        W = X + rng.standard_normal((n, p)) * sigma_u
        Y = X @ beta_star + draw_eps(n, tau, rng)
        SW = W.T @ W / n
        C_hat = np.eye(p) - Su @ np.linalg.inv(SW)
        for tag, Cuse in (("known", C), ("plug", C_hat)):
            muW = W @ Cuse.T
            b_l1, d_l1 = fit_l1(Y, muW, tau, lam)
            support = np.abs(b_l1) > 1e-8
            b_rf, d_rf = refit(Y, muW, support, tau)
            b_rd, _, se_rd = debias(Y, muW, b_rf, d_rf, tau, score="exact")
            sup = np.arange(s)
            out[tag]["l1"].append((b_l1 - beta_star)[sup])
            out[tag]["rf"].append((b_rf - beta_star)[sup])
            out[tag]["deb"].append((b_rd - beta_star)[sup])
            out[tag]["se"].append(se_rd[sup])
    res = {}
    for tag in out:
        res[tag] = {k: np.array(v) for k, v in out[tag].items()}
    return res


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    for sigma_u in (0.5, 0.3):
        for lf in (1.0, 0.5):
            r = run(R=R, sigma_u=sigma_u, lam_factor=lf)
            print("=" * 92)
            print(f"sigma_u = {sigma_u}, lambda = {lf} x sqrt(log p / n), R = {R}")
            for tag in ("known", "plug"):
                d = r[tag]
                print(f"  {tag:6s} mean bias on the 5 signal coords:")
                print(f"         l1     {np.round(d['l1'].mean(axis=0), 4)}")
                print(f"         refit  {np.round(d['rf'].mean(axis=0), 4)}")
                print(f"         debias {np.round(d['deb'].mean(axis=0), 4)}"
                      f"   mean s.e. {d['se'].mean():.4f}")
                print(f"         coverage of the debiased interval: "
                      f"{np.mean(np.abs(d['deb']) <= Z95 * d['se']):.3f}")


if __name__ == "__main__":
    main()

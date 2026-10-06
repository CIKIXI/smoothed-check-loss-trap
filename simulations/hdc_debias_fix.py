"""Fix attempt: the one-step debiasing needs an initial estimator whose bias is small
relative to its standard error.  The plain l1-QR at lambda = sqrt(log p/n) has a bias of
about -0.93 on a coefficient of size 2 (diagnosed in hdc_debias_diag.py), which the
one-step correction then overshoots.

Two standard remedies are tested:
  (a) a smaller penalty lambda;
  (b) a REFITTED initial estimator: take the support selected by the l1 fit and refit by
      unpenalised QR on those coordinates (post-selection refit), then debias once.
"""

import sys
import numpy as np
sys.path.insert(0, "simulations")
from hdc_debias import fit_l1, debias, Z95
from routeA_lib import calibration_matrix, draw_eps
from sklearn.linear_model import QuantileRegressor

n, p, s, beta_val, tau = 400, 40, 5, 2.0, 0.5
sigma_u = 0.5
beta_star = np.zeros(p)
beta_star[:s] = beta_val
C = calibration_matrix(np.eye(p), sigma_u ** 2 * np.eye(p))
R = 80
sup = np.arange(s)


def refit(Y, muW, support, tau):
    """Unpenalised QR on the selected coordinates (plus intercept)."""
    idx = np.where(support)[0]
    if len(idx) == 0:
        return np.zeros(muW.shape[1]), 0.0
    q = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")
    q.fit(muW[:, idx], Y)
    b = np.zeros(muW.shape[1])
    b[idx] = q.coef_
    return b, float(q.intercept_)


def run(lam_list=(0.02, 0.05, 0.096)):
    print(f"n={n}, p={p}, s={s}, sigma_u={sigma_u}, R={R}, tau={tau}")
    print(f"{'lambda':>8} | {'l1 bias':>9} {'debias bias':>12} {'debias cov':>11} | "
          f"{'refit bias':>11} {'refit+debias bias':>18} {'cov':>7} {'|S|':>5}")
    for lam in lam_list:
        acc = {k: [] for k in ("b_l1", "b_db", "cov_db", "b_rf", "b_rfd", "cov_rfd", "nsup")}
        for rep in range(R):
            rng = np.random.RandomState(4321 + rep * 7919)
            X = rng.standard_normal((n, p))
            W = X + rng.standard_normal((n, p)) * sigma_u
            Y = X @ beta_star + draw_eps(n, tau, rng)
            muW = W @ C.T
            b_l1, d_l1 = fit_l1(Y, muW, tau, lam)
            b_db, _, se_db = debias(Y, muW, b_l1, d_l1, tau, score="exact")
            support = np.abs(b_l1) > 1e-8
            b_rf, d_rf = refit(Y, muW, support, tau)
            b_rfd, _, se_rfd = debias(Y, muW, b_rf, d_rf, tau, score="exact")
            acc["b_l1"].append(float(np.mean(b_l1[sup] - beta_star[sup])))
            acc["b_db"].append(float(np.mean(b_db[sup] - beta_star[sup])))
            acc["cov_db"].append(float(np.mean(np.abs(b_db[sup] - beta_star[sup]) <= Z95 * se_db[sup])))
            acc["b_rf"].append(float(np.mean(b_rf[sup] - beta_star[sup])))
            acc["b_rfd"].append(float(np.mean(b_rfd[sup] - beta_star[sup])))
            acc["cov_rfd"].append(float(np.mean(np.abs(b_rfd[sup] - beta_star[sup]) <= Z95 * se_rfd[sup])))
            acc["nsup"].append(int(support.sum()))
        print(f"{lam:>8.3f} | {np.mean(acc['b_l1']):>+9.4f} {np.mean(acc['b_db']):>+12.4f} "
              f"{np.mean(acc['cov_db']):>11.3f} | {np.mean(acc['b_rf']):>+11.4f} "
              f"{np.mean(acc['b_rfd']):>+18.4f} {np.mean(acc['cov_rfd']):>7.3f} "
              f"{np.mean(acc['nsup']):>5.1f}")


if __name__ == "__main__":
    run()

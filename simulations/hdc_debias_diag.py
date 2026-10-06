"""Diagnostic: why does the one-step debiasing fail on the calibrated design while it
works on the raw design?  Isolates (a) the l1 initial bias, (b) the debiasing machinery,
by also running the machinery from an oracle initialisation."""

import sys
import numpy as np
sys.path.insert(0, "simulations")
from hdc_debias import fit_l1, debias, Z95
from routeA_lib import calibration_matrix, draw_eps

n, p, s, beta_val, tau = 400, 40, 5, 2.0, 0.5
sigma_u = 0.5
lam = float(np.sqrt(np.log(p) / n))
beta_star = np.zeros(p)
beta_star[:s] = beta_val
C = calibration_matrix(np.eye(p), sigma_u ** 2 * np.eye(p))
R = 80
acc = {k: [] for k in ("bias_l1", "bias_deb", "bias_deb_oracle", "se_deb", "se_deb_oracle",
                       "cov_deb", "cov_deb_oracle", "cov_oracle_clean")}
for rep in range(R):
    rng = np.random.RandomState(1234 + rep * 7919)
    X = rng.standard_normal((n, p))
    W = X + rng.standard_normal((n, p)) * sigma_u
    Y = X @ beta_star + draw_eps(n, tau, rng)
    muW = W @ C.T
    b_l1, d_l1 = fit_l1(Y, muW, tau, lam)
    b_db, _, se_db = debias(Y, muW, b_l1, d_l1, tau, score="exact")
    # oracle initialisation: start the debiasing machinery at the truth
    e0 = Y - muW @ beta_star
    d0 = float(np.quantile(e0, tau))
    b_or, _, se_or = debias(Y, muW, beta_star, d0, tau, score="exact")
    sup = np.arange(s)
    acc["bias_l1"].append(float(np.mean(b_l1[sup] - beta_star[sup])))
    acc["bias_deb"].append(float(np.mean(b_db[sup] - beta_star[sup])))
    acc["bias_deb_oracle"].append(float(np.mean(b_or[sup] - beta_star[sup])))
    acc["se_deb"].append(float(np.mean(se_db[sup])))
    acc["se_deb_oracle"].append(float(np.mean(se_or[sup])))
    acc["cov_deb"].append(float(np.mean(np.abs(b_db[sup] - beta_star[sup]) <= Z95 * se_db[sup])))
    acc["cov_deb_oracle"].append(float(np.mean(np.abs(b_or[sup] - beta_star[sup]) <= Z95 * se_or[sup])))
    acc["cov_oracle_clean"].append(float(np.mean(np.abs(beta_star[sup] - beta_star[sup]) <= Z95 * se_or[sup])))

print(f"lambda = {lam:.4f}, n={n}, p={p}, s={s}, sigma_u={sigma_u}, R={R}")
for k, v in acc.items():
    print(f"  {k:>17}: mean {np.mean(v):+.4f}  sd {np.std(v):.4f}")
print()
print("Reading: if bias_deb_oracle is ~0 and cov_deb_oracle ~0.95, the debiasing machinery")
print("is fine and the failure comes from the l1 initial bias; if bias_deb_oracle is not")
print("~0, the machinery itself is mis-specified.")

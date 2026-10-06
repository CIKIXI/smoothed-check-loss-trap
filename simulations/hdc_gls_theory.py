"""Theoretical gain from combining QR estimators across quantile levels (GLS).

In the location-shift model every tau estimates the same beta*, and the joint asymptotic
covariance of the slope estimators is, in units of (E[mu_W mu_W'])^-1,

    C_{kl} = (tau_k ^ tau_l - tau_k tau_l) / (f_k f_l),      f_k = f_e(Q_{tau_k}(e)).

The GLS combination has variance (1' C^{-1} 1)^{-1}; the ratio to the single-level
variance C_{tau0,tau0} measures what is gained by combining.
"""

import numpy as np
from scipy.stats import norm, laplace, uniform


def gls_ratio(taus, dist):
    taus = list(taus)
    K = len(taus)
    q = dist.ppf(taus)
    f = dist.pdf(q)
    C = np.array([[(min(taus[k], taus[l]) - taus[k] * taus[l]) / (f[k] * f[l])
                   for l in range(K)] for k in range(K)])
    Ci = np.linalg.inv(C)
    one = np.ones(K)
    var_comb = 1.0 / (one @ Ci @ one)
    k0 = int(np.argmin(np.abs(np.array(taus) - 0.5)))
    return var_comb / C[k0, k0]


if __name__ == "__main__":
    grids = {"3 levels": [0.4, 0.5, 0.6],
             "7 levels": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
             "9 levels": list(np.round(np.arange(0.1, 0.91, 0.1), 3))}
    dists = {"Laplace": laplace(0, 1), "Gaussian": norm(0, 1),
             "Uniform": uniform(-1, 1)}
    print("GLS variance ratio (combined / median-only) in the location-shift model")
    for dn, d in dists.items():
        cells = " | ".join(f"{gn}: {gls_ratio(g, d):.4f}" for gn, g in grids.items())
        print(f"{dn:9s} | {cells}")

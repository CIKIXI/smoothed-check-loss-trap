"""
Does the local expansion of the manuscript reproduce the exact target?

Scalar design: X ~ N(0,1), U ~ N(0,sigma_u^2), beta* = 2, eps ~ Laplace(0,1), tau = 1/2.
The exact population minimiser of the fixed-scale smoothed loss is beta_bar = beta*/(1+sigma_u^2)
(verified to 1e-11 by quadrature).  The manuscript's Remark 3.8 combines the local expansion
with the Hessian identity

    H = (Sigma_x + Sigma_u) d(sigma) + (Sigma_u beta*)(Sigma_u beta*)' E[h''(U'beta*)]

to conclude beta_bar_sigma = beta* - H^{-1} g + O(rho^{3/2}) and reads the leading term as
beta_bar.  This script computes g and H by Monte Carlo and checks whether H^{-1} g equals the
true displacement beta* - beta_bar.
"""

import numpy as np
from scipy.integrate import quad

BETA = 2.0
TAU = 0.5


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / np.sqrt(2 * np.pi)


def d_sigma(sigma, sig2):
    s = np.sqrt(sig2 + sigma ** 2)
    return float(quad(lambda e: (np.exp(-0.5 * (e / s) ** 2) / (s * np.sqrt(2 * np.pi)))
                      * 0.5 * np.exp(-abs(e)), -60, 60, limit=400)[0])


for su in (0.3, 0.5):
    Su = su ** 2
    sig2 = BETA ** 2 * Su                      # sigma*^2 = beta*' Sigma_u beta*
    beta_bar = BETA / (1 + Su)
    n = 20_000_000
    rng = np.random.RandomState(3)
    X = rng.standard_normal(n)
    U = su * rng.standard_normal(n)
    W = X + U
    eps = rng.laplace(0, 1, n)
    r = eps - U * BETA                          # r*
    print("=" * 96)
    print(f"sigma_u = {su}:  exact argmin beta_bar = {beta_bar:.6f},  "
          f"displacement beta* - beta_bar = {BETA - beta_bar:.6f}")
    for sigma in (beta_bar * su, 1.0):
        psi = np.array([0.0])
        from scipy.special import ndtr
        psi = ndtr(r / sigma) - (1 - TAU)
        g = -float(np.mean(W * psi))            # grad L(beta*) = -E[W psi]
        H = float(np.mean(W ** 2 * phi(r / sigma) / sigma))
        d = d_sigma(sigma, sig2)
        # Hessian identity: (Sigma_x+Sigma_u) d + (Sigma_u beta*)^2 E[h'']
        h2_term = H - (1 + Su) * d
        step = BETA - H ** (-1) * g             # leading-order prediction of the minimiser
        print(f"  sigma = {sigma:5.3f}")
        print(f"     g = grad L(beta*)      MC {g:+.6f}   theorem {Su * BETA * d:+.6f}")
        print(f"     H = grad^2 L(beta*)    MC {H:+.6f}   (Sigma_x+Sigma_u)d = "
              f"{(1 + Su) * d:+.6f}   rank-one term {h2_term:+.6f}")
        print(f"     beta* - H^-1 g         = {step:.6f}   (exact minimiser {beta_bar:.6f}, "
              f"difference {step - beta_bar:+.6f})")
        print(f"     h'' term / d           = {h2_term / d:+.6f}  "
              f"(manuscript says this is O(rho^1/2) relative to d)")

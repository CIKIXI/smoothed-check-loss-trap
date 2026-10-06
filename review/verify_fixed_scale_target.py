"""
Exact population minimisers of the smoothed losses (quadrature, no Monte Carlo).

For a candidate b the residual of the fixed-scale smoothed loss is

    Y - W b - sigma Z = eps + X(beta*-b) - U b - sigma Z,
    i.e.  eps + N(0, s^2(b)),   s^2(b) = (beta*-b)^2 + sigma_u^2 b^2 + sigma^2,

so the population loss is a function of b only through s(b), and

    E[rho_tau(eps + s Z)] = (tau-1/2) E[eps] + (1/2) E|eps + s Z|,

which is strictly increasing in s.  Hence the population minimiser is the minimiser of
s^2(b).  This script evaluates that minimiser exactly (quadrature + fine golden search) and
also computes it for the adaptive scale sigma(b) = |b| sigma_u, where the Gaussian variance
becomes (beta*-b)^2 + 2 sigma_u^2 b^2.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BETA_STAR = 2.0


def abs_moment(s, tau, eps_kind):
    """E|eps + s Z| for Z ~ N(0,1) and eps Laplace(0,1) or asymmetric Laplace."""
    def normal_abs(e):
        z = e / s
        return e * (2 * ndtr(z) - 1) + 2 * s * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    e = np.linspace(-80, 80, 600001)
    if eps_kind == 0.5:
        f = 0.5 * np.exp(-np.abs(e))
    else:                       # asymmetric Laplace with quantile 0 at tau
        tau = eps_kind
        lam_l, lam_r = 1.0 / tau, 1.0 / (1 - tau)
        f = np.where(e < 0, tau * lam_l * np.exp(lam_l * e),
                     (1 - tau) * lam_r * np.exp(-lam_r * e))
    vals = normal_abs(e)
    return float(np.trapz(f * vals, e))


def population_loss(b, sigma_u, tau, sigma_fun):
    s = np.sqrt((BETA_STAR - b) ** 2 + sigma_u ** 2 * b ** 2 + sigma_fun(b) ** 2)
    return 0.5 * abs_moment(s, tau, tau)


def minimiser(sigma_u, tau, sigma_fun, lo=0.8, hi=2.6):
    r = minimize_scalar(lambda b: population_loss(b, sigma_u, tau, sigma_fun),
                        bounds=(lo, hi), method="bounded",
                        options=dict(xatol=1e-10))
    return float(r.x), float(r.fun)


def s2_minimiser(sigma_u, sigma_const=None, adaptive=False):
    """argmin of s^2(b) in closed form."""
    if adaptive:
        # (beta*-b)^2 + 2 sigma_u^2 b^2
        return BETA_STAR / (1.0 + 2 * sigma_u ** 2)
    return BETA_STAR / (1.0 + sigma_u ** 2)


print("Fixed-scale smoothed loss: exact population minimiser vs beta_bar and vs s^2 argmin")
print(f"{'sigma_u':>8} {'tau':>5} {'sigma':>7} {'quadrature':>12} {'beta_bar':>9} "
      f"{'s^2 argmin':>11} {'quad-beta_bar':>14}")
for sigma_u in (0.1, 0.3, 0.5):
    beta_bar = BETA_STAR / (1 + sigma_u ** 2)
    for tau in (0.3, 0.5, 0.7):
        for sig_lab, sig in (("pilot", beta_bar * sigma_u), ("beta*", BETA_STAR * sigma_u),
                             ("1.0", 1.0)):
            b, val = minimiser(sigma_u, tau, lambda bb, s=sig: s)
            b_s2 = s2_minimiser(sigma_u)
            print(f"{sigma_u:8.2f} {tau:5.2f} {sig_lab:>7} {b:12.6f} {beta_bar:9.4f} "
                  f"{b_s2:11.4f} {b - beta_bar:14.2e}")
print()
print("Adaptive scale sigma(b) = |b| sigma_u")
for sigma_u in (0.3, 0.5):
    for tau in (0.3, 0.5, 0.7):
        b, _ = minimiser(sigma_u, tau, lambda bb, s=sigma_u: s * abs(bb))
        pred = s2_minimiser(sigma_u, adaptive=True)
        print(f"  sigma_u={sigma_u:4.2f} tau={tau:3.1f}: quadrature argmin {b:.6f}, "
              f"closed form beta*/(1+2 sigma_u^2) = {pred:.4f}")

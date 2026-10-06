"""
M6: what error law do the simulations actually use, and what are the correct analytic
values for the naive score?

draw_eps(tau) returns
    tau = 1/2 : Laplace(0, 1)                     density at 0 = 0.5
    tau != 1/2: (1-tau)*Exp(1) - tau*Exp(1)       density at 0 = 1,  ALD scale tau(1-tau)

For X = (1-tau)E1 - tau*E2 with E_i ~ Exp(1) the density is
    f(x) = exp(-rho_tau(x / b)) / b?   ->  directly:  f(x) = exp(-x/(1-tau))  (x>0),
                                              f(x) = exp(x/tau)            (x<0),
which is the asymmetric Laplace with f(0) = 1.

The naive score at the truth is  E[W_j (tau - 1{r*<0})] = -(Sigma_u beta*)_j d_naive,
d_naive = (N(0, sigma*^2) * f_eps)(0) = E[phi_{sigma*}(eps)].
This script evaluates d_naive in closed form (by quadrature) for each of the three tau
values used in the manuscript and compares with the Monte-Carlo values and with the
"analytic" column printed in the paper.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

SIG_STAR = 1.0          # sigma*^2 = beta*' Sigma_u beta* = 4 * 0.25


def f_eps_laplace1(x):
    return 0.5 * np.exp(-np.abs(x))


def f_eps_sampler(x, tau):
    """density of (1-tau)Exp(1) - tau*Exp(1)"""
    x = np.asarray(x, float)
    return np.where(x > 0, np.exp(-x / (1 - tau)), np.exp(x / tau))


def d_naive(tau, kind):
    """(N(0, sigma*^2) * f_eps)(0) = E[phi_{sigma*}(eps)]"""
    s = SIG_STAR
    g = lambda e: (np.exp(-0.5 * (e / s) ** 2) / (s * np.sqrt(2 * np.pi))) * kind(e, tau)
    return float(quad(g, -80, 80, limit=400)[0])


print("naive score  E[W_1 (tau - 1{r*<0})] = -(Sigma_u beta*)_1 * d_naive,")
print("(Sigma_u beta*)_1 = 0.25 * 2 = 0.5,  sigma* = 1")
print()
print(f"{'tau':>5} {'law':<28} {'d_naive':>9} {'predicted score':>16}")
for tau in (0.3, 0.5, 0.7):
    if tau == 0.5:
        d = d_naive(tau, lambda e, t: f_eps_laplace1(e))
        law = "Laplace(0,1)  [what the code uses]"
    else:
        d = d_naive(tau, lambda e, t: f_eps_sampler(e, t))
        law = f"(1-tau)E1 - tau E2, tau={tau}"
    print(f"{tau:5.1f} {law:<28} {d:9.5f} {-0.5 * d:16.5f}")

print()
print("for comparison, the unit-scale ALD  f(x) = tau(1-tau) exp(-rho_tau(x))  (scale 1):")
for tau in (0.3, 0.7):
    g = lambda e: (np.exp(-0.5 * e ** 2) / np.sqrt(2 * np.pi)) * \
        tau * (1 - tau) * np.exp(-(e * (tau - (e < 0))))
    d = float(quad(g, -80, 80, limit=400)[0])
    print(f"  tau={tau}: d_naive = {d:.5f},  score = {-0.5 * d:.5f}")

print()
print("Monte Carlo check of the law actually used (tau=0.3):")
rng = np.random.RandomState(7)
n = 4_000_000
eps = (1 - 0.3) * rng.exponential(1.0, n) - 0.3 * rng.exponential(1.0, n)
for q in (0.3, 0.5, 0.9):
    print(f"   quantile {q}: {np.quantile(eps, q):+.5f}   (should be 0 at q=tau)")
print(f"   density at 0 (kernel, h=0.02): "
      f"{np.mean(np.abs(eps) < 0.02) / 0.04:.4f}")

"""Numerical check of Knight's identity as written in Step 3 of the proof of Theorem 5.4."""

import numpy as np

rng = np.random.RandomState(5)
tau = 0.3


def rho(u):
    u = np.asarray(u, float)
    return u * (tau - (u < 0).astype(float))


def lhs(e, v):
    return float(rho(e - v) - rho(e))


def rhs_path(e, v, n=200001):
    if v == 0:
        return 0.0
    s = np.linspace(0.0, v, n)
    integ = np.trapz((e < s).astype(float) - float(e < 0), s)
    return float(-v * (tau - float(e < 0)) + integ)


errs = [abs(lhs(e, v) - rhs_path(e, v))
        for e, v in zip(rng.laplace(0, 1, 2000), rng.normal(0, 1.5, 2000))]
print(f"Knight identity (pathwise, centred at the tau-quantile = 0):  "
      f"max |LHS - RHS| = {max(errs):.3e} over 2000 random (e, v)")

# second-order lower bound near the quantile
delta = float(np.quantile(rng.laplace(0, 1, 2_000_000), tau))
f0 = float(np.mean(np.abs(rng.laplace(0, 1, 4_000_000) - delta) < 0.05) / 0.1)
print(f"tau-quantile of Laplace(0,1) at tau={tau}: delta = {delta:+.5f}, "
      f"density there f_e(delta) ~= {f0:.4f}")
for v in (0.05, 0.1, 0.2):
    e = rng.laplace(0, 1, 2_000_000) - delta          # centred so that F_e(0) = tau
    print(f"   v = {v:.2f}:  E[rho_tau(e-v) - rho_tau(e)] = "
          f"{np.mean(rho(e - v) - rho(e)):.6f}   >= f_e(delta) v^2 / 2 = {f0 * v * v / 2:.6f}")

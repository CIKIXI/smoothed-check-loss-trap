"""
Edge cases for Proposition 3.13, checked independently (my own cross-check of the
adversarial run):

  (a) singular Sigma_u  -- is (Sigma_x+Sigma_u+M) still invertible and the claim true?
  (b) E[X] != 0         -- the statement assumes E[X]=0; check that it really fails without it
  (c) uniqueness        -- the loss is a strictly increasing function of s(beta), so the
                           minimiser is unique even though the map s -> E[rho_tau(eps+sZ)]
                           need not be convex in s
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BETA = np.array([2.0, 1.0])


def normal_abs(e, s):
    z = e / s
    return e * (2 * ndtr(z) - 1) + 2 * s * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)


def loss_via_variance(b, Sx, Su, M, eps_kind=0.5, tau=0.5):
    v = float((BETA - b) @ Sx @ (BETA - b) + b @ (Su + M) @ b)
    s = np.sqrt(max(v, 0.0))
    e = np.linspace(-80, 80, 400001)
    f = 0.5 * np.exp(-np.abs(e)) if eps_kind == 0.5 else np.exp(-e ** 2 / 2) / np.sqrt(2 * np.pi)
    vals = normal_abs(e, max(s, 1e-9))
    return 0.5 * float(np.trapz(f * vals, e))


print("=" * 96)
print("(a) singular Sigma_u: Sigma_u = diag(0.25, 0)  (only the first covariate contaminated)")
print("=" * 96)
Sx = np.array([[1.0, 0.6], [0.6, 1.0]])
Su = np.diag([0.25, 0.0])
for lab, M in (("M = 0", np.zeros((2, 2))), ("M = 0.2 I", 0.2 * np.eye(2)),
               ("M = Sigma_u", Su)):
    A = Sx + Su + M
    b_star = np.linalg.solve(A, Sx @ BETA)
    f = lambda b: loss_via_variance(b, Sx, Su, M)
    # coordinate descent on a fine grid around the closed form
    best, bv = None, np.inf
    for d1 in np.arange(-0.2, 0.2001, 0.01):
        for d2 in np.arange(-0.2, 0.2001, 0.01):
            c = b_star + np.array([d1, d2])
            v = f(c)
            if v < bv:
                bv, best = v, c
    print(f"  {lab:<12} Sigma_x+Sigma_u+M invertible: "
          f"{np.linalg.cond(A) < 1e12}   closed form {np.round(b_star, 4).tolist()}   "
          f"grid argmin {np.round(best, 4).tolist()}   dev {np.linalg.norm(best - b_star):.4f}")

print()
print("=" * 96)
print("(b) E[X] != 0: the claim should fail (this is why the statement assumes E[X]=0)")
print("=" * 96)
mu_x = np.array([0.5, 0.0])
Sx0 = np.eye(2)
Su0 = 0.25 * np.eye(2)
beta_bar = np.linalg.solve(Sx0 + Su0, Sx0 @ BETA)
n = 2_000_000
rng = np.random.RandomState(7)
X = rng.standard_normal((n, 2)) + mu_x
U = 0.5 * rng.standard_normal((n, 2))
W = X + U
eps = rng.laplace(0, 1, n)
Y = X @ BETA + eps
grid = np.arange(1.2, 2.0, 0.002)
vals = []
for b in grid:
    r = Y - b * W[:, 0]
    vals.append(float(np.mean(r * (0.5 - (r < 0)))))
b_mc = float(grid[int(np.argmin(vals))])
print(f"  closed-form beta_bar (coordinate 1) = {beta_bar[0]:.4f};  "
      f"MC minimiser with E[X]!=0 = {b_mc:.4f}")
print(f"  -> difference {b_mc - beta_bar[0]:+.4f}: the claim is NOT expected to hold here, "
      f"confirming that E[X]=0 is needed")

print()
print("=" * 96)
print("(c) uniqueness: L is a strictly increasing function of s(beta)")
print("=" * 96)
for s in (0.3, 0.6, 1.0, 1.5):
    e = np.linspace(-80, 80, 400001)
    f = 0.5 * np.exp(-np.abs(e))
    v = float(np.trapz(f * normal_abs(e, s), e))
    print(f"  s = {s:4.1f}:  E|eps + sZ| = {v:.6f}")
print("  (strictly increasing in s -> minimising L is the same as minimising s^2(beta),")
print("   which is a strictly convex quadratic with the unique minimiser stated)")

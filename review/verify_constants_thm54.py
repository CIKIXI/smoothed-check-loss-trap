"""
Adversarial check of the constant chain in the proof of Theorem 5.4 (estimated calibration).

The step under test is the restricted-eigenvalue transfer:
    A  := (1/n) sum_i (mu_i' D)^2      >= kappa ||D||_2^2        (Assumption 5.3)
    B  := (1/n) sum_i ((mu_hat_i-mu_i)' D)^2 <= r^2 Z_n^2 ||D||_1^2 <= 16 r^2 Z_n^2 s ||D||_2^2
and one wants  A_hat := (1/n) sum_i (mu_hat_i' D)^2 >= (kappa/2) ||D||_2^2.

With beta := 4 r Z_n sqrt(s) (the bound on |b_i| / ||D||_2 on the cone) the exact worst case
over admissible perturbations is

    inf A_hat / ||D||_2^2 = (sqrt(kappa) - beta)^2 ,

obtained by aligning every b_i against a_i.  The script (a) verifies that identity
numerically, and (b) shows which smallness assumption on r delivers the claimed kappa/2.
"""

import numpy as np

rng = np.random.RandomState(2026)
p, n = 4, 40_000


def worst_case_ratio(kappa, beta, reps=200):
    """Monte Carlo worst case of A_hat/||D||^2 with |b_i| <= beta*||D||_2."""
    vals = []
    for r in range(reps):
        a = rng.standard_normal(n) * np.sqrt(kappa)          # a_i = mu_i' D / ||D||_2
        b = -np.sign(a) * beta * (0.9 + 0.2 * rng.rand())    # adversarial alignment
        vals.append(float(np.mean((a + b) ** 2)))
    return min(vals)


print("=" * 100)
print("Worst-case transfer constant: (sqrt(kappa) - beta)^2 vs the paper's kappa/2")
print("=" * 100)
for kappa in (0.5, 1.0):
    for rzn in (0.05, 0.1, 0.2, 0.3):
        beta = 4 * rzn * np.sqrt(1)          # s = 1 for simplicity: beta = 4 r Z_n sqrt(s)
        claimed = kappa / 2
        worst = (np.sqrt(kappa) - beta) ** 2
        mc = worst_case_ratio(kappa, beta)
        print(f"  kappa={kappa:4.2f}  rZ_n=sqrt(s)={rzn:4.2f}  beta={beta:5.3f}   "
              f"claimed >= {claimed:.4f}   worst case (sqrt(k)-b)^2 = {worst:.4f}   "
              f"MC min = {mc:.4f}   {'OK' if worst >= claimed else 'CLAIM FAILS'}")

print()
print("Which assumption on r delivers kappa/2?")
for label, fac in (("16 r^2 Z_n^2 s <= kappa/2 (paper)", 1 / 32),
                   ("16 r^2 Z_n^2 s <= kappa/4", 1 / 64),
                   ("16 r^2 Z_n^2 s <= kappa/16", 1 / 256)):
    for kappa in (0.5, 1.0):
        # 16 r^2 Z_n^2 s <= c*kappa  =>  r Z_n sqrt(s) <= sqrt(c*kappa)/4
        c = fac * 32 if "kappa/2 " in label else (0.25 if "kappa/4" in label else 0.0625)
        rzn_max = np.sqrt(c * kappa) / 4
        beta = 4 * rzn_max * np.sqrt(1)
        worst = (np.sqrt(kappa) - beta) ** 2
        print(f"  {label:<36} kappa={kappa:4.2f}: r Z_n sqrt(s) <= {rzn_max:.4f}, "
              f"transfer >= {worst:.4f}  ({worst / kappa:.2f} kappa)  "
              f"{'>= kappa/2 OK' if worst >= kappa / 2 else 'below kappa/2'}")

print()
print("=" * 100)
print("The elementary inequality used in the fix: (a+b)^2 >= (1-t)a^2 - ((1-t)/t) b^2")
print("=" * 100)
for t in (0.25, 0.5):
    a = rng.standard_normal(2_000_000)
    b = rng.standard_normal(2_000_000) * 3
    lhs = (a + b) ** 2
    rhs = (1 - t) * a ** 2 - ((1 - t) / t) * b ** 2
    print(f"  theta={t}: min(lhs - rhs) = {np.min(lhs - rhs):+.6f}  "
          f"(violations: {int(np.sum(lhs < rhs - 1e-9))})")

print()
print("=" * 100)
print("Downstream algebra with the transferred constant kappa_hat >= kappa/2")
print("(u is reported in units of lambda sqrt(s)/kappa, as in the theorem's constants)")
print("=" * 100)
for kappa in (0.2, 0.5, 1.0):
    # kappa_hat u^2 <= 1.5 lambda sqrt(s) u + lambda^2 s, with kappa_hat = kappa/2
    kh = kappa / 2
    # u <= (1.5 + sqrt(2.25 + 4 kh)) lambda sqrt(s) / (2 kh);  divide by lambda sqrt(s)/kappa
    u_over = (1.5 + np.sqrt(1.5 ** 2 + 4 * kh)) * kappa / (2 * kh)
    C1 = u_over
    C2 = 4 * u_over
    C3 = 1.5 * u_over + 1
    print(f"  kappa={kappa:4.2f}: C_1 = {C1:5.2f} (paper 6) {'OK' if C1 <= 6 else 'FAILS'};  "
          f"C_2 = {C2:5.2f} (paper 24) {'OK' if C2 <= 24 else 'FAILS'};  "
          f"C_3 = {C3:5.2f}/kappa (paper 18/kappa) {'OK' if C3 <= 18 else 'FAILS'}")

r"""
Independent check of the exact population curves plotted in Figure 1.

simulations/routeA_figures.py now draws the population objectives by quadrature instead of
by simulation.  That makes four claims that must hold:

  (1) the quadrature  F(s) = E[rho_tau(eps + G)],  eps ~ Laplace(0,1), G ~ N(0, s^2),
      agrees with a direct Monte-Carlo average;
  (2) the scale functions used for the four curves are the true scales of the corresponding
      random variables, i.e. for each beta
          smoothed at W, pilot scale :  Y - W beta - sigma_hat Z
          true risk                  :  Y - X beta
          calibrated check loss      :  Y - mu_W beta
          posterior kernel, adaptive :  Y - mu_W beta - |beta| sqrt(Sigma_{x|w}) Z
      have the Gaussian part that the quadrature assumes (checked by matching the
      Monte-Carlo minimisers and values);
  (3) the minimisers of the quadrature curves are the population values predicted by
      Proposition 3.13:  beta_bar = 1.6 for the smoothed and the posterior-kernel curve,
      beta^* = 2 for the true risk and the calibrated check loss;
  (4) the Monte-Carlo versions of the same four objectives are minimised near the same
      points, with the deviations explained by simulation error.

Run:  python review/verify_fig1_exact_curves.py
"""

import io
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, roots_laguerre

SIGMA_U = 0.5
BETA_STAR = 2.0
TAU = 0.5
LAM = 1.0 / (1.0 + SIGMA_U ** 2)
SXW = LAM * SIGMA_U ** 2
BETA_BAR = LAM * BETA_STAR
SIGMA_HAT = abs(BETA_BAR) * SIGMA_U

NODES, WEIGHTS = roots_laguerre(120)


def phi(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)


def rho_tilde(r, s, tau=TAU):
    r = np.asarray(r, float)
    if s <= 0:
        return r * (tau - (r < 0))
    z = r / s
    return r * (ndtr(z) - (1 - tau)) + s * phi(z)


def loss_of_scale(s):
    s = max(float(s), 0.0)
    return 0.5 * float(np.sum(WEIGHTS * (rho_tilde(NODES, s) + rho_tilde(-NODES, s))))


def scales(b):
    """The four scale functions s(beta) used by the figure."""
    return {
        "smoothed": np.sqrt((BETA_STAR - b) ** 2 + b ** 2 * SIGMA_U ** 2 + SIGMA_HAT ** 2),
        "true_risk": abs(BETA_STAR - b),
        "calibrated": np.sqrt(LAM * (BETA_STAR - b) ** 2 + SXW * BETA_STAR ** 2),
        "posterior": np.sqrt(LAM * (BETA_STAR - b) ** 2 + SXW * BETA_STAR ** 2
                            + SXW * b ** 2),
    }


def check_quadrature(N=8_000_000, seed=20240):
    """Claim 1: quadrature against Monte-Carlo for several scales."""
    rng = np.random.RandomState(seed)
    eps = rng.laplace(0.0, 1.0, N)
    rows = []
    for s in (0.0, 0.4, 0.894, 1.2, 1.2649, 1.6, 2.0):
        z = rng.standard_normal(N)
        r = eps - s * z
        mc = float(np.mean(np.where(r < 0, (1 - TAU) * (-r), TAU * r)))
        q = loss_of_scale(s)
        rows.append(dict(s=s, quadrature=q, monte_carlo=mc, diff=q - mc,
                         mc_se=float(np.std(np.where(r < 0, (1 - TAU) * (-r), TAU * r))
                                     / np.sqrt(N))))
    return rows


def mc_objectives(beta, N, rng):
    """Claim 2: the four objectives evaluated by simulating the model directly."""
    X = rng.standard_normal(N)
    U = rng.standard_normal(N) * SIGMA_U
    W = X + U
    Y = X * BETA_STAR + rng.laplace(0.0, 1.0, N)
    muW = LAM * W
    Z = rng.standard_normal(N)
    out = {}
    out["smoothed"] = np.mean(rho_tilde(Y - W * beta, SIGMA_HAT))
    out["true_risk"] = np.mean(np.where(Y - X * beta < 0, (1 - TAU) * (X * beta - Y),
                                        TAU * (Y - X * beta)))
    out["calibrated"] = np.mean(np.where(Y - muW * beta < 0, (1 - TAU) * (muW * beta - Y),
                                         TAU * (Y - muW * beta)))
    out["posterior"] = np.mean(rho_tilde(Y - muW * beta, abs(beta) * np.sqrt(SXW)))
    return out


def check_scales(N=4_000_000, seed=777):
    """Claim 2: the quadrature curve must match the Monte-Carlo objective at every beta
    (a mismatch would mean the scale function is wrong); the comparison is done on the
    *excess* loss over the curve's own Monte-Carlo minimum, with a common random sample."""
    rng = np.random.RandomState(seed)
    grid = np.linspace(1.0, 2.6, 17)
    mc = {k: [] for k in ("smoothed", "true_risk", "calibrated", "posterior")}
    for b in grid:
        vals = mc_objectives(b, N, rng)
        for k, v in vals.items():
            mc[k].append(float(v))
    rows = []
    for k in mc:
        mcv = np.array(mc[k])
        mc_min = float(mcv.min())
        qv = np.array([loss_of_scale(scales(b)[k]) for b in grid])
        q_min = float(qv.min())
        dev = np.abs((mcv - mc_min) - (qv - q_min))
        rows.append(dict(curve=k, max_abs_dev=float(dev.max()),
                         grid_step=float(grid[1] - grid[0]),
                         mc_argmin=float(grid[int(np.argmin(mcv))]),
                         quad_argmin=float(grid[int(np.argmin(qv))])))
    return rows


def check_minimisers():
    """Claim 3: quadrature minimisers equal the population values."""
    rows = []
    for k, target in (("smoothed", BETA_BAR), ("true_risk", BETA_STAR),
                      ("calibrated", BETA_STAR), ("posterior", BETA_BAR)):
        r = minimize_scalar(lambda b: loss_of_scale(scales(b)[k]), bounds=(0.5, 3.5),
                            method="bounded", options={"xatol": 1e-13})
        rows.append(dict(curve=k, argmin=float(r.x),
                         value=float(r.fun), target=float(target),
                         dev=float(r.x - target)))
    return rows


def check_mc_minimisers(seeds=range(101, 113), N=200_000):
    """Claim 4: the Monte-Carlo sample objectives of the *previous* figure are minimised
    near the same points, so the values formerly quoted (1.596/2.001/1.993/1.597) were
    simulation error and not a contradiction."""
    grid = np.arange(0.8, 3.0001, 0.004)
    acc = {k: [] for k in ("smoothed", "true_risk", "calibrated", "posterior")}
    for sd in seeds:
        rng = np.random.RandomState(sd)
        X = rng.standard_normal(N)
        U = rng.standard_normal(N) * SIGMA_U
        W = X + U
        Y = X * BETA_STAR + rng.laplace(0.0, 1.0, N)
        muW = LAM * W
        Z = rng.standard_normal(N)
        curves = {k: [] for k in acc}
        for b in grid:
            curves["smoothed"].append(np.mean(rho_tilde(Y - W * b, SIGMA_HAT)))
            curves["true_risk"].append(np.mean(np.where(Y - X * b < 0,
                                                        (1 - TAU) * (X * b - Y),
                                                        TAU * (Y - X * b))))
            curves["calibrated"].append(np.mean(np.where(Y - muW * b < 0,
                                                         (1 - TAU) * (muW * b - Y),
                                                         TAU * (Y - muW * b))))
            curves["posterior"].append(np.mean(rho_tilde(Y - muW * b,
                                                         abs(b) * np.sqrt(SXW))))
        for k in acc:
            acc[k].append(float(grid[int(np.argmin(curves[k]))]))
    rows = []
    for k, v in acc.items():
        v = np.array(v)
        target = BETA_BAR if k in ("smoothed", "posterior") else BETA_STAR
        rows.append(dict(curve=k, mean=float(v.mean()), sd=float(v.std(ddof=1)),
                         target=float(target), mean_dev=float(v.mean() - target)))
    return rows


def main():
    res = {}
    print("=" * 78)
    print("(1) quadrature vs Monte-Carlo for F(s) = E[rho_tau(eps + N(0,s^2))]")
    res["quadrature"] = check_quadrature()
    for r in res["quadrature"]:
        ok = abs(r["diff"]) < 5 * r["mc_se"] + 1e-9
        print(f"    s = {r['s']:.4f}  quad {r['quadrature']:.8f}  "
              f"mc {r['monte_carlo']:.8f}  diff {r['diff']:+.2e}  "
              f"(5 se = {5*r['mc_se']:.1e})  {'ok' if ok else 'MISMATCH'}")

    print("=" * 78)
    print("(2) scale functions: quadrature curve vs direct simulation of the model")
    res["scales"] = check_scales()
    for r in res["scales"]:
        print(f"    {r['curve']:11s} max |excess difference| = {r['max_abs_dev']:.2e}  "
              f"argmin mc {r['mc_argmin']:.2f} vs quad {r['quad_argmin']:.2f}")

    print("=" * 78)
    print("(3) quadrature minimisers vs the population values of Proposition 3.13")
    res["minimisers"] = check_minimisers()
    for r in res["minimisers"]:
        print(f"    {r['curve']:11s} argmin {r['argmin']:.9f}  target {r['target']:.1f}  "
              f"deviation {r['dev']:+.2e}")

    print("=" * 78)
    print("(4) Monte-Carlo minimisers at N = 2e5 over 12 seeds (claim of Monte-Carlo error)")
    res["mc_minimisers"] = check_mc_minimisers()
    for r in res["mc_minimisers"]:
        print(f"    {r['curve']:11s} mean {r['mean']:.4f}  sd {r['sd']:.4f}  "
              f"target {r['target']:.1f}  mean deviation {r['mean_dev']:+.4f}")

    with io.open("review/verify_fig1_exact_curves.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("=" * 78)
    print("[saved] review/verify_fig1_exact_curves.json")


if __name__ == "__main__":
    main()

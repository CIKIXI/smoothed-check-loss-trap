#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adversarial_check.py
====================
Adversarial numerical verification of the three-part claim in
``review/adversarial/statement_prop313.md`` (population minimisers of
quantile-regression-type criteria under a Gaussian design with additive
measurement error).

Model (exactly as stated; nothing else is assumed):

    X ~ N(mu_x, Sx),  Sx > 0,  mu_x = 0
    U ~ N(0, Su),     Su >= 0,  independent of X
    W = X + U
    Y = X'beta* + eps,   eps independent of (X,U), E|eps| < inf

Criteria (tau in (0,1), rho_tau(r) = r(tau - 1{r<0})):

    (i)   R(b)      = E[ rho_tau(Y - b'W) ]
    (ii)  R_s(b)    = E[ rho_tau(Y - b'W - sigma Z) ]        (Z ~ N(0,1) indep)
    (iii) R_m(b)    = E[ rho_tau(Y - b'W - sigma_M(b) Z) ],  sigma_M(b)=sqrt(b'Mb)

Claimed minimisers:  bbar = (Sx+Su)^-1 Sx beta*   (i),(ii)
                     bdag = (Sx+Su+M)^-1 Sx beta* (iii)

THREE NUMERICALLY INDEPENDENT ROUTES
------------------------------------
ROUTE 1 ("reduction route").  Uses only the elementary fact that a linear
functional of a jointly Gaussian vector is Gaussian (a property of the model,
not of the claim):

    F(b) = Xi( sqrt( q(b) ) ),   q(b) = (b-b*)'Sx(b-b*) + b'Su b + b'Mb + sigma^2
    Xi(s) = E[ (eps + s Z)^- ],  Z ~ N(0,1)

Xi is tabulated per error distribution by *adaptive quadrature against the
exact density* (or exactly, for purely atomic eps), so it is accurate to
~1e-12 for every distribution used here, including multimodal and heavy-tailed
ones.  Note rho_tau(r) = tau*r + r^- and E[Y - b'W] = 0 when E[X] = 0, so the
whole tau-dependence drops out of the criterion; tau is nevertheless kept
explicitly everywhere so that the claim can be tested as stated.

ROUTE 2 ("direct route").  Tensor Gauss-Hermite quadrature over the *model*
variables (X,U), so the reduction above is never used:

    F(b) = sum_n w_n * J( t_n(b), sigma(b) ),   t_n(b) = x_n'(b*-b) - u_n'b
    J(t,s) = E[ (eps + t - s Z)^- ]  = E_eps[ s*phi((eps+t)/s) - (eps+t)*Phi(-(eps+t)/s) ]
           = int_{-inf}^{-t} F_eps(u) du                            (s = 0)
    J is tabulated in t by adaptive quadrature (exactly, for atomic eps).

ROUTE 3 ("Monte-Carlo / empirical route").  Raw simulation (X,U,eps) -> (Y,W).
    (i)    the empirical criterion is convex piecewise linear; its exact
           minimiser is a linear program (quantile regression)
    (ii)/(iii) the smoothed empirical criterion is smooth & convex; minimised
           with L-BFGS using analytic gradients.
    Additionally the *population subgradient at the claimed minimiser* is
    tested directly by Monte Carlo (must contain 0).

Plus diagnostics: uniqueness scans, numeric Hessians, high-precision mpmath
checks at huge bandwidths, and stress tests of the assumptions
(mu_x != 0, singular Sx, eps without a density, E|eps| = inf).

Run:  python adversarial_check.py            (full run)
      python adversarial_check.py --quick    (tiny subsets)
Outputs: results/*.json, results/*.csv, results/run_log.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings

import numpy as np
from scipy import integrate, optimize, stats
from scipy.interpolate import CubicSpline
from scipy.special import erf, ndtr

warnings.filterwarnings("ignore")
np.set_printoptions(precision=6, linewidth=140)

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

SQRT2 = np.sqrt(2.0)
PHI0 = 1.0 / np.sqrt(2.0 * np.pi)          # phi(0)
INF = np.inf


def Phi(x):
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / SQRT2))


def phi(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / np.sqrt(2.0 * np.pi)


def rho(tau, r):
    """check loss rho_tau(r) = r (tau - 1{r<0})."""
    r = np.asarray(r, dtype=float)
    return np.where(r >= 0.0, tau * r, (tau - 1.0) * r)


def g_neg(m, s):
    """E[max(-W,0)] for W ~ N(m,s^2)  (= E[(-W)^+], non-negative).

    Closed form:  s*phi(m/s) - m*Phi(-m/s).   For s = 0 this is max(-m,0).
    Used as  rho_tau(r) = tau*r + E[(-r)^+]  and  E_Z[rho_tau(r-sZ)]
    = tau*r + E[max(-(r-sZ),0)] = tau*r + g_neg(r,s).
    """
    m = np.asarray(m, dtype=float)
    if np.isscalar(s):
        if s == 0.0:
            return np.maximum(-m, 0.0)
        return s * phi(m / s) - m * Phi(-m / s)
    s = np.asarray(s, dtype=float)
    ss = np.where(s > 0.0, s, 1.0)
    out = ss * phi(m / ss) - m * Phi(-m / ss)
    return np.where(s > 0.0, out, np.maximum(-m, 0.0))


def rho_smooth(tau, r, s):
    """rho~_tau(r,s) = E_Z[rho_tau(r - sZ)] = tau*r + E[max(-(r-sZ),0)]."""
    r = np.asarray(r, dtype=float)
    return tau * r + g_neg(r, s)


# ---------------------------------------------------------------------------
# 1.  error distributions
# ---------------------------------------------------------------------------
class Eps:
    """eps: cdf (always), pdf (if it has a density), rvs, optional atoms."""

    def __init__(self, name, cdf, rvs, pdf=None, ppf=None, kinks=(),
                 atoms=None, mean=None, sd=None, exact_xi=None, note=""):
        self.name = name
        self._cdf = cdf
        self._rvs = rvs
        self._pdf = pdf
        self._ppf = ppf
        self.kinks = tuple(float(k) for k in kinks)
        self.atoms = atoms                 # (nodes, probs) or None
        self._mean = mean
        self._sd = sd
        self.exact_xi = exact_xi
        self.note = note

    def cdf(self, x):
        return np.asarray(self._cdf(np.asarray(x, dtype=float)), dtype=float)

    def pdf(self, x):
        if self._pdf is None:
            raise RuntimeError("no density for " + self.name)
        return np.asarray(self._pdf(np.asarray(x, dtype=float)), dtype=float)

    def has_density(self):
        return self._pdf is not None

    def rvs(self, n, rng):
        return np.asarray(self._rvs(n, rng), dtype=float).ravel()

    def mean(self):
        if self._mean is None:
            if self.atoms is not None:
                x, p = self.atoms
                self._mean = float(p @ x)
            else:
                self._mean = float(integrate.quad(
                    lambda x: x * self.pdf(x), -INF, INF, points=self.kinks,
                    limit=400)[0])
        return self._mean

    def var(self):
        if self._sd is None:
            m = self.mean()
            if self.atoms is not None:
                x, p = self.atoms
                m2 = float(p @ x ** 2)
            else:
                m2 = float(integrate.quad(
                    lambda x: x ** 2 * self.pdf(x), -INF, INF, points=self.kinks,
                    limit=400)[0])
            self._sd = float(np.sqrt(max(m2 - m ** 2, 0.0)))
        return self._sd ** 2


def make_eps_specs():
    out = {}

    # 1. standard Gaussian (mean 0, var 1) -- analytic Xi(s) = s*phi(0)
    out["gauss"] = Eps(
        "gauss", stats.norm(0, 1).cdf, lambda n, r: r.normal(size=n),
        pdf=stats.norm(0, 1).pdf, ppf=stats.norm(0, 1).ppf, mean=0.0, sd=1.0,
        exact_xi=lambda s: np.sqrt(1.0 + np.asarray(s, float) ** 2) * PHI0,
        note="reference; eps~N(0,1) so eps+sZ ~ N(0,1+s^2)")

    # 2. Laplace, scaled to variance 1
    lap = stats.laplace(0.0, 1.0 / np.sqrt(2.0))
    out["laplace"] = Eps("laplace", lap.cdf,
                         lambda n, r: lap.rvs(size=n, random_state=r),
                         pdf=lap.pdf, ppf=lap.ppf, kinks=(0.0,), mean=0.0, sd=1.0)

    # 3. asymmetric Laplace (kappa = 3), standardised
    m = stats.laplace_asymmetric(3.0).mean()
    s = stats.laplace_asymmetric(3.0).std()
    al = stats.laplace_asymmetric(3.0, loc=-m / s, scale=1.0 / s)
    out["asym_laplace"] = Eps("asym_laplace", al.cdf,
                              lambda n, r: al.rvs(size=n, random_state=r),
                              pdf=al.pdf, ppf=al.ppf, kinks=(-m / s,),
                              mean=0.0, sd=1.0)

    # 4. uniform on the variance-1 interval
    un = stats.uniform(-np.sqrt(3.0), 2 * np.sqrt(3.0))
    out["uniform"] = Eps("uniform", un.cdf,
                         lambda n, r: un.rvs(size=n, random_state=r),
                         pdf=un.pdf, ppf=un.ppf,
                         kinks=(-np.sqrt(3.0), np.sqrt(3.0)), mean=0.0, sd=1.0)

    # 5. centred exponential: Exp(1)-1  (mean 0, var 1, support [-1,inf))
    ex = stats.expon(scale=1.0, loc=-1.0)
    out["exp_centred"] = Eps("exp_centred", ex.cdf,
                             lambda n, r: ex.rvs(size=n, random_state=r),
                             pdf=ex.pdf, ppf=ex.ppf, kinks=(-1.0,),
                             mean=0.0, sd=1.0)

    # 6. shifted exponential: 3+Exp(1)  (no mass below 3; density jump at 3)
    ex2 = stats.expon(scale=1.0, loc=3.0)
    out["exp_shift3"] = Eps("exp_shift3", ex2.cdf,
                            lambda n, r: ex2.rvs(size=n, random_state=r),
                            pdf=ex2.pdf, ppf=ex2.ppf, kinks=(3.0,),
                            mean=4.0, sd=1.0)

    # 7. Student t, 3 df, standardised (heavy tail, E|eps| < inf)
    t3 = stats.t(df=3.0, loc=0.0, scale=1.0 / np.sqrt(3.0))
    out["t3"] = Eps("t3", t3.cdf, lambda n, r: t3.rvs(size=n, random_state=r),
                    pdf=t3.pdf, ppf=t3.ppf, mean=0.0, sd=1.0)

    # 8. skew-normal alpha = 8, standardised (own skewed example #1)
    m = stats.skewnorm(8.0).mean()
    s = stats.skewnorm(8.0).std()
    sn = stats.skewnorm(8.0, loc=-m / s, scale=1.0 / s)
    out["skewnorm8"] = Eps("skewnorm8", sn.cdf,
                           lambda n, r: sn.rvs(size=n, random_state=r),
                           pdf=sn.pdf, ppf=sn.ppf, mean=0.0, sd=1.0)

    # 9. split (two-piece) normal s-=1, s+=4, standardised (own example #2)
    s1, s2 = 1.0, 4.0
    cc = np.sqrt(2.0 / np.pi) / (s1 + s2)

    def sp_pdf(x):
        x = np.asarray(x, float)
        return cc * np.exp(-x ** 2 / (2 * np.where(x < 0, s1, s2) ** 2))

    def sp_cdf(x):
        x = np.asarray(x, float)
        return np.where(x < 0, 2 * s1 / (s1 + s2) * Phi(x / s1),
                        s1 / (s1 + s2) + 2 * s2 / (s1 + s2) * (Phi(x / s2) - 0.5))

    def sp_rvs(n, rng):
        z = np.abs(rng.normal(size=n))
        side = rng.random(n) < s1 / (s1 + s2)
        return np.where(side, -s1 * z, s2 * z)

    def sp_ppf(u):
        u = np.asarray(u, float)
        frac = s1 / (s1 + s2)
        return np.where(u < frac, s1 * stats.norm.ppf(u * (s1 + s2) / (2 * s1)),
                        s2 * stats.norm.ppf(0.5 + (u - frac) * (s1 + s2) / (2 * s2)))

    spm = float(integrate.quad(lambda x: x * sp_pdf(x), -INF, INF)[0])
    spv = float(integrate.quad(lambda x: x ** 2 * sp_pdf(x), -INF, INF)[0])
    sps = float(np.sqrt(spv - spm ** 2))
    out["splitnorm"] = Eps(
        "splitnorm", lambda x: sp_cdf(spm + sps * np.asarray(x, float)),
        lambda n, r: (sp_rvs(n, r) - spm) / sps,
        pdf=lambda x: sp_pdf(spm + sps * np.asarray(x, float)) * sps,
        ppf=lambda u: (sp_ppf(np.asarray(u, float)) - spm) / sps,
        kinks=(-spm / sps,), mean=0.0, sd=1.0)

    # 10. bimodal 0.5 N(-2.5,.5^2) + 0.5 N(2.5,.5^2): no mass near 0
    mix = [stats.norm(-2.5, 0.5), stats.norm(2.5, 0.5)]

    def bm_cdf(x):
        x = np.asarray(x, float)
        return 0.5 * mix[0].cdf(x) + 0.5 * mix[1].cdf(x)

    out["bimodal"] = Eps(
        "bimodal", bm_cdf,
        lambda n, r: np.where(r.random(n) < 0.5,
                              mix[0].rvs(size=n, random_state=r),
                              mix[1].rvs(size=n, random_state=r)),
        pdf=lambda x: 0.5 * mix[0].pdf(x) + 0.5 * mix[1].pdf(x),
        mean=0.0, sd=np.sqrt(6.5))

    # 11. purely atomic (NO density): 0.4 d_{-1} + 0.6 d_{+2}
    atoms = np.array([-1.0, 2.0])
    aprob = np.array([0.4, 0.6])

    def at_cdf(x):
        x = np.asarray(x, float)
        return aprob[0] * (x >= atoms[0]) + aprob[1] * (x >= atoms[1])

    out["atomic"] = Eps("atomic", at_cdf,
                        lambda n, r: np.where(r.random(n) < aprob[0],
                                              atoms[0], atoms[1]),
                        atoms=(atoms, aprob), kinks=(),
                        mean=float(aprob @ atoms),
                        sd=float(np.sqrt(aprob @ atoms ** 2 - (aprob @ atoms) ** 2)),
                        note="no density")

    # 12. degenerate eps == 10 (no density, extreme shift)
    out["const10"] = Eps("const10",
                         lambda x: (np.asarray(x, float) >= 10.0) * 1.0,
                         lambda n, r: np.full(n, 10.0),
                         atoms=(np.array([10.0]), np.array([1.0])),
                         mean=10.0, sd=0.0, note="no density")
    return out


EPS = make_eps_specs()


# ---------------------------------------------------------------------------
# 2.  Xi(s) = E[(eps + s Z)^-]  (route 1) and  J(t,s) (route 2)
# ---------------------------------------------------------------------------
class XiFun:
    """Xi(s) = E[max(-(eps+sZ),0)], s >= 0.

    s <  s_lo : adaptive quadrature against the density (exact for every eps)
    s >= s_lo : the fixed density-based rule (accurate to ~1e-11 there)

    Tabulated on a graded grid and interpolated with a cubic spline; the
    achieved accuracy is verified against fresh quadrature in the driver.
    """

    def __init__(self, eps, s_lo=0.12, s_max=200.0, s_tail=1e6):
        self.eps = eps
        if eps.exact_xi is not None:
            s = np.concatenate([np.linspace(0.0, s_max, 30001),
                                np.geomspace(s_max, s_tail, 600)[1:]])
            vals = np.asarray(eps.exact_xi(s), float)
        else:
            s_small = np.concatenate([[0.0], np.geomspace(1e-4, s_lo, 38)])
            s_big = np.concatenate([
                np.linspace(s_lo, s_max, 20001)[1:],
                np.geomspace(s_max, s_tail, 600)[1:]])
            s = np.concatenate([s_small, s_big])
            vals = np.empty_like(s)
            for i, sv in enumerate(s_small):
                vals[i] = xi_quad(eps, float(sv))
            xr = xrule(eps)
            vals[len(s_small):] = xr.weights @ g_neg(xr.nodes[:, None],
                                                     s_big[None, :])
        self.s = s
        self.vals = vals
        # Xi'(0) = 0 exactly (Xi'(s) = s E[f_eps(-sZ)]); clamp the left end so
        # that the spline does not have to invent the curvature at s = 0.
        self.spl = CubicSpline(s, vals, bc_type=((1, 0.0), (1, PHI0)))
        self.tail_s0 = float(s_tail)
        self.tail_v0 = float(vals[-1])
        self.tail_slope = PHI0

    def __call__(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        return np.where(s <= self.tail_s0,
                        self.spl(np.minimum(s, self.tail_s0)),
                        self.tail_v0 + self.tail_slope * (s - self.tail_s0))

    def deriv(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        return np.where(s <= self.tail_s0,
                        self.spl.derivative()(np.minimum(s, self.tail_s0)),
                        self.tail_slope)


def quad_split(f, a, b, breaks=(), tol=1e-14):
    """int_a^b f by adaptive quadrature, split at all interior break points."""
    br = sorted(set(float(x) for x in breaks if a < x < b))
    edges = [a] + br + [b]
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        tot += integrate.quad(f, lo, hi, limit=400, epsabs=tol, epsrel=1e-13)[0]
    return float(tot)


def quad_full(f, eps, breaks=()):
    """int_R f over the law of eps: bulk panel + reciprocal-map tails.

    The tails are handled by x = A - y/(1-y) (left) and x = B + y/(1-y)
    (right), which makes the transformed integrand vanish at y -> 1 for any
    polynomially decaying density, so heavy tails cost nothing.
    """
    a, b = eps_range(eps)
    tot = quad_split(f, a, b, breaks)
    tot += integrate.quad(lambda y: f(a - y / (1 - y)) / (1 - y) ** 2, 0, 1,
                          limit=400, epsabs=1e-15, epsrel=1e-13)[0]
    tot += integrate.quad(lambda y: f(b + y / (1 - y)) / (1 - y) ** 2, 0, 1,
                          limit=400, epsabs=1e-15, epsrel=1e-13)[0]
    return float(tot)


_EPS_RANGE = {}


def eps_range(eps, q=1e-9):
    """Finite interval holding all but ~q of the mass (plus all density jumps).

    Cached per (distribution, q): different rules use different bulk ranges.
    """
    key = (eps.name, q)
    if key in _EPS_RANGE:
        return _EPS_RANGE[key]
    if eps.atoms is not None:
        x, _ = eps.atoms
        lo, hi = float(x.min()), float(x.max())
    else:
        if eps._ppf is not None:
            lo = float(eps._ppf(q)); hi = float(eps._ppf(1 - q))
        else:
            lo = float(optimize.brentq(lambda z: eps.cdf(z) - q, -1e7, 1e7,
                                       xtol=1e-12))
            hi = float(optimize.brentq(lambda z: eps.cdf(z) - (1 - q), -1e7, 1e7,
                                       xtol=1e-12))
        if eps.kinks:
            lo = min(lo, min(eps.kinks)); hi = max(hi, max(eps.kinks))
    _EPS_RANGE[key] = (lo, hi)
    return lo, hi


def xi_quad(eps, s):
    """Xi(s) = E[max(-(eps + sZ), 0)]  (non-negative, increasing in s).

    For small s the integrand g_neg(x,s) f(x) has a feature of width ~s at
    x = 0; a geometric ladder of break points around 0 (and at every density
    kink) makes the adaptive quadrature resolve it.
    """
    if eps.atoms is not None:
        x, p = eps.atoms
        return float(p @ g_neg(x, s))
    br = list(eps.kinks)
    if s < 0.5:
        a, b = eps_range(eps)
        ladder = [0.0]
        for k in range(0, 40):
            d = s * 2.0 ** k
            if a < -d < b:
                ladder.append(-d)
            if a < d < b:
                ladder.append(d)
            if d > max(abs(a), abs(b)):
                break
        br += ladder
    return quad_full(lambda x: g_neg(x, s) * eps.pdf(x), eps, br)


def J_quad(eps, sigma, t):
    """J(t,sigma) = E[max(-(eps + t - sigma Z), 0)]."""
    if eps.atoms is not None:
        x, p = eps.atoms
        if sigma == 0.0:
            return float(p @ np.maximum(-(x + t), 0.0))
        return float(p @ g_neg(x + t, sigma))
    if sigma == 0.0:
        # J(t,0) = E[max(-(eps+t),0)] = int_{-inf}^{-t} F_eps(u) du
        a, b = eps_range(eps)
        if -t <= a:
            return 0.0                       # F <= q on the whole range
        if -t <= b:
            return quad_split(lambda u: eps.cdf(u), a, -t)
        base = quad_split(lambda u: eps.cdf(u), a, b)
        return float(base + (-t - b))        # F ~ 1 beyond b
    return quad_full(lambda x: g_neg(x + t, sigma) * eps.pdf(x), eps, eps.kinks)


class XRule:
    """Fixed quadrature rule (nodes, weights) for E_eps[h(eps)].

    Composite Gauss-Legendre against the density, with reciprocal-map panels
    for the two tails (so heavy tails carry no truncation error) and panel
    edges placed at every density kink.  The number of panels is doubled until
    the total mass is reproduced to `target`, so the rule adapts to how fast
    the density varies (e.g. t_3 has poles close to the real axis and needs
    many more panels than a Gaussian).  Accurate for *smooth* h, which is what
    route 2 needs (sigma > 0); for atomic eps the atoms are used exactly.
    """

    def __init__(self, eps, order=16, n_tail=40, target=1e-13, n_pan_max=4000,
                 q_bulk=1e-8):
        self.eps = eps
        self.order = order
        if eps.atoms is not None:
            self.nodes, self.weights = eps.atoms[0].copy(), eps.atoms[1].copy()
            self.mass_err = abs(self.weights.sum() - 1.0)
            self.n_pan = 0
            return
        a, b = eps_range(eps, q=q_bulk)
        gl_x, gl_w = np.polynomial.legendre.leggauss(order)
        yedges = np.linspace(0.0, 1.0, n_tail + 1)
        n_pan = 60
        while True:
            edges = np.array(sorted(
                set(np.linspace(a, b, n_pan + 1).tolist()) |
                {k for k in eps.kinks if a < k < b}))
            nodes, weights = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
                x = mid + half * gl_x
                nodes.append(x)
                weights.append(gl_w * half * eps.pdf(x))
            for lo, hi in zip(yedges[:-1], yedges[1:]):
                mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
                y = mid + half * gl_x
                for side in (-1, 1):
                    x = (a - y / (1 - y)) if side < 0 else (b + y / (1 - y))
                    nodes.append(x)
                    weights.append(gl_w * half * eps.pdf(x) / (1 - y) ** 2)
            nodes = np.concatenate(nodes)
            weights = np.concatenate(weights)
            mass_err = abs(weights.sum() - 1.0)
            if mass_err < target or n_pan >= n_pan_max or nodes.size > 40000:
                break
            n_pan *= 2
        self.nodes, self.weights = nodes, weights
        self.mass_err = mass_err
        self.n_pan = n_pan

    def __call__(self, h):
        return float(np.dot(self.weights, h(self.nodes)))


_XRULE_CACHE = {}


def xrule(eps):
    if eps.name not in _XRULE_CACHE:
        _XRULE_CACHE[eps.name] = XRule(eps)
    return _XRULE_CACHE[eps.name]


def xrule_light(eps):
    """Smaller rule used when the integrand is smooth (sigma > 0, route 2 (iii))."""
    key = eps.name + "|light"
    if key not in _XRULE_CACHE:
        _XRULE_CACHE[key] = XRule(eps, order=10, n_tail=12, target=1e-10,
                                  n_pan_max=60, q_bulk=1e-4)
    return _XRULE_CACHE[key]


_GTAB_CACHE = {}


def gtable(eps, T=200.0, h=0.005):
    """Spline of G(t) = E[max(-(eps+t),0)] = J(t,0) = int_{-inf}^{-t} F(u) du.

    Exact identity used:  G(0) = E[max(-eps,0)]  and
    G(t) = G(0) - int_0^t F_eps(-s) ds,  with the exact cdf under the
    integral.  The antiderivative of a cubic-spline interpolant of F_eps(-s)
    is integrated exactly, so the tabulation error is O(h^4) ~ 1e-11.
    """
    if eps.name in _GTAB_CACHE:
        return _GTAB_CACHE[eps.name]
    if eps.atoms is not None:
        x, p = eps.atoms

        def G(t):
            t = np.asarray(t, dtype=float)
            return np.maximum(-(x[None, :] + t[:, None]), 0.0) @ p
        _GTAB_CACHE[eps.name] = G
        return G
    G0 = xi_quad(eps, 0.0)                     # E[max(-eps,0)], exact
    n = int(2 * T / h) + 1
    grid = np.linspace(-T, T, n)               # symmetric, 0 is a node
    Fv = eps.cdf(-grid)
    anti = CubicSpline(grid, Fv, bc_type="not-a-knot").antiderivative()
    a0 = float(anti(0.0))

    def G(t):
        t = np.asarray(t, dtype=float)
        tc = np.clip(t, -T, T)
        v = G0 - (anti(tc) - a0)
        v = np.where(t > T, 0.0, v)                       # G -> 0
        v = np.where(t < -T, v + (-T - t), v)             # F(-s) ~ 1
        return v

    _GTAB_CACHE[eps.name] = G
    return G


# ---------------------------------------------------------------------------
# 3.  configurations
# ---------------------------------------------------------------------------
def sym(A):
    return 0.5 * (A + A.T)


def posdef_from(rng, p, cond):
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)
    return Q @ np.diag(np.geomspace(1.0, 1.0 / cond, p)) @ Q.T


def commutator(A, B):
    return float(np.linalg.norm(A @ B - B @ A))


def describe_mat(A, tol=1e-12):
    A = np.asarray(A, float)
    if np.allclose(A, 0.0, atol=tol):
        return "0"
    p = A.shape[0]
    d = np.diag(A)
    if np.allclose(A - np.diag(d), 0.0, atol=tol) and np.allclose(d, d[0]):
        return "%.3g*I" % d[0]
    ev = np.linalg.eigvalsh(sym(A))
    rank = int(np.sum(ev > max(1e-10, 1e-10 * ev.max())))
    return "sym(ev=[%s],rk=%d)" % (",".join("%.2f" % v for v in ev), rank)


def make_cfg(name, p, tau, eps, Sx, Su, M, sigma, bs, mu=None, block=""):
    return dict(name=name, p=p, tau=float(tau), eps=eps,
                Sx=sym(np.atleast_2d(np.asarray(Sx, float))),
                Su=sym(np.atleast_2d(np.asarray(Su, float))),
                M=sym(np.atleast_2d(np.asarray(M, float))),
                sigma=float(sigma), bs=np.asarray(bs, float).ravel(),
                mu=np.zeros(p) if mu is None else np.asarray(mu, float).ravel(),
                block=block)


def claimed_beta(cfg):
    """(Sx+Su+M)^-1 Sx beta*, via lstsq when the matrix is singular."""
    G = cfg["Sx"] + cfg["Su"] + cfg["M"]
    rhs = cfg["Sx"] @ cfg["bs"]
    try:
        return np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, rhs, rcond=None)[0]


def q_inner(cfg, beta):
    b = np.asarray(beta, float).ravel()
    a = b - cfg["bs"]
    return float(a @ cfg["Sx"] @ a + b @ cfg["Su"] @ b + b @ cfg["M"] @ b)


# --------------------------- ROUTE 1 ---------------------------------------
_XI_CACHE = {}


def xi_fun(eps):
    if eps.name not in _XI_CACHE:
        _XI_CACHE[eps.name] = XiFun(eps)
    return _XI_CACHE[eps.name]


def tau_shift(cfg, b):
    """tau * E[Y - b'W] = tau * mu'(beta* - b)  (identically 0 when E[X] = 0)."""
    return cfg["tau"] * float(cfg["mu"] @ (cfg["bs"] - np.asarray(b, float).ravel()))


def route1_obj(cfg, xi, sigma=None):
    """F(b) = tau*E[Y-b'W] + Xi(sqrt(q(b))),  q(b) = Var(residual) + sigma^2."""
    sig2 = (cfg["sigma"] ** 2) if sigma is None else sigma ** 2

    def F(b):
        b = np.asarray(b, float).ravel()
        s = np.sqrt(max(q_inner(cfg, b) + sig2, 1e-300))
        return float(tau_shift(cfg, b) + xi(np.array([s]))[0])

    return F


def route1_min(cfg, xi, n_restart=3, rng=None):
    rng = rng or np.random.default_rng(0)
    p = cfg["p"]
    sig2 = cfg["sigma"] ** 2
    G = cfg["Sx"] + cfg["Su"] + cfg["M"]
    bs = cfg["bs"]

    def F(b):
        b = np.asarray(b, float).ravel()
        s = np.sqrt(max(q_inner(cfg, b) + sig2, 1e-300))
        return float(tau_shift(cfg, b) + xi(np.array([s]))[0])

    def grad(b):
        g = -cfg["tau"] * cfg["mu"]
        s = np.sqrt(max(q_inner(cfg, b) + sig2, 1e-300))
        if s < 1e-12:
            return g
        dq = 2 * (G @ b - cfg["Sx"] @ bs)
        return g + float(xi.deriv(np.array([s]))[0]) * dq / (2 * s)

    starts = [claimed_beta(cfg), np.zeros(p)]
    scale = max(1.0, float(np.linalg.norm(bs)))
    for _ in range(n_restart):
        starts.append(claimed_beta(cfg) + rng.normal(scale=1.5 * scale, size=p))
    best, bestv = None, INF
    for s0 in starts:
        try:
            r = optimize.minimize(F, s0, jac=grad, method="BFGS",
                                  options=dict(gtol=1e-12, maxiter=2000))
        except Exception:
            continue
        if r.fun < bestv:
            best, bestv = np.asarray(r.x, float).ravel(), float(r.fun)
    return best, bestv, F


def numeric_hessian(F, x, h=1e-4):
    x = np.asarray(x, float).ravel()
    p = x.size
    H = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p); ei[i] = h
            ej = np.zeros(p); ej[j] = h
            H[i, j] = H[j, i] = (F(x + ei + ej) - F(x + ei - ej)
                                 - F(x - ei + ej) + F(x - ei - ej)) / (4 * h * h)
    return H


# --------------------------- ROUTE 2 ---------------------------------------
_GH_CACHE = {}


def gh_nodes(n):
    if n not in _GH_CACHE:
        x, w = np.polynomial.hermite_e.hermegauss(n)
        _GH_CACHE[n] = (x, w / np.sqrt(2 * np.pi))
    return _GH_CACHE[n]


def psd_sqrt(A):
    """Symmetric PSD square root (works for singular A, unlike Cholesky)."""
    w, V = np.linalg.eigh(sym(A))
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def model_nodes(Sx, Su, mu, n_gh):
    """Tensor Gauss-Hermite nodes/weights for the joint law of (X, U)."""
    p = Sx.shape[0]
    z, w = gh_nodes(n_gh)
    Lx = psd_sqrt(Sx)
    Lu = psd_sqrt(sym(Su))
    grids = np.meshgrid(*([z] * (2 * p)), indexing="ij")
    Z = np.stack([g.ravel() for g in grids], axis=1)
    idx = [np.searchsorted(z, Z[:, k]) for k in range(2 * p)]
    Wg = np.ones(Z.shape[0])
    for k in range(2 * p):
        Wg *= w[idx[k]]
    Xn = Z[:, :p] @ Lx.T + np.asarray(mu, float)
    Un = Z[:, p:] @ Lu.T
    return Xn, Un, Wg


_J_CACHE = {}


def jtab(eps, sigma, T=60.0, n=4001, t_tail=1e5, n_tail=200):
    """Spline of t -> J(t,sigma) for sigma > 0, built from the density rule."""
    key = (eps.name, round(float(sigma), 12))
    if key in _J_CACHE:
        return _J_CACHE[key]
    xr = xrule(eps)
    t = np.concatenate([-np.geomspace(T, t_tail, n_tail)[::-1],
                        np.linspace(-T, T, n),
                        np.geomspace(T, t_tail, n_tail)])
    t = np.unique(t)
    vals = np.empty_like(t)
    chunk = 128                      # keep the (n_rule x chunk) block small
    for i in range(0, len(t), chunk):
        sl = slice(i, min(i + chunk, len(t)))
        vals[sl] = xr.weights @ g_neg(xr.nodes[:, None] + t[None, sl], sigma)
    spl = CubicSpline(t, vals, bc_type="natural", extrapolate=False)

    def J(tq):
        tq = np.asarray(tq, dtype=float)
        v = spl(np.clip(tq, -t_tail, t_tail))
        v = np.where(tq < -t_tail, -(tq + eps.mean()), v)   # J ~ -t - E eps
        return np.where(tq > t_tail, 0.0, v)

    _J_CACHE[key] = J
    return J


def route2_obj(cfg, n_gh=24, nodes=None, xr=None):
    """Direct model quadrature.  No use of the reduction.

    F(b) = tau*mu'(b*-b) + sum_n w_n * J( t_n(b), sigma(b) )
    with t_n(b) = x_n'(b*-b) - u_n'b and J(t,s) = E_eps[max(-(eps+t-sZ),0)].

    * M = 0 (parts (i),(ii)): J comes from a precomputed spline table in t
      (exact table build via the density rule for s > 0, via the exact cdf
      antiderivative for s = 0)
    * M != 0 (part (iii)): sigma varies with b, so the double sum
      sum_i v_i g_neg(x_i + t_n(b), sigma(b)) is evaluated directly
    """
    if nodes is None:
        nodes = model_nodes(cfg["Sx"], cfg["Su"], cfg["mu"], n_gh)
    Xn, Un, Wn = nodes
    bs, M, sig = cfg["bs"], cfg["M"], cfg["sigma"]
    use_M = bool(np.any(M != 0))
    eps = cfg["eps"]
    if not use_M:
        J = gtable(eps) if sig == 0.0 else jtab(eps, sig)

        def F(b):
            b = np.asarray(b, float).ravel()
            t = Xn @ (bs - b) - Un @ b
            return float(tau_shift(cfg, b) + Wn @ J(t))
        return F
    xr = xrule_light(eps) if xr is None else xr
    xn, vn = xr.nodes, xr.weights

    def F(b):
        b = np.asarray(b, float).ravel()
        t = Xn @ (bs - b) - Un @ b
        sm = np.sqrt(max(float(b @ M @ b), 0.0))
        s = np.sqrt(sm ** 2 + sig ** 2)
        if s == 0.0:
            return float(tau_shift(cfg, b) + Wn @ gtable(eps)(t))
        # sum_i v_i sum_n w_n g_neg(x_i + t_n, s)
        z = (xn[:, None] + t[None, :]) / s
        g = s * (np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi) - z * ndtr(-z))
        return float(tau_shift(cfg, b) + vn @ (g @ Wn))

    return F


def route2_min(cfg, n_gh, starts=None):
    """Minimise the direct (reduction-free) criterion.

    BFGS with numerical gradients first (the criterion is smooth in beta when
    sigma > 0, and C^1 piecewise smooth when sigma = 0), then a short
    Nelder-Mead polish; several starts to guard against local minima.
    """
    F = route2_obj(cfg, n_gh=n_gh)
    if starts is None:
        starts = [claimed_beta(cfg), np.zeros(cfg["p"]),
                  claimed_beta(cfg) + 0.5]
    best, bestv = None, INF
    for s0 in starts:
        try:
            r = optimize.minimize(F, s0, method="BFGS",
                                  options=dict(gtol=1e-10, maxiter=400))
            cand, val = np.asarray(r.x, float), float(r.fun)
        except Exception:
            continue
        if val < bestv:
            best, bestv = cand, val
    try:
        r = optimize.minimize(F, best, method="Nelder-Mead",
                              options=dict(xatol=1e-9, fatol=1e-12,
                                           maxiter=800, maxfev=800))
        if float(r.fun) < bestv:
            best, bestv = np.asarray(r.x, float), float(r.fun)
    except Exception:
        pass
    return dict(bhat=best, fhat=bestv, F=F, bclaim=claimed_beta(cfg),
                fclaim=float(F(claimed_beta(cfg))))


# --------------------------- ROUTE 3 ---------------------------------------
def simulate(cfg, n, rng):
    p = cfg["p"]
    Lx = psd_sqrt(cfg["Sx"])
    Lu = psd_sqrt(cfg["Su"])
    X = rng.normal(size=(n, p)) @ Lx.T + cfg["mu"]
    U = rng.normal(size=(n, p)) @ Lu.T
    W = X + U
    Y = X @ cfg["bs"] + cfg["eps"].rvs(n, rng)
    return Y, W


def empirical_qr_lp(Y, W, tau):
    """Exact minimiser of sum_i rho_tau(Y_i - W_i'b) by sparse LP (HiGHS)."""
    import scipy.sparse as sp
    n, p = W.shape
    c = np.concatenate([np.zeros(2 * p), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A = sp.hstack([sp.csr_matrix(W), -sp.csr_matrix(W),
                   sp.identity(n, format="csr"), -sp.identity(n, format="csr")],
                  format="csr")
    bounds = [(0, None)] * (2 * p + 2 * n)
    res = optimize.linprog(c, A_eq=A, b_eq=Y, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("LP failed: " + str(res.message))
    return res.x[:p] - res.x[p:2 * p]


def fit_smoothed_empirical(Y, W, cfg, b0):
    """Minimise the empirical smoothed criterion (ii)/(iii) with L-BFGS."""
    tau, M, sig = cfg["tau"], cfg["M"], cfg["sigma"]
    use_M = bool(np.any(M != 0))
    n = len(Y)

    def f_and_g(b):
        r = Y - W @ b
        s = np.sqrt(max(float(b @ M @ b), 0.0)) if use_M else 0.0
        s = np.sqrt(s ** 2 + sig ** 2)
        if s <= 1e-14:
            return float(np.mean(rho(tau, r))), -(W.T @ (tau - (r < 0))) / n
        v = float(np.mean(rho_smooth(tau, r, s)))
        g = -(W.T @ (tau - Phi(r / s))) / n
        if use_M:
            g = g - float(np.mean(phi(r / s))) * (M @ b) / s
        return v, g

    res = optimize.minimize(f_and_g, b0, jac=True, method="L-BFGS-B",
                            options=dict(gtol=1e-12, maxiter=5000, ftol=1e-16))
    return np.asarray(res.x, float)


# ---------------------------------------------------------------------------
# 4.  configuration enumeration
# ---------------------------------------------------------------------------
def scalar_configs(quick=False):
    cfgs = []
    taus = [0.1, 0.3, 0.5, 0.7, 0.9] if not quick else [0.1, 0.5, 0.9]
    epsnames = (list(EPS.keys()) if not quick
                else ["gauss", "laplace", "t3", "uniform"])
    sus = [0.0, 0.1, 0.5, 1.0, 2.0]
    sigmas = ([0.0, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
              if not quick else [0.0, 0.1, 1.0, 10.0])
    Ms = [0.0, 0.09, 0.25, 1.0, 4.0] if not quick else [0.0, 1.0]
    for en in epsnames:
        eps = EPS[en]
        for su in sus:
            for tau in taus:
                for sg in sigmas:
                    cfgs.append(make_cfg(
                        "S|%s|su%g|tau%g|s%g" % (en, su, tau, sg), 1, tau, eps,
                        [[1.0]], [[su ** 2]], [[0.0]], sg, [1.0],
                        block="scalar_(i)/(ii)"))
                for mm in Ms:
                    cfgs.append(make_cfg(
                        "S3|%s|su%g|tau%g|M%g" % (en, su, tau, mm), 1, tau, eps,
                        [[1.0]], [[su ** 2]], [[mm]], 0.0, [1.0],
                        block="scalar_(iii)"))
                cfgs.append(make_cfg(
                    "S3|%s|su%g|tau%g|M=Su" % (en, su, tau), 1, tau, eps,
                    [[1.0]], [[su ** 2]], [[su ** 2]], 0.0, [1.0],
                    block="scalar_(iii)_M=Su"))
    return cfgs


def matrix_configs(quick=False):
    out = []
    taus = [0.3, 0.5, 0.7] if not quick else [0.5]
    epsnames = (["gauss", "laplace", "t3", "skewnorm8"] if not quick
                else ["gauss"])
    sigmas = [0.0, 0.5, 2.0] if not quick else [0.0, 1.0]
    for p in ([2, 3, 4] if not quick else [2]):
        bs = np.linspace(1.0, 0.4, p)
        Sx_list = {
            "Sx_diag": np.diag(np.linspace(1.0, 0.5, p)),
            "Sx_corr": sym(0.8 * np.eye(p) + 0.2 * np.ones((p, p))),
            "Sx_ill": posdef_from(np.random.default_rng(7 + p), p, 50.0),
        }
        Su_list = {
            "Su0": np.zeros((p, p)),
            "SuI": 0.7 * np.eye(p),
            "Su_noncomm": posdef_from(np.random.default_rng(31 + p), p, 8.0),
            "Su_rank1": np.outer(np.arange(1.0, p + 1), np.arange(1.0, p + 1)) / p,
            "Su_cSx": 0.5 * sym(0.8 * np.eye(p) + 0.2 * np.ones((p, p))),
        }
        M_list = {
            "M0": np.zeros((p, p)),
            "McI": 0.3 * np.eye(p),
            "M_noncomm": posdef_from(np.random.default_rng(53 + p), p, 5.0),
            "M_rank1": np.outer(np.linspace(1.0, 2.0, p),
                                np.linspace(1.0, 2.0, p)) / p,
            "M=Su": None,
        }
        for sxn, Sx in Sx_list.items():
            for sun, Su in Su_list.items():
                for en in epsnames:
                    eps = EPS[en]
                    for tau in taus:
                        for sg in sigmas:
                            out.append(make_cfg(
                                "M|p%d|%s|%s|%s|tau%g|s%g" % (p, sxn, sun, en, tau, sg),
                                p, tau, eps, Sx, Su, np.zeros((p, p)), sg, bs,
                                block="matrix_(ii)"))
                    for mn, M in M_list.items():
                        out.append(make_cfg(
                            "M3|p%d|%s|%s|%s|%s" % (p, sxn, sun, mn, en), p, 0.4,
                            eps, Sx, Su, Su if M is None else M, 0.0, bs,
                            block="matrix_(iii)"))
    return out


# ---------------------------------------------------------------------------
# 5.  drivers
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-mc", action="store_true")
    ap.add_argument("--n-mc", type=int, default=8000,
                    help="sample size for the LP (exact empirical minimiser)")
    ap.add_argument("--n-mc-grad", type=int, default=2000000,
                    help="sample size for the subgradient test at the claim")
    ap.add_argument("--n-sm", type=int, default=200000,
                    help="sample size for the smoothed empirical fits")
    ap.add_argument("--stage", default="all",
                    help="comma list: val,sweep,route2,mc,uniq,stress,hp")
    args = ap.parse_args()
    stages = set(args.stage.split(","))
    do = lambda s: ("all" in stages) or (s in stages)

    t_start = time.time()
    log_lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log_lines.append(s)

    def save():
        summary["runtime_s"] = time.time() - t_start
        with open(os.path.join(RES, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
        with open(os.path.join(RES, "run_log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

    log("=" * 100)
    log("adversarial check of statement_prop313.md   (quick=%s, stages=%s)"
        % (args.quick, args.stage))
    log("=" * 100)

    summary = dict(quick=args.quick, stages=sorted(stages))
    _sp = os.path.join(RES, "summary.json")
    if os.path.exists(_sp):                  # merge with earlier partial runs
        try:
            with open(_sp) as f:
                old = json.load(f)
            old.update(summary)
            summary = old
        except Exception:
            pass

    # ---------------- 0. validate Xi tables / x-rules / G tables ----------
    if do("val"):
        log("\n[0] validation of the numerical machinery against fresh adaptive")
        log("    quadrature and against Monte Carlo")
        val_rows = []
        rngv = np.random.default_rng(4)
        for name, eps in EPS.items():
            xi = xi_fun(eps)
            ss = np.concatenate([rngv.uniform(1e-6, 1, 30), rngv.uniform(1, 40, 30),
                                 rngv.uniform(40, 300, 15)])
            err = max(abs(float(xi(np.array([s]))[0]) - xi_quad(eps, float(s)))
                      for s in ss)
            xr = xrule(eps)
            # x-rule vs adaptive quadrature of J(t,sigma) for sigma > 0
            xerr = 0.0
            for sg in [0.3, 1.0, 3.0]:
                ts = rngv.uniform(-25, 25, 15)
                xerr = max(xerr, max(abs(float(np.dot(xr.weights,
                                                    g_neg(xr.nodes + t, sg)))
                                         - J_quad(eps, sg, float(t))) for t in ts))
            gerr = 0.0
            if eps.atoms is None:
                G = gtable(eps)
                ts = rngv.uniform(-25, 25, 15)
                gerr = max(abs(float(G(np.array([t]))[0]) - J_quad(eps, 0.0, float(t)))
                           for t in ts)
            # Monte-Carlo cross-check of Xi at a few s
            mc_err = 0.0
            if eps.atoms is None:
                smp = eps.rvs(4000000, np.random.default_rng(3))
                for s in [0.5, 2.0]:
                    zz = np.random.default_rng(5).normal(size=smp.size)
                    ref = float(np.mean(np.maximum(-(smp + s * zz), 0.0)))
                    mc_err = max(mc_err, abs(ref - xi_quad(eps, s)))
            val_rows.append(dict(eps=name, mean=eps.mean(),
                                 sd=float(np.sqrt(eps.var())), xi_err=err,
                                 xrule_err=xerr, gtable_err=gerr, mc_err=mc_err,
                                 has_density=eps.has_density()))
            log("  %-13s mean=%+8.4f sd=%7.4f |Xi err|=%.2e xrule err=%.2e "
                "G err=%.2e  MC check=%.2e"
                % (name, eps.mean(), np.sqrt(eps.var()), err, xerr, gerr, mc_err))
        summary["validation"] = val_rows
        summary["max_xi_err"] = max(r["xi_err"] for r in val_rows)
        summary["max_xrule_err"] = max(r["xrule_err"] for r in val_rows)
        summary["max_gtable_err"] = max(r["gtable_err"] for r in val_rows)
        summary["max_mc_check_err"] = max(r["mc_err"] for r in val_rows)
        log("  -> max|Xi err|=%.2e  max x-rule err=%.2e  max G-table err=%.2e"
            % (summary["max_xi_err"], summary["max_xrule_err"],
               summary["max_gtable_err"]))

        # Gauss-Hermite convergence of route 2 (smooth case sigma>0 and kinked
        # case sigma=0) on a scalar config
        log("\n  route-2 Gauss-Hermite order convergence (p=1, Su=0.25, laplace eps)")
        cfgc = make_cfg("conv", 1, 0.5, EPS["laplace"], [[1.0]], [[0.25]],
                        [[0.0]], 0.0, [1.0])
        conv = []
        for sg in [0.0, 1.0]:
            cfgc["sigma"] = sg
            ref = None
            for n_gh in [10, 20, 30, 45]:
                F = route2_obj(cfgc, n_gh=n_gh)
                v = F(np.array([0.8]))
                r1v = float(route1_obj(cfgc, xi_fun(EPS["laplace"]))(
                    np.array([0.8])))
                if ref is None:
                    ref = r1v
                conv.append(dict(sigma=sg, n_gh=n_gh, F2=v, F1=r1v, diff=v - r1v))
                log("    sigma=%.1f n_gh=%2d: F_route2=%.12f  F_route1=%.12f  "
                    "diff=%+.2e" % (sg, n_gh, v, r1v, v - r1v))
        summary["route2_gh_convergence"] = conv

    # ---------------- 1. broad sweep, route 1 -----------------------------
    if do("sweep"):
        log("\n[1] broad sweep over the required axes (route 1)")
        cfgs = scalar_configs(args.quick) + matrix_configs(args.quick)
        rng = np.random.default_rng(11)
        rows = []
        t0 = time.time()
        for i, cfg in enumerate(cfgs):
            try:
                bhat, fhat, F = route1_min(cfg, xi_fun(cfg["eps"]),
                                           n_restart=2 if not args.quick else 1,
                                           rng=rng)
            except Exception as ex:
                log("  !! %s failed: %r" % (cfg["name"], ex))
                continue
            bc = claimed_beta(cfg)
            fc = F(bc)
            H = numeric_hessian(F, bhat)
            ev = np.linalg.eigvalsh(sym(H))
            rows.append(dict(
                name=cfg["name"], block=cfg["block"], p=cfg["p"], tau=cfg["tau"],
                eps=cfg["eps"].name, sigma=cfg["sigma"],
                Sx=describe_mat(cfg["Sx"]), Su=describe_mat(cfg["Su"]),
                M=describe_mat(cfg["M"]),
                comm_SxSu=commutator(cfg["Sx"], cfg["Su"]),
                comm_SxM=commutator(cfg["Sx"], cfg["M"]),
                err_inf=float(np.max(np.abs(bhat - bc))),
                err_rel=float(np.max(np.abs(bhat - bc))
                              / max(1.0, np.max(np.abs(bc)))),
                gap=fhat - fc, f_at_claim=fc,
                Hess_min_eig=float(ev[0]), Hess_max_eig=float(ev[-1]),
                beta_claim=";".join("%.8g" % v for v in bc),
                beta_num=";".join("%.8g" % v for v in bhat)))
            if (i + 1) % 1000 == 0:
                log("    %5d/%d configs (%.0fs)" % (i + 1, len(cfgs), time.time() - t0))
        log("  finished %d configs in %.1fs" % (len(rows), time.time() - t0))
        if rows:
            summary["n_sweep"] = len(rows)
            summary["max_err_sweep"] = max(r["err_inf"] for r in rows)
            summary["max_gap_sweep"] = max(r["gap"] for r in rows)
            summary["min_Hess_eig_sweep"] = min(r["Hess_min_eig"] for r in rows)
            summary["worst_cfgs"] = sorted(
                rows, key=lambda r: -r["err_inf"])[:10]
            with open(os.path.join(RES, "config_table.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)

    # ---------------- 2. route-2 independent re-check ---------------------
    if do("route2"):
        log("\n[2] independent re-check (route 2: direct Gauss-Hermite over the")
        log("    model variables; no use of the reduction)")
        curated = []
        eps_list = list(EPS.keys())
        if args.quick:
            eps_list = ["gauss", "laplace", "bimodal"]
        for en in eps_list:
            for su in [0.0, 0.5, 1.0]:
                for sg in ([0.0, 0.3, 3.0] if not args.quick else [0.0]):
                    for tau in ([0.3, 0.7] if not args.quick else [0.5]):
                        curated.append(make_cfg(
                            "R2|%s|su%g|s%g|tau%g" % (en, su, sg, tau), 1, tau,
                            EPS[en], [[1.0]], [[su ** 2]], [[0.0]], sg, [1.0],
                            block="route2_scalar"))
        for en in (["gauss", "laplace", "t3", "atomic", "exp_shift3", "bimodal"]
                   if not args.quick else ["gauss"]):
            for mm in ([0.25, 2.0] if not args.quick else [1.0]):
                curated.append(make_cfg("R23|%s|M%g" % (en, mm), 1, 0.5, EPS[en],
                                        [[1.0]], [[0.36]], [[mm]], 0.0, [1.0],
                                        block="route2_scalar_(iii)"))
        Sx2 = np.array([[1.0, 0.6], [0.6, 1.0]])
        Su2 = np.array([[0.5, 0.3], [0.3, 0.9]])
        Su2r1 = np.outer([1.0, 2.0], [1.0, 2.0]) / 3.0
        M2 = np.array([[0.4, -0.25], [-0.25, 0.9]])
        if not args.quick:
            for en in ["gauss", "laplace", "t3", "bimodal", "atomic"]:
                for tau in [0.3]:
                    curated.append(make_cfg(
                        "R2p2|%s|tau%g|Su_noncomm" % (en, tau), 2, tau, EPS[en],
                        Sx2, Su2, np.zeros((2, 2)), 0.0, [1.0, 0.5],
                        block="route2_p2_(ii)"))
                    curated.append(make_cfg(
                        "R2p2|%s|tau%g|Su_rank1" % (en, tau), 2, tau, EPS[en],
                        Sx2, Su2r1, np.zeros((2, 2)), 0.7, [1.0, 0.5],
                        block="route2_p2_(ii)"))
                    if en in ("gauss", "laplace", "t3"):
                        curated.append(make_cfg(
                            "R23p2|%s|tau%g|M_noncomm" % (en, tau), 2, tau, EPS[en],
                            Sx2, Su2r1, M2, 0.0, [1.0, 0.5],
                            block="route2_p2_(iii)"))
        rows2 = []
        t0 = time.time()
        for cfg in curated:
            n_gh = 30 if cfg["p"] == 1 else 8
            r2 = route2_min(cfg, n_gh)
            r1 = route1_min(cfg, xi_fun(cfg["eps"]), n_restart=1,
                            rng=np.random.default_rng(1))
            F1 = route1_obj(cfg, xi_fun(cfg["eps"]))
            rec = dict(name=cfg["name"], block=cfg["block"], p=cfg["p"],
                       eps=cfg["eps"].name, sigma=cfg["sigma"], tau=cfg["tau"],
                       err_r2=float(np.max(np.abs(r2["bhat"] - r2["bclaim"]))),
                       err_r1=float(np.max(np.abs(r1[0] - r2["bclaim"]))),
                       gap_r2=r2["fhat"] - r2["fclaim"],
                       dF_r1_r2_claim=float(F1(r2["bclaim"]) - r2["fclaim"]),
                       comm_SxSu=commutator(cfg["Sx"], cfg["Su"]),
                       comm_SxM=commutator(cfg["Sx"], cfg["M"]))
            rows2.append(rec)
            log("  %-34s p=%d err_r2=%.3e err_r1=%.3e gap=%.3e dF(r1-r2)=%+.2e"
                % (cfg["name"], cfg["p"], rec["err_r2"], rec["err_r1"],
                   rec["gap_r2"], rec["dF_r1_r2_claim"]))
        log("  %d configs in %.1fs;  max err_r2 = %.3e ; max |dF| = %.3e"
            % (len(rows2), time.time() - t0,
               max(r["err_r2"] for r in rows2),
               max(abs(r["dF_r1_r2_claim"]) for r in rows2)))
        summary["n_route2"] = len(rows2)
        summary["max_err_route2"] = max(r["err_r2"] for r in rows2)
        summary["max_dF_route1_route2"] = max(abs(r["dF_r1_r2_claim"]) for r in rows2)
        with open(os.path.join(RES, "route2_table.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
            w.writeheader()
            for r in rows2:
                w.writerow(r)

    # ---------------- 3. Monte-Carlo / empirical route --------------------
    if do("mc") and not args.skip_mc:
        log("\n[3] Monte-Carlo route (raw simulation; LP quantile regression for")
        log("    the check loss; L-BFGS for the smoothed criteria)")
        rows3 = []
        n_mc = args.n_mc if not args.quick else 20000
        n_grad = args.n_mc_grad if not args.quick else 200000
        mc_cfgs = []
        el = ["gauss", "laplace", "t3", "uniform", "skewnorm8", "splitnorm"]
        for en in el:
            for su in [0.0, 0.5, 1.0]:
                for tau in [0.3, 0.9]:
                    mc_cfgs.append(make_cfg("MC|%s|su%g|tau%g" % (en, su, tau), 1,
                                            tau, EPS[en], [[1.0]], [[su ** 2]],
                                            [[0.0]], 0.0, [1.0], block="mc_(i)"))
        mc_cfgs.append(make_cfg("MC|p2", 2, 0.5, EPS["gauss"],
                                [[1.0, 0.6], [0.6, 1.0]],
                                [[0.4, 0.2], [0.2, 0.5]], np.zeros((2, 2)), 0.0,
                                [1.0, 0.5], block="mc_(i)_p2"))
        mc_cfgs.append(make_cfg("MC|p3", 3, 0.7, EPS["laplace"],
                                np.diag([1.0, 0.8, 0.6]), 0.3 * np.eye(3),
                                np.zeros((3, 3)), 0.0, [1.0, 0.5, 0.25],
                                block="mc_(i)_p3"))
        if args.quick:
            mc_cfgs = mc_cfgs[:6]
        rngm = np.random.default_rng(77)
        for cfg in mc_cfgs:
            bc = claimed_beta(cfg)
            devs = []
            for rep in range(5 if not args.quick else 2):
                Y, W = simulate(cfg, n_mc, rngm)
                devs.append(empirical_qr_lp(Y, W, cfg["tau"]) - bc)
            devs = np.array(devs)
            md = devs.mean(axis=0)
            se = (devs.std(axis=0, ddof=1) / np.sqrt(len(devs))
                  if len(devs) > 1 else np.full(cfg["p"], np.nan))
            # population subgradient at the claimed minimiser (big sample)
            Y, W = simulate(cfg, n_grad, rngm)
            r = Y - W @ bc
            comp = W * (cfg["tau"] - (r < 0))[:, None]
            g = comp.mean(axis=0)
            gse = comp.std(axis=0, ddof=1) / np.sqrt(len(Y))
            rec = dict(name=cfg["name"], block=cfg["block"], p=cfg["p"], n=n_mc,
                       n_grad=int(n_grad),
                       mean_dev=md.tolist(), se=se.tolist(),
                       max_abs_z=float(np.nanmax(np.abs(md / se))) if np.any(se > 0)
                       else None,
                       grad=g.tolist(), grad_se=gse.tolist(),
                       max_abs_grad_z=float(np.max(np.abs(g / gse))),
                       bclaim=bc.tolist())
            rows3.append(rec)
            log("  %-20s p=%d n=%d: mean(bhat-bclaim)=%s max|z|=%.2f | "
                "subgradient max|z| (n=%d)=%.2f"
                % (cfg["name"], cfg["p"], n_mc,
                   np.array2string(md, precision=4, floatmode="fixed"),
                   rec["max_abs_z"] or -1, n_grad, rec["max_abs_grad_z"]))

        log("\n[3b] smoothed criteria: empirical (ii) and (iii) by L-BFGS")
        sm_cfgs = [
            make_cfg("MCs|gauss|su0.5|s1", 1, 0.5, EPS["gauss"], [[1.0]], [[0.25]],
                     [[0.0]], 1.0, [1.0], block="mc_(ii)"),
            make_cfg("MCs|laplace|su1|s0.3", 1, 0.5, EPS["laplace"], [[1.0]], [[1.0]],
                     [[0.0]], 0.3, [1.0], block="mc_(ii)"),
            make_cfg("MCs|t3|su1|s2", 1, 0.3, EPS["t3"], [[1.0]], [[1.0]],
                     [[0.0]], 2.0, [1.0], block="mc_(ii)"),
            make_cfg("MCs|bimodal|su0.5|s2", 1, 0.6, EPS["bimodal"], [[1.0]],
                     [[0.25]], [[0.0]], 2.0, [1.0], block="mc_(ii)"),
            make_cfg("MCs3|gauss|M0.5", 1, 0.5, EPS["gauss"], [[1.0]], [[0.25]],
                     [[0.5]], 0.0, [1.0], block="mc_(iii)"),
            make_cfg("MCs3p2|laplace|M_noncomm", 2, 0.7, EPS["laplace"],
                     [[1.0, 0.6], [0.6, 1.0]], [[0.5, 0.3], [0.3, 0.9]],
                     [[0.4, -0.25], [-0.25, 0.9]], 0.0, [1.0, 0.5],
                     block="mc_(iii)_p2"),
        ]
        if args.quick:
            sm_cfgs = sm_cfgs[:2]
        n_sm = args.n_sm if not args.quick else 20000
        for cfg in sm_cfgs:
            bc = claimed_beta(cfg)
            devs = []
            for rep in range(3 if not args.quick else 1):
                Y, W = simulate(cfg, n_sm, rngm)
                devs.append(fit_smoothed_empirical(Y, W, cfg, bc) - bc)
            devs = np.array(devs)
            log("  %-26s n=%d mean(bhat-bclaim)=%s" % (
                cfg["name"], n_sm,
                np.array2string(devs.mean(axis=0), precision=5, floatmode="fixed")))
            rows3.append(dict(name=cfg["name"], block=cfg["block"], p=cfg["p"],
                              n=n_sm, mean_dev=devs.mean(axis=0).tolist(),
                              se=(devs.std(axis=0, ddof=1) / np.sqrt(len(devs))
                                  if len(devs) > 1 else [None] * cfg["p"]),
                              bclaim=bc.tolist()))
        summary["n_mc"] = len(rows3)
        summary["mc_max_abs_grad_z"] = max(r["max_abs_grad_z"] for r in rows3
                                           if "max_abs_grad_z" in r)
        summary["mc_max_abs_dev_z"] = max((r["max_abs_z"] for r in rows3
                                           if r.get("max_abs_z")), default=None)
        with open(os.path.join(RES, "mc_table.csv"), "w", newline="") as f:
            keys = sorted({k for r in rows3 for k in r})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows3:
                w.writerow(r)

    # ---------------- 4. uniqueness diagnostics ---------------------------
    if do("uniq"):
        log("\n[4] uniqueness diagnostics")
        summary["uniqueness"] = uniqueness_diagnostics(log)
        save()

    # ---------------- 5. stress tests -------------------------------------
    if do("stress"):
        log("\n[5] stress tests of the assumptions")
        summary["stress"] = stress_tests(log)
        save()

    # ---------------- 6. huge-bandwidth high-precision check --------------
    if do("hp"):
        log("\n[6] huge-bandwidth check in 50-digit arithmetic (mpmath)")
        summary["huge_bandwidth"] = huge_bandwidth_check(log)
        save()

    # ---------------- 7. flat-criterion cases in 300-digit arithmetic ------
    if do("flat"):
        log("\n[7] errors bounded away from 0 (eps == 10, eps = 3+Exp(1)):")
        log("    the criterion is numerically flat in double precision; redo it")
        log("    in 300-digit arithmetic")
        summary["flat_cases"] = high_precision_flat_cases(log)
        save()

    save()
    log("\nwrote results to " + RES)
    log("total runtime %.1f s" % (time.time() - t_start))


# ---------------------------------------------------------------------------
# 7.  uniqueness diagnostics
# ---------------------------------------------------------------------------
def uniqueness_diagnostics(log):
    out = {}
    # (a) is Xi strictly increasing on [0, inf)?  (crux lemma of the derivation)
    worst = 0.0
    worst_name = None
    for name, eps in EPS.items():
        xi = xi_fun(eps)
        s = np.linspace(0.0, 30.0, 300001)
        d = np.diff(xi(s))
        if -d.min() > worst:
            worst, worst_name = float(-d.min()), name
    out["max_negative_increment_of_Xi"] = worst
    out["max_negative_increment_dist"] = worst_name
    log("  (a) max negative increment of Xi over a 3e5-point grid: %.3e (%s)"
        % (worst, worst_name))

    # (b) scalar (i)/(ii): dense global grid scan, unique local minimum?
    fails = []
    for name, eps in EPS.items():
        xi = xi_fun(eps)
        for su in [0.0, 0.5, 1.0]:
            for tau in [0.1, 0.5, 0.9]:
                for sg in [0.0, 0.7]:
                    cfg = make_cfg("u", 1, tau, eps, [[1.0]], [[su ** 2]],
                                   [[0.0]], sg, [1.0])
                    bc = claimed_beta(cfg)[0]
                    for st in [0.05, 0.001]:
                        gg = np.arange(-8.0, 8.0 + st, st)
                        v = xi(np.sqrt((gg - 1.0) ** 2 + su ** 2 * gg ** 2 + sg ** 2))
                        i = int(np.argmin(v))
                        if 0 < i < len(gg) - 1 and abs(gg[i] - bc) > 5 * st:
                            fails.append((name, su, tau, sg, st, float(gg[i]),
                                          float(bc)))
    out["scalar_grid_scan_failures"] = fails
    log("  (b) scalar global grid scans (all 12 eps x su x tau x sigma): "
        "%d failures" % len(fails))

    # (c) p-dim: 200 random line scans through the claim must increase away
    mono_bad = []
    rng = np.random.default_rng(9)
    for p in [2, 3, 4]:
        Sx = posdef_from(np.random.default_rng(100 + p), p, 5.0)
        Su = posdef_from(np.random.default_rng(200 + p), p, 3.0)
        M = posdef_from(np.random.default_rng(300 + p), p, 4.0)
        bs = np.linspace(1.0, 0.5, p)
        for sig, Mus, tag in [(0.0, np.zeros((p, p)), "(i)"),
                              (1.7, np.zeros((p, p)), "(ii)"),
                              (0.0, M, "(iii)")]:
            cfg = make_cfg("u", p, 0.4, EPS["laplace"], Sx, Su, Mus, sig, bs)
            F = route1_obj(cfg, xi_fun(EPS["laplace"]))
            bc = claimed_beta(cfg)
            f0 = F(bc)
            for _ in range(200):
                d = rng.normal(size=p)
                d /= np.linalg.norm(d)
                ts = np.concatenate([np.linspace(1e-3, 3.0, 60),
                                     np.linspace(3.0, 30.0, 20)])
                vals = np.array([F(bc + t * d) for t in ts])
                if np.any(vals < f0 - 1e-11):
                    mono_bad.append((p, sig, tag, float(f0 - vals.min())))
    out["line_scan_violations"] = mono_bad
    log("  (c) 200 random line scans through the claim for p=2,3,4 and each of "
        "(i),(ii),(iii): %d violations" % len(mono_bad))

    # (d) numeric Hessian of the objective at the claim: positive definite?
    hess_bad = []
    for p in [2, 3, 4]:
        Sx = posdef_from(np.random.default_rng(400 + p), p, 8.0)
        Su = posdef_from(np.random.default_rng(500 + p), p, 8.0)
        M = posdef_from(np.random.default_rng(600 + p), p, 8.0)
        bs = np.linspace(1.0, 0.5, p)
        for sig, Mus in [(0.0, np.zeros((p, p))), (0.0, M)]:
            cfg = make_cfg("u", p, 0.5, EPS["t3"], Sx, Su, Mus, sig, bs)
            F = route1_obj(cfg, xi_fun(EPS["t3"]))
            H = numeric_hessian(F, claimed_beta(cfg))
            ev = np.linalg.eigvalsh(sym(H))
            if ev[0] <= 0:
                hess_bad.append((p, sig, float(ev[0])))
    out["hessian_not_pd"] = hess_bad
    log("  (d) numeric Hessian at the claim (p=2,3,4, (i) and (iii)): "
        "%d non-PD" % len(hess_bad))
    return out


# ---------------------------------------------------------------------------
# 8.  stress tests of the assumptions
# ---------------------------------------------------------------------------
def stress_tests(log):
    out = {}

    # (a) E[X] != 0  -- the claim assumes mu_x = 0
    recs = []
    for p, mu, Sx, Su, en, tau in [
        (1, [1.0], [[1.0]], [[0.25]], "gauss", 0.5),
        (1, [0.3], [[1.0]], [[0.25]], "gauss", 0.5),
        (1, [2.0], [[1.0]], [[0.25]], "laplace", 0.5),
        (1, [1.0], [[1.0]], [[0.0]], "gauss", 0.5),
        (1, [1.0], [[1.0]], [[0.0]], "gauss", 0.3),
        (1, [1.0], [[1.0]], [[0.25]], "gauss", 0.3),
        (2, [0.8, -0.6], [[1.0, 0.4], [0.4, 1.0]], [[0.3, 0.1], [0.1, 0.2]], "gauss",
         0.5),
    ]:
        cfg = make_cfg("mu", p, tau, EPS[en], Sx, Su, np.zeros((p, p)), 0.0,
                       np.linspace(1.0, 0.5, p), mu=mu)
        r = route2_min(cfg, 30 if p == 1 else 16)
        bc = claimed_beta(cfg)
        rec = dict(p=p, mu=list(np.asarray(mu, float)), eps=en, tau=tau,
                   beta_claim=bc.tolist(), beta_num=r["bhat"].tolist(),
                   dev_inf=float(np.max(np.abs(r["bhat"] - bc))),
                   F_claim=r["fclaim"], F_num=r["fhat"],
                   gap=r["fclaim"] - r["fhat"])
        recs.append(rec)
        log("  (a) mu=%s p=%d %-8s tau=%.1f: claim=%s num=%s  dev=%.4f  "
            "F(claim)-F(min)=%+.3e"
            % (np.array2string(np.asarray(mu, float), precision=2), p, en, tau,
               np.array2string(bc, precision=4),
               np.array2string(r["bhat"], precision=4), rec["dev_inf"],
               rec["gap"]))
    out["mu_x_nonzero"] = recs

    # (b) singular Sx
    Sx_sing = np.array([[1.0, 0.0], [0.0, 0.0]])
    recs = []
    for label, Su in [("Su=0", np.zeros((2, 2))), ("Su=0.5I", 0.5 * np.eye(2))]:
        cfg = make_cfg("Sxsing", 2, 0.5, EPS["gauss"], Sx_sing, Su,
                       np.zeros((2, 2)), 0.0, [1.0, 0.0])
        F = route1_obj(cfg, xi_fun(EPS["gauss"]))
        b0 = claimed_beta(cfg)
        vals = [F(b0 + np.array([0.0, t])) for t in [-5, -1, 0, 1, 5]]
        rec = dict(Su=label, beta_claim=b0.tolist(), F_along_e2=vals,
                   spread=float(max(vals) - min(vals)),
                   note="Sx+Su is singular: bbar is only a least-squares "
                        "solution and the minimiser is a whole affine set"
                        if label == "Su=0" else
                        "Sx+Su > 0: unique minimiser, claim holds")
        recs.append(rec)
        log("  (b) singular Sx, %-6s: F along the unidentified direction e2 = %s "
            "(spread %.2e)" % (label, np.array2string(np.array(vals), precision=6),
                               rec["spread"]))
    out["Sx_singular"] = recs

    # (c) E|eps| = inf (Cauchy): criterion is +inf for every beta
    cauchy = Eps("cauchy", stats.cauchy().cdf,
                 lambda n, r: stats.cauchy().rvs(size=n, random_state=r),
                 pdf=stats.cauchy().pdf, ppf=stats.cauchy().ppf, note="E|eps|=inf")
    vals = []
    for b in [0.0, 0.5, 1.0]:
        Y, W = simulate(make_cfg("c", 1, 0.5, cauchy, [[1.0]], [[0.25]], [[0.0]],
                                 0.0, [1.0]), 400000, np.random.default_rng(0))
        vals.append(float(rho(0.5, Y - b * W[:, 0]).mean()))
    out["cauchy"] = dict(sample_means=vals,
                         note="E|eps| = inf: population criterion is +inf")
    log("  (c) Cauchy eps (E|eps|=inf): sample criterion means = %s "
        "(diverge; population value is +inf)" % np.array2string(np.array(vals),
                                                                precision=2))

    # (d) M indefinite: sigma_M(b) = sqrt(b'Mb) is not real-valued
    p = 2
    Mind = np.array([[1.0, 0.0], [0.0, -0.4]])
    btest = np.array([0.0, 1.0])
    out["M_indefinite"] = dict(
        bMb=float(btest @ Mind @ btest),
        b=btest.tolist(),
        note="b'Mb < 0 for some b: sqrt(b'Mb) is not real, so the criterion of "
             "(iii) is not defined; the claim explicitly requires M >= 0")
    log("  (d) indefinite M: b'Mb = %.3f < 0 for b=%s -> sigma_M(b) undefined"
        % (out["M_indefinite"]["bMb"], btest.tolist()))
    return out


# ---------------------------------------------------------------------------
# 9.  huge-bandwidth check with mpmath
# ---------------------------------------------------------------------------
def high_precision_flat_cases(log):
    """Errors bounded away from 0 (eps == 10, eps = 3 + Exp(1)).

    For these, Xi(s) = E[max(-(eps+sZ),0)] is astronomically small for
    moderate s (the loss is locally linear), so the population criterion is
    EXACTLY 0.0 in double precision over a whole ellipsoid of beta: the
    criterion is mathematically still strictly increasing but numerically
    flat, and a double-precision optimiser returns an arbitrary point of that
    ellipsoid.  Redo the comparison in 300-digit arithmetic.
    """
    import mpmath as mp
    mp.mp.dps = 300
    out = []

    def ncdf_safe(z):
        """Phi(z) in arbitrary precision; asymptotic series for |z| > 20
        (mpmath's erfc overflows for very large arguments)."""
        z = mp.mpf(z)
        if abs(z) <= 20:
            return mp.ncdf(z)
        t = mp.npdf(z) / abs(z) * (1 - 1 / z ** 2 + 3 / z ** 4 - 15 / z ** 6
                                   + 105 / z ** 8)
        return t if z < 0 else 1 - t

    def R_case(which, su, M, sigma, b):
        b = mp.mpf(b)
        q = (1 - b) ** 2 + mp.mpf(su) ** 2 * b ** 2 + mp.mpf(M) * b ** 2 \
            + mp.mpf(sigma) ** 2
        s = mp.sqrt(q)
        # Xi(s) = E[max(-(eps + sZ),0)] = E[ s*phi(eps/s) - eps*Phi(-eps/s) ]
        if which == "const10":
            m = mp.mpf(10)
            if s == 0:
                return max(-m, mp.mpf(0))
            return s * mp.npdf(m / s) - m * ncdf_safe(-m / s)
        if which == "atomic":            # 0.4 d_{-1} + 0.6 d_{+2}, exactly
            if s == 0:
                return mp.mpf('0.4') * 1 + mp.mpf('0.6') * 0
            return (mp.mpf('0.4') * (s * mp.npdf(-1 / s) - (-1) * ncdf_safe(1 / s))
                    + mp.mpf('0.6') * (s * mp.npdf(2 / s) - 2 * ncdf_safe(-2 / s)))
        if which == "exp_shift3":
            def g(y):
                m = 3 + y
                if s == 0:
                    return max(-m, mp.mpf(0))
                return s * mp.npdf(m / s) - m * ncdf_safe(-m / s)
            return mp.quad(lambda y: g(y) * mp.e ** (-y), [0, 1, 5, 20, mp.inf])
        raise ValueError(which)

    for which in ["const10", "exp_shift3", "atomic"]:
        mp.mp.dps = 120 if which == "exp_shift3" else 300
        for su in ["0.0", "0.5", "1.0"]:
            for M, sigma, tag in [("0", "0.5", "(ii)"), ("0", "0", "(i)")]:
                bbar = mp.mpf(1) / (1 + mp.mpf(su) ** 2 + mp.mpf(M))
                f0 = R_case(which, su, M, sigma, bbar)
                row = dict(eps=which, su=su, M=M, sigma=sigma, part=tag,
                           beta_claim=float(bbar), R_claim=mp.nstr(f0, 12))
                for h in ["0.1", "0.01", "0.001"]:
                    dp = R_case(which, su, M, sigma, bbar + mp.mpf(h)) - f0
                    dm = R_case(which, su, M, sigma, bbar - mp.mpf(h)) - f0
                    row["dR_plus_" + h] = mp.nstr(dp, 12)
                    row["dR_minus_" + h] = mp.nstr(dm, 12)
                    row["dR_plus_" + h + "_dp"] = float(dp)
                    row["dR_minus_" + h + "_dp"] = float(dm)
                # high-precision argmin on a coarse grid.  No local refinement:
                # the landscape is flat to ~1e-200, so any optimiser is limited
                # by resolution, not by the mathematics.  (Skipped for the
                # shifted-exponential law, whose quadrature is expensive at high
                # precision; the increments below already prove monotonicity.)
                if which == "exp_shift3":
                    row["beta_argmin_hp"] = "not searched"
                    row["argmin_err"] = "n/a"
                else:
                    lo, hi = float(bbar) - 2.0, float(bbar) + 2.0
                    grid = np.linspace(lo, hi, 41)
                    vals = [R_case(which, su, M, sigma, mp.mpf(g)) for g in grid]
                    i = min(range(len(vals)), key=lambda j: vals[j])  # mpf compare
                    row["beta_argmin_hp"] = "%.6f" % grid[i]
                    row["argmin_err"] = "%.4f (grid resolution 0.1)" % (
                        grid[i] - float(bbar))
                    log("  %-11s su=%-4s M=%s sigma=%-4s %s: bbar=%.10f  "
                        "grid argmin=%.4f" % (which, su, M, sigma, tag,
                                              float(bbar), grid[i]))
                row["flatness_log10_dR"] = float(np.log10(max(
                    float(row["dR_plus_0.001_dp"]), 1e-320))) \
                    if float(row["dR_plus_0.001_dp"]) > 0 else None
                out.append(row)
                log("        R(claim)=%s  dR(+0.01)=%s  dR(-0.01)=%s  "
                    "(double precision: %s, %s)"
                    % (row["R_claim"], row["dR_plus_0.01"], row["dR_minus_0.01"],
                       row["dR_plus_0.01_dp"], row["dR_minus_0.01_dp"]))
        # atomic law: the flatness is a vanishing derivative, not underflow
        for su in ["0.0", "0.5", "1.0"]:
            cfg = make_cfg("atomic", 1, 0.5, EPS["atomic"], [[1.0]], [[float(su) ** 2]],
                           [[0.0]], 0.0, [1.0])
            bbar = 1.0 / (1.0 + float(su) ** 2)
            xi = xi_fun(EPS["atomic"])
            row = dict(eps="atomic", su=su, M="0", sigma="0", part="(i)",
                       beta_claim=bbar, R_claim="%.15g" % float(xi(np.array(
                           [np.sqrt((1 - bbar) ** 2 + float(su) ** 2 * bbar ** 2)]))[0]))
            for h in ["0.1", "0.01", "0.001"]:
                s_p = np.sqrt((1 - (bbar + float(h))) ** 2
                              + float(su) ** 2 * (bbar + float(h)) ** 2)
                s_m = np.sqrt((1 - (bbar - float(h))) ** 2
                              + float(su) ** 2 * (bbar - float(h)) ** 2)
                dp = float(xi(np.array([s_p]))[0]) - float(xi(np.array(
                    [np.sqrt((1 - bbar) ** 2 + float(su) ** 2 * bbar ** 2)]))[0])
                dm = float(xi(np.array([s_m]))[0]) - float(xi(np.array(
                    [np.sqrt((1 - bbar) ** 2 + float(su) ** 2 * bbar ** 2)]))[0])
                # exact derivative Xi'(s) = E[phi(eps/s)] > 0, computed in logs
                # (its magnitude is ~1e-217 at s = 1e-3, i.e. far below double
                #  precision: strict monotonicity is real but unobservable)
                s_chk = float(h)
                lse = np.logaddexp(np.log(0.4) - 0.5 * (1.0 / s_chk) ** 2,
                                   np.log(0.6) - 0.5 * (2.0 / s_chk) ** 2)
                row["dR_plus_" + h] = "%.6g" % dp
                row["dR_minus_" + h] = "%.6g" % dm
                row["dR_plus_" + h + "_dp"] = dp
                row["dR_minus_" + h + "_dp"] = dm
                row["log10_exact_slope"] = float((lse - 0.5 * np.log(2 * np.pi))
                                                 / np.log(10.0))
            row["beta_argmin_hp"] = "%.15g" % bbar
            row["argmin_err"] = "0 (exact closed form)"
            out.append(row)
            log("  %-11s su=%-4s M=%s sigma=%-4s %s: bbar=%.10f  R=%.12g  "
                "dR(+0.01)=%.3e dR(-0.01)=%.3e"
                % ("atomic", su, "0", "0", "(i)", bbar, float(row["R_claim"]),
                   row["dR_plus_0.01_dp"], row["dR_minus_0.01_dp"]))
    return dict(rows=out)


def huge_bandwidth_check(log):
    """For sigma huge the contrast of R_sigma is O(1/sigma), invisible in double
    precision.  Redo the whole thing in 50-digit arithmetic.

    Scalar model: Sx = 1, Su = 1/4, beta* = 1, tau arbitrary (it drops out
    because E[Y - b'W] = 0 here).  R_sigma(b) = Xi( sqrt( (1-b)^2 + b^2/4 + s^2 ) )
    with Xi(s) = E[max(-(eps+sZ),0)].
    """
    import mpmath as mp
    mp.mp.dps = 50
    res = []

    def Xi_gauss(s):
        # eps ~ N(0,1):  eps + sZ ~ N(0, 1+s^2)
        return mp.sqrt(1 + s ** 2) / mp.sqrt(2 * mp.pi)

    def Xi_lap(s):
        # eps ~ Laplace(0, 1/sqrt2):  Xi(s) = E[ s*phi(eps/s) - eps*Phi(-eps/s) ]
        # (single 1-D quadrature against the density; the density controls the
        #  integration range, so this stays cheap even for huge sigma)
        def f(x):
            x = mp.mpf(x)
            return (s * mp.npdf(x / s) - x * mp.ncdf(-x / s)) \
                * (1 / mp.sqrt(2)) * mp.e ** (-mp.sqrt(2) * abs(x))
        return mp.quad(f, [-mp.inf, -40, -5, 0, 5, 40, mp.inf])

    for sig in [1e3, 1e6, 1e9, 1e12]:
        for en in ["gauss", "laplace"]:
            bbar = mp.mpf(1) / mp.mpf('1.25')      # (Sx+Su)^-1 Sx beta*

            def R(b):
                b = mp.mpf(b)
                q = (1 - b) ** 2 + mp.mpf('0.25') * b ** 2 + mp.mpf(sig) ** 2
                return Xi_gauss(mp.sqrt(q)) if en == "gauss" else Xi_lap(mp.sqrt(q))

            f0 = R(bbar)
            h = mp.mpf('1e-4')
            fp = R(bbar + h) - f0
            fm = R(bbar - h) - f0
            fp2 = R(bbar + mp.mpf('0.5')) - f0
            fm2 = R(bbar - mp.mpf('0.5')) - f0
            curv = (fp + fm) / h ** 2
            rec = dict(sigma=sig, eps=en, beta_claim=float(bbar),
                       R_at_claim=mp.nstr(f0, 20),
                       dR_plus_1e4=mp.nstr(fp, 6), dR_minus_1e4=mp.nstr(fm, 6),
                       dR_plus_0p5=mp.nstr(fp2, 6), dR_minus_0p5=mp.nstr(fm2, 6),
                       curvature=mp.nstr(curv, 6))
            res.append(rec)
            log("  sigma=%-8g eps=%-8s R(claim)=%s  dR(+1e-4)=%s dR(-1e-4)=%s  "
                "dR(+-0.5)=%s,%s  curv=%s"
                % (sig, en, rec["R_at_claim"], rec["dR_plus_1e4"],
                   rec["dR_minus_1e4"], rec["dR_plus_0p5"], rec["dR_minus_0p5"],
                   rec["curvature"]))
    return dict(checks=res)


if __name__ == "__main__":
    main()

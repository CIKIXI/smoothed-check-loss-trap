"""
Solvers and data generators shared by the Route-A experiments.

Design philosophy
-----------------
Every estimator here is fitted by the SAME optimiser on the SAME smoothed loss
    rho_h(r) = r(Phi(r/h) - (1-tau)) + h phi(r/h)      (= E_{Z~N(0,h^2)}[rho_tau(r-Z)])
and differs ONLY in
    (i)  the design it is fitted on  (raw W  vs  calibrated mu_W = E[X|W]),
    (ii) the bandwidth h             (h -> 0 : computational smoothing only;
                                      h = sigma_hat : the manuscript's "corrected" loss).
This isolates the two things the paper conflates.

rho_h with h -> 0 is the ordinary check loss; the smoothing bias is O(h^2), so with
h = 0.01 it is ~1e-4, i.e. three orders of magnitude below the effects studied here.
"""

import numpy as np
from scipy.special import ndtr


# ------------------------------------------------------------------ loss pieces
def phi_std(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)


def rho_h(r, h, tau):
    h = max(float(h), 1e-12)
    z = np.asarray(r, float) / h
    return r * (ndtr(z) - (1 - tau)) + h * phi_std(z)


def psi_h(r, h, tau):
    h = max(float(h), 1e-12)
    return ndtr(np.asarray(r, float) / h) - (1 - tau)


# ------------------------------------------------------------------ generators
def draw_eps(n, tau, rng):
    """Errors whose tau-th quantile is 0 (Laplace for tau=1/2)."""
    if tau == 0.5:
        return rng.laplace(0.0, 1.0, n)
    return (1 - tau) * rng.exponential(1.0, n) - tau * rng.exponential(1.0, n)


def gen_lowdim(n, p, s, sigma_u, tau, seed, beta_val=2.0):
    rng = np.random.RandomState(seed)
    beta = np.zeros(p)
    beta[:s] = beta_val
    X = rng.standard_normal((n, p))
    U = rng.standard_normal((n, p)) * sigma_u
    Y = X @ beta + draw_eps(n, tau, rng)
    return X, X + U, Y, beta


def gen_highdim(n, p, s, sigma_u, tau, seed, beta_scale=2.0, beta_fixed=None):
    """Same DGP as the manuscript's own Table 1 (large_scale_experiments.py:29-32),
    except that beta* can be held fixed across replications (pass beta_fixed)."""
    rng = np.random.RandomState(seed)
    if beta_fixed is None:
        beta = np.zeros(p)
        beta[:s] = rng.randn(s) * beta_scale
    else:
        beta = np.asarray(beta_fixed, float)
    X = rng.standard_normal((n, p))
    U = rng.standard_normal((n, p)) * sigma_u
    Y = X @ beta + draw_eps(n, tau, rng)
    return X, X + U, Y, beta


# ------------------------------------------------------------------ solvers
def fit_smooth_unpenalized(Y, D, h, tau, beta0=None, max_iter=300, tol=1e-11):
    """Damped Newton for  mean rho_h(Y - D b).  D must already contain any intercept
    column.  Returns (beta, max|grad|)."""
    n, p = D.shape
    b = np.zeros(p) if beta0 is None else np.array(beta0, float)

    def fg(bb):
        r = Y - D @ bb
        g_i = psi_h(r, h, tau)
        val = float(np.mean(rho_h(r, h, tau)))
        grad = -(D.T @ g_i) / n
        return val, grad, r

    for _ in range(max_iter):
        val, grad, r = fg(b)
        if np.max(np.abs(grad)) < tol:
            break
        w = phi_std(r / h) / h
        H = (D.T * w) @ D / n
        lam, Q = np.linalg.eigh(H)
        lam = np.maximum(lam, max(lam.max(), 1e-300) * 1e-13)
        d = Q @ ((Q.T @ grad) / lam)
        gd = float(grad @ d)
        t = 1.0
        for _ in range(60):
            if fg(b - t * d)[0] <= val - 1e-4 * t * gd:
                break
            t *= 0.5
        b = b - t * d
    return b, float(np.max(np.abs(fg(b)[1])))


def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def fit_smooth_l1(Y, D, h, tau, lam, step=0.005, max_iter=1500, tol=1e-6,
                  beta0=None, warm_iters=None):
    """Proximal gradient for  mean rho_h(Y - D b) + lam*||b||_1  (b excluding intercept:
    pass the intercept as a separate scalar through `beta0`/unpenalised first column is
    NOT allowed -- use fit_smooth_l1_with_intercept)."""
    n, p = D.shape
    b = np.zeros(p) if beta0 is None else np.array(beta0, float)
    losses = []
    for _ in range(max_iter):
        r = Y - D @ b
        grad = -(D.T @ psi_h(r, h, tau)) / n
        eta = step
        for _ in range(12):
            b_try = soft_threshold(b - eta * grad, eta * lam)
            if np.mean(rho_h(Y - D @ b_try, h, tau)) <= np.mean(rho_h(r, h, tau)):
                break
            eta *= 0.5
        b = b_try
        losses.append(float(np.mean(rho_h(Y - D @ b, h, tau)) + lam * np.abs(b).sum()))
        if len(losses) > 5 and abs(losses[-1] - losses[-2]) < tol * abs(losses[-2] + 1e-12):
            break
    return b, losses[-1] if losses else np.nan


def fit_smooth_l1_intercept(Y, D, h, tau, lam, step=0.005, max_iter=1500,
                            tol=1e-6, beta0=None):
    """Same, but the last column of D is an unpenalised intercept."""
    n, p = D.shape
    b = np.zeros(p) if beta0 is None else np.array(beta0, float)
    lam_vec = np.full(p, float(lam))
    lam_vec[-1] = 0.0
    losses = []
    for _ in range(max_iter):
        r = Y - D @ b
        grad = -(D.T @ psi_h(r, h, tau)) / n
        eta = step
        for _ in range(12):
            b_try = soft_threshold(b - eta * grad, eta * lam_vec)
            if np.mean(rho_h(Y - D @ b_try, h, tau)) <= np.mean(rho_h(r, h, tau)):
                break
            eta *= 0.5
        b = b_try
        losses.append(float(np.mean(rho_h(Y - D @ b, h, tau))
                            + float(np.abs(b[:-1]).sum()) * lam))
        if len(losses) > 5 and abs(losses[-1] - losses[-2]) < tol * abs(losses[-2] + 1e-12):
            break
    return b, losses[-1] if losses else np.nan


# ------------------------------------------------------------------ calibration
def calibration_matrix(Sigma_x, Sigma_u):
    """C = Sigma_x (Sigma_x + Sigma_u)^{-1}, so that mu_W = C W."""
    return Sigma_x @ np.linalg.inv(Sigma_x + Sigma_u)


def posterior_cov(Sigma_x, Sigma_u):
    """Var(X | W) for jointly Gaussian (X, W)."""
    C = calibration_matrix(Sigma_x, Sigma_u)
    return Sigma_x - C @ Sigma_x

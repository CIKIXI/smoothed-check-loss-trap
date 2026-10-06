"""
Generate the figures of paper/main.tex from the saved JSON results (plus one direct
computation for the population-loss figure).
"""

import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, roots_laguerre

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Publication style: Computer Modern to match the LaTeX body font (matplotlib ships
# cmr10 and the mathtext 'cm' fontset), figures sized so that at \textwidth the
# effective font size stays above 7pt in both the manuscript and the IMS layout.
plt.rcParams.update({
    "font.size": 9.5,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.framealpha": 0.92,
    # embed TrueType rather than Type 3 fonts in the vector output: publisher preflight
    # tools routinely flag Type 3 as unusable for typesetting.
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_MAIN = "#1f4e79"
C_BAD = "#c0392b"
C_GOOD = "#1e8449"
C_MID = "#b9770e"
C_OR = "#6c3483"


def phi(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / np.sqrt(2 * np.pi)


def save_fig(fig, name):
    """Write both a raster (quick viewing) and a vector (journal submission) version."""
    png = os.path.join(FIGDIR, name + ".png")
    pdf = os.path.join(FIGDIR, name + ".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def rho_tilde(r, s, tau=0.5):
    r = np.asarray(r, float)
    if s <= 0:
        return r * (tau - (r < 0))
    z = r / s
    return r * (ndtr(z) - (1 - tau)) + s * phi(z)


def check_loss(u, tau=0.5):
    return u * (tau - (u < 0).astype(float))


# Gauss-Laguerre nodes/weights for the Laplace density, used below for exact curves.
_LAG_NODES, _LAG_WEIGHTS = roots_laguerre(120)


def loss_of_scale(s, tau=0.5):
    r"""$\E[\rho_\tau(\varepsilon+G)]$ for $\varepsilon\sim$ Laplace$(0,1)$ (the case
    $\tau=1/2$) and $G\sim\cN(0,s^2)$ independent of it, evaluated by Gauss-Laguerre
    quadrature in the Laplace density:
    $\frac12\int_0^\infty[\tilde\rho_\tau(u,s)+\tilde\rho_\tau(-u,s)]e^{-u}du$."""
    return 0.5 * float(np.sum(_LAG_WEIGHTS * (rho_tilde(_LAG_NODES, s, tau)
                                              + rho_tilde(-_LAG_NODES, s, tau))))


# ------------------------------------------------------------------ Figure: losses
def fig_losses(sigma_u=0.5, beta_star=2.0, tau=0.5):
    r"""Population objectives, computed exactly rather than simulated.

    Writing $\lambda=1/(1+\sigma_u^2)$, $\Sxw=\lambda\sigma_u^2$ and
    $\xi=X-\Ccal W$ (so $\Var\xi=\Sxw$, $\xi\perp W$), every objective below is
    $\E[\rho_\tau(\varepsilon+G(\beta))]$ with $G(\beta)\sim\cN(0,s^2(\beta))$, hence
    equals `loss_of_scale(s(beta))`; the scale functions are

      smoothed at $W$, pilot scale:  $s^2=(\beta^*-\beta)^2+\beta^2\sigma_u^2+\hat\sigma^2$
      true risk:                     $s^2=(\beta^*-\beta)^2$
      check loss at $\mu_W$:         $s^2=\lambda(\beta^*-\beta)^2+\Sxw\beta^{*2}$
      posterior kernel, adaptive:    $s^2=\lambda(\beta^*-\beta)^2+\Sxw\beta^{*2}+\Sxw\beta^2$

    The minimisers are therefore the population ones -- $\bbar=\lambda\beta^*$ for the
    first and the last, $\betas$ for the middle two (Proposition 3.13) -- up to the
    accuracy of the one-dimensional optimiser, and no Monte-Carlo error enters.
    """
    lam = 1.0 / (1.0 + sigma_u ** 2)
    Sxw = lam * sigma_u ** 2
    beta_bar = lam * beta_star
    sigma_hat = abs(beta_bar) * sigma_u          # Stage-1 (pilot) scale

    def s_smoothed(b):
        return np.sqrt((beta_star - b) ** 2 + b ** 2 * sigma_u ** 2 + sigma_hat ** 2)

    def s_true(b):
        return np.abs(beta_star - b)

    def s_rc(b):
        return np.sqrt(lam * (beta_star - b) ** 2 + Sxw * beta_star ** 2)

    def s_ctrl(b):
        return np.sqrt(lam * (beta_star - b) ** 2 + Sxw * beta_star ** 2 + Sxw * b ** 2)

    def argmin(f):
        r = minimize_scalar(lambda b: loss_of_scale(f(b)), bounds=(0.5, 3.5),
                            method="bounded", options={"xatol": 1e-12})
        return float(r.x)

    curves = [
        ("smoothed", s_smoothed, C_BAD, "-",
         r"smoothed at $W$"),
        ("true", s_true, C_MAIN, "--",
         r"true risk"),
        ("rc", s_rc, C_GOOD, "-",
         r"calibrated check loss"),
        ("ctrl", s_ctrl, C_MID, "-.",
         r"posterior kernel, adaptive"),
    ]
    mins = {n: argmin(f) for n, f, _, _, _ in curves}
    loss = {n: loss_of_scale(f(mins[n])) for n, f, _, _, _ in curves}

    grid = np.arange(0.9, 2.7001, 0.002)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    style = dict((c[0], c[1:]) for c in curves)

    def draw(ax, names):
        for n in names:
            f, col, ls, lab = style[n]
            y = np.array([loss_of_scale(f(b)) for b in grid]) - loss[n]
            ax.plot(grid, y, color=col, lw=1.8, ls=ls,
                    label=f"{lab}, min ${mins[n]:.3f}$")
            ax.plot([mins[n]], [0.0], marker="o", ms=4.5, color=col, zorder=5)
        for x in (beta_bar, beta_star):
            ax.axvline(x, color="0.45", lw=0.8, ls=(0, (1, 2)), zorder=0)
        ax.set_ylim(-0.004, 0.185)
        ax.set_xlim(0.98, 2.62)
        ax.legend(fontsize=7.5, loc="upper center", handlelength=1.9)
        ax.set_xlabel(r"candidate coefficient $\beta$")

    ax = axes[0]
    draw(ax, ["smoothed", "true"])
    ax.set_ylabel("excess population loss")
    ax.set_title(r"(a) smoothing moves the minimiser to $\bar\beta$", fontsize=9.5)

    ax = axes[1]
    draw(ax, ["rc", "ctrl", "smoothed"])
    ax.set_title(r"(b) conditioning keeps it at $\beta^\ast$", fontsize=9.5)

    plt.tight_layout()
    out = save_fig(fig, "fig1_loss_curves")
    report = dict(sigma_u=sigma_u, tau=tau, beta_star=beta_star, beta_bar=beta_bar,
                  sigma_hat=sigma_hat, Sxw=Sxw, method="quadrature",
                  argmin_smoothed=mins["smoothed"], argmin_true_risk=mins["true"],
                  argmin_rc=mins["rc"], argmin_control_posterior=mins["ctrl"])
    with io.open("simulations/fig1_loss_argmins.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"[figure] {out}   argmins: smoothed={mins['smoothed']:.6f} "
          f"true={mins['true']:.6f} rc={mins['rc']:.6f} "
          f"control={mins['ctrl']:.6f}")
    print("[saved] simulations/fig1_loss_argmins.json")
    return report


# ------------------------------------------------------------------ Figure: scaling
def fig_scaling():
    with io.open("simulations/routeA_results_E2.json", encoding="utf-8") as fh:
        res = json.load(fh)
    meta = res["_meta"]
    ns = sorted({int(k.split("_n")[1]) for k in res if k != "_meta"})
    taus = sorted({float(k.split("_")[0][3:]) for k in res if k != "_meta"})

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axes[0]
    tau = 0.5
    def series(method):
        vals = [res[f"tau{tau}_n{n}"][method] for n in ns]
        ses = [res[f"tau{tau}_n{n}"][method + "_se"] for n in ns]
        return np.array(vals), np.array(ses)
    for method, lab, c, m in (("conv", "smoothed (convolved)", C_BAD, "o"),
                              ("naive", "naive QR on $W$", C_MID, "s"),
                              ("rc", "RC (calibrated)", C_GOOD, "^"),
                              ("rc_int", "RC + intercept", C_GOOD, "v"),
                              ("oracle", "oracle (clean $X$)", C_MAIN, "D")):
        v, s = series(method)
        ax.errorbar(ns, v, yerr=s, marker=m, color=c, lw=1.5, ms=4,
                    capsize=2.5, label=lab, alpha=0.9)
    ax.axhline(meta["beta_bar_dist"], color=C_BAD, ls="--", lw=1.2,
               label=r"$\|\bar\beta-\beta^\ast\|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.004, 1.6)
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel(r"$\|\hat\beta-\beta^\ast\|_2$")
    ax.set_title(r"(a) $\tau=1/2$: the smoothed error floors", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2, handlelength=1.9)

    ax = axes[1]
    for tau, c in zip(taus, (C_BAD, C_MID, C_MAIN)):
        v = np.array([res[f"tau{tau}_n{n}"]["conv"] for n in ns])
        vb = np.array([res[f"tau{tau}_n{n}"]["conv_to_bar"] for n in ns])
        ax.plot(ns, v, marker="o", color=c, lw=1.5, ms=4,
                label=fr"$\tau={tau:g}$: to $\beta^\ast$")
        ax.plot(ns, vb, marker="x", color=c, lw=1.0, ls=":",
                label=fr"$\tau={tau:g}$: to $\bar\beta$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.004, 1.6)
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel(r"$\ell_2$ error")
    ax.set_title(r"(b) distance to $\beta^\ast$ is flat, to $\bar\beta$ decays", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left", handlelength=1.9)

    plt.tight_layout()
    out = save_fig(fig, "fig2_scaling")
    print(f"[figure] {out}")


# ------------------------------------------------------------------ Figure: high-dim
def fig_highdim():
    with io.open("simulations/routeA_results_E3.json", encoding="utf-8") as fh:
        res = json.load(fh)
    keys = sorted([k for k in res if k != "_meta"], key=lambda z: float(z.split("sigma_u")[1]))
    su = [float(k.split("sigma_u")[1]) for k in keys]
    methods = [("conv_paper", "smoothed\n(published)", C_BAD),
               ("naive", "naive", "#7f8c8d"),
               ("rc_int", "RC (calibrated)", C_GOOD),
               ("oracle", "oracle", C_MAIN)]
    x = np.arange(len(keys))
    width = 0.19
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for i, (m, lab, c) in enumerate(methods):
        med = [res[k][m + "_median"] for k in keys]
        lo = [res[k][m + "_median"] - res[k][m + "_q25"] for k in keys]
        hi = [res[k][m + "_q75"] - res[k][m + "_median"] for k in keys]
        ax.bar(x + (i - 1.5) * width, med, width, yerr=[lo, hi], capsize=2.0,
               color=c, label=lab, alpha=0.9)
    for j, k in enumerate(keys):
        ax.hlines(res[k]["beta_bar_dist"], x[j] - 0.5, x[j] + 0.5, color="black",
                  ls="--", lw=1.3,
                  label=(r"$\|\bar\beta-\beta^\ast\|$" if j == 0 else None))
        ax.hlines(res[k]["conv_paper_to_bar_median"], x[j] - 0.5, x[j] + 0.5,
                  color=C_BAD, ls=":", lw=1.3,
                  label=(r"smoothed $\to\bar\beta$" if j == 0 else None))
    ax.set_xticks(x)
    ax.set_xticklabels([fr"$\sigma_u={s:g}$" for s in su])
    ax.set_ylabel(r"$\|\hat\beta-\beta^\ast\|_2$ (median, IQR)")
    ax.set_title("High-dimensional setting: $n=400$, $p=500$, $s=10$, $\\tau=1/2$",
                 fontsize=9)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", handlelength=1.9)
    plt.tight_layout()
    out = save_fig(fig, "fig3_highdim")
    print(f"[figure] {out}")


if __name__ == "__main__":
    fig_losses()
    fig_scaling()
    fig_highdim()
    print("all figures written")

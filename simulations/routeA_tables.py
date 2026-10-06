"""
Generate the LaTeX tables of paper/main.tex directly from the saved JSON results,
so that text, tables and figures cannot drift apart.

Run AFTER simulations/routeA_targets.py and simulations/routeA_experiments.py.
"""

import io
import json
import os

import numpy as np

PAPER_TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "tables")
os.makedirs(PAPER_TABLES, exist_ok=True)


def w(name, text):
    path = os.path.join(PAPER_TABLES, name)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[written] paper/tables/{name}")


# ---------------------------------------------------------------- Table: targets
def table_targets():
    with io.open("simulations/routeA_targets.json", encoding="utf-8") as fh:
        rows = json.load(fh)
    su_list = sorted({r["sigma_u"] for r in rows})
    tau_list = sorted({r["tau"] for r in rows})

    # ---- Panel A: estimators built on the RAW W (no conditioning) -----------------
    body = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
            r"\caption{Exact population targets of the estimators fitted on the raw $W$ "
            r"(scalar case, $X\sim\cN(0,1)$, $N=2\times10^5$; each entry is the minimiser "
            r"of a population objective, computed without an optimiser). "
            r"$\betas=2$; $\bbar=\Sigma_x(\Sigma_x+\Sigma_u)^{-1}\betas$ is the "
            r"attenuation target. By Proposition~\ref{prop:exact-target} the population "
            r"minimiser is exactly $\bbar$ for naive quantile regression and for every "
            r"fixed-scale smoothed loss, and exactly "
            r"$(\Sigma_x+\Sigma_u+M)^{-1}\Sigma_x\betas$ for the adaptive scale with "
            r"kernel $M$; the tabulated Monte-Carlo values agree with those exact values "
            r"to within $0.006$, which is also the size of their deviation from $\bbar$ in "
            r"the posterior-kernel column (Proposition~\ref{prop:posterior-adaptive}). "
            r"Column 5 uses the pilot scale $\sigmah$ from a calibrated Stage~1 "
            r"(idealised: the true $\Sigma_x$ is used, which favours the smoothed "
            r"method).}",
            r"\label{tab:targets}",
            r"\begin{tabular}{llcccccc}", r"\toprule",
            r"$\sigma_u$ & $\tau$ & $\bbar$ & smoothed & smoothed & smoothed & smoothed & naive \\",
            r" & & & (fixed $\sigma$) & (pilot $\sigmah$) & (adaptive) & (post.\ kernel) & \\",
            r"\midrule"]
    for su in su_list:
        for tau in tau_list:
            r = next(x for x in rows if x["sigma_u"] == su and x["tau"] == tau)
            body.append(
                f"{su:g} & {tau:g} & {r['beta_bar']:.3f} & "
                f"{r['conv_fixed_betastar']:.3f} & {r['conv_fixed_stage1']:.3f} & "
                f"{r['conv_adaptive']:.3f} & {r['conv_posterior_kernel']:.3f} & "
                f"{r['naive_qr']:.3f} \\\\")
        body.append(r"\midrule")
    body = body[:-1]
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_targets.tex", "\n".join(body) + "\n")

    # ---- Panel B: calibration-based estimators -----------------------------------
    body = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
            r"\caption{Population targets of the calibration-based estimators (same "
            r"designs as Table~\ref{tab:targets}). ``RC'' regresses $Y$ on "
            r"$\muv=\Ccal W$; ``RC$+$int'' adds an intercept. Both recover $\betas$ "
            r"within Monte-Carlo error, and the fitted intercept agrees with the "
            r"predicted quantile shift $\delta_\tau$ of Theorem~\ref{thm:calib}. For "
            r"reference the exact value $\betas=2$ and the oracle fit on the clean $X$ "
            r"are shown.}",
            r"\label{tab:targets_calib}",
            r"\begin{tabular}{lcccccc}", r"\toprule",
            r"$\sigma_u$ & $\tau$ & RC & RC$+$int & fitted intercept & "
            r"predicted $\delta_\tau$ & oracle \\", r"\midrule"]
    for su in su_list:
        for tau in tau_list:
            r = next(x for x in rows if x["sigma_u"] == su and x["tau"] == tau)
            body.append(
                f"{su:g} & {tau:g} & {r['rc_qr']:.3f} & {r['rc_qr_int']:.3f} & "
                f"{r['rc_qr_int_intercept']:+.4f} & {r['delta']:+.4f} & "
                f"{r['oracle_qr']:.3f} \\\\")
        body.append(r"\midrule")
    body = body[:-1]
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_targets_calib.tex", "\n".join(body) + "\n")


# ---------------------------------------------------------------- Table: score
def table_score():
    with io.open("simulations/routeA_results_E4.json", encoding="utf-8") as fh:
        res = json.load(fh)
    body = [r"\begin{table}[ht]", r"\centering", r"\small",
            r"\caption{Population score at the truth, $\E[W_{i1}\psit(r_i^*,\sigma^*)]$ "
            r"(smoothed loss, with $\sigma=\sigma^*$) and "
            r"$\E[{\muv}_{i1}(\tau-\ind{e_i<\delta_\tau})]$ (calibrated, check loss), "
            r"estimated from $N=4\times10^5$ observations. "
            r"The last column is the analytic prediction $-(\Sigma_u\betas)_1 d(\sigma^*)$ "
            r"of Theorem~\ref{thm:score}. The smoothed score is non-zero with "
            r"$t$-statistics in the tens to hundreds; the calibrated score is "
            r"indistinguishable from zero.}",
            r"\label{tab:score}",
            r"\begin{tabular}{ccccccc}", r"\toprule",
            r"$\sigma_u$ & $\tau$ & $\E[W_{i1}\psit]$ & s.e. & $t$ & analytic & "
            r"$\E[{\muv}_{i1}\psi]$ ($t$) \\", r"\midrule"]
    for k in sorted(res, key=lambda z: (res[z]["sigma_u"], res[z]["tau"])):
        r = res[k]
        body.append(f"{r['sigma_u']:g} & {r['tau']:g} & {r['E_Wpsi']:+.5f} & "
                    f"{r['se_Wpsi']:.5f} & {r['t_Wpsi']:+.1f} & {r['pred_E_Wpsi']:+.5f} & "
                    f"{r['E_muW_psi']:+.5f} ({r['t_muW_psi']:+.1f}) \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_score.tex", "\n".join(body) + "\n")


# ---------------------------------------------------------------- Table: scaling
def table_scaling():
    with io.open("simulations/routeA_results_E2.json", encoding="utf-8") as fh:
        res = json.load(fh)
    meta = res["_meta"]
    body = [r"\begin{table}[ht]", r"\centering", r"\small",
            r"\caption{Low-dimensional scaling ($p=10$, $s=5$, $\sigma_u=0.5$, "
            r"unpenalised, $R=50$ replications; mean $\pm$ standard error of "
            r"$\norm{\hat\beta-\betas}_2$). The smoothed estimator is flat in $n$ at the "
            r"attenuation distance $\norm{\bbar-\betas}$, while its distance to $\bbar$ "
            r"decays; the calibrated estimator tracks the oracle up to the efficiency "
            r"factor of the remark following Theorem~\ref{thm:lowdim}.}",
            r"\label{tab:scaling}",
            r"\begin{tabular}{cccccccc}", r"\toprule",
            r"$\tau$ & $n$ & smoothed & smoothed$\,\to\bbar$ & naive & RC & RC$+$int & "
            r"oracle \\", r"\midrule"]
    keys = [k for k in res if k != "_meta"]
    keys.sort(key=lambda z: (float(z.split("_")[0][3:]), int(z.split("_n")[1])))
    for key in keys:
        r = res[key]
        tau = key.split("_")[0][3:]
        n = key.split("_n")[1]
        body.append(f"{tau} & {n} & {r['conv']:.3f}$\\pm${r['conv_se']:.3f} & "
                    f"{r['conv_to_bar']:.3f} & {r['naive']:.3f} & {r['rc']:.3f} & "
                    f"{r['rc_int']:.3f} & {r['oracle']:.3f} \\\\")
    body += [r"\midrule",
             r"\multicolumn{8}{l}{$\norm{\bbar-\betas}_2 = "
             f"{meta['beta_bar_dist']:.3f}" + r"$ for every row.}\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_scaling.tex", "\n".join(body) + "\n")


# ---------------------------------------------------------------- Table: high-dim
def table_highdim():
    with io.open("simulations/routeA_results_E3.json", encoding="utf-8") as fh:
        res = json.load(fh)
    body = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{High-dimensional comparison ($n=400$, $p=500$, $s=10$, $\tau=1/2$, "
            r"$\lambda=0.02$, $R=30$, fixed $\betas$): median $\ell_2$ error against "
            r"$\betas$, IQR in parentheses. ``smoothed'' is the two-stage smoothed "
            r"estimator run through its own published code; ``naive'', ``RC'' and "
            r"``oracle'' are exact linear-programming quantile regressions on $W$, on the "
            r"calibrated $\muv=\Ccal W$, and on the clean $X$. The error is "
            r"variance-dominated here, so the smoothed estimator is competitive in "
            r"$\ell_2$ although it targets $\bbar$; its distance to $\bbar$, reported "
            r"beneath each block, stays bounded away from zero.}",
            r"\label{tab:highdim}",
            r"\begin{tabular}{lcccc}", r"\toprule",
            r"$\sigma_u$ & smoothed & naive & RC & oracle \\", r"\midrule"]
    keys = sorted([k for k in res if k != "_meta"],
                  key=lambda z: float(z.split("sigma_u")[1]))
    for k in keys:
        r = res[k]
        su = k.split("sigma_u")[1]

        def fmt(m, q25, q75):
            return f"{r[m]:.3f} ({r[q25]:.3f}, {r[q75]:.3f})"

        body.append(
            f"{su} & {fmt('conv_paper_median','conv_paper_q25','conv_paper_q75')} & "
            f"{fmt('naive_median','naive_q25','naive_q75')} & "
            f"{fmt('rc_int_median','rc_int_q25','rc_int_q75')} & "
            f"{fmt('oracle_median','oracle_q25','oracle_q75')} \\\\")
        body.append(r"\multicolumn{5}{l}{\quad smoothed: "
                    f"$\\norm{{\\bbar-\\betas}}_2={r['beta_bar_dist']:.3f}$, "
                    f"$\\norm{{\\hat\\beta-\\bbar}}_2={r['conv_paper_to_bar_median']:.3f}$"
                    r"} \\")
        body.append(r"\midrule")
    body = body[:-1]
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_highdim.tex", "\n".join(body) + "\n")


def table_coverage():
    with io.open("simulations/routeA_results_E5.json", encoding="utf-8") as fh:
        res = json.load(fh)
    meta = res["_meta"]
    keys = [k for k in res if k != "_meta"]
    keys.sort(key=lambda z: (float(z.split("_")[0][3:]), int(z.split("_n")[1])))
    body = [r"\begin{table}[ht]", r"\centering", r"\small",
            r"\caption{Coverage of nominal $90\%$ and $95\%$ Wald intervals "
            f"($\\sigma_u={meta['sigma_u']:g}$, $\\betas={meta['beta_star']:g}$, "
            f"$\\bbar={meta['beta_bar']:.1f}$, {meta['nrep']} replications). "
            r"Intervals use the sandwich covariance of Theorem~\ref{thm:lowdim} for the "
            r"calibrated estimator and the corresponding smoothed-score covariance for "
            r"the smoothed estimator. The calibrated intervals cover $\betas$ at close to "
            r"the nominal rate; the smoothed intervals are equally well calibrated, but "
            r"for $\bbar$, and cover $\betas$ essentially never.}",
            r"\label{tab:coverage}",
            r"\begin{tabular}{cccccc}", r"\toprule",
            r"& & \multicolumn{2}{c}{calibrated, target $\betas$} & "
            r"\multicolumn{2}{c}{smoothed, $95\%$} \\",
            r"$\tau$ & $n$ & $90\%$ & $95\%$ & covers $\betas$ & covers $\bbar$ \\",
            r"\midrule"]
    for k in keys:
        r = res[k]
        tau = k.split("_")[0][3:]
        n = k.split("_n")[1]
        body.append(f"{tau} & {n} & {r['cov90_cal']:.3f} & {r['cov95_cal']:.3f} & "
                    f"{r['cov95_smoothed_to_star']:.3f} & "
                    f"{r['cov95_smoothed_to_bar']:.3f} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_coverage.tex", "\n".join(body) + "\n")


def table_hdc():
    with io.open("simulations/hdc_results.json", encoding="utf-8") as fh:
        res = json.load(fh)
    meta = res["_meta"]
    keys = sorted([k for k in res if k != "_meta"], key=lambda z: float(z.split("sigma_u")[1]))
    body = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Estimated calibration in high dimensions "
            f"($n={meta['n']}$, $p={meta['p']}$, $s={meta['s']}$, $\\tau={meta['tau']:g}$, "
            f"$R={meta['R']}$ replications, banded $\\Sigma_x$, $\\hat\\Sigma_x$ from "
            r"thresholded Ledoit--Wolf shrinkage). $\hat\Ccal$ is computed from the "
            r"estimated $\hat\Sigma_x$; the oracle column uses the true $\Ccal$. The "
            r"sup-norm scores are evaluated at the truth, "
            r"$\|n^{-1}\sum_i{\hat\muv}_i\psi_i\|_\infty$, with the theoretical scale "
            "$\\sqrt{\\log p/n} = " + f"{meta['sqrt_logp_over_n']:.4f}" + "$.}",
            r"\label{tab:hdc}",
            r"\begin{tabular}{lccccc}", r"\toprule",
            r"$\sigma_u$ & $\|\hat\Ccal-\Ccal\|_2$ & naive & oracle $\Ccal$ & "
            r"estimated $\hat\Ccal$ & sup-norm score (oracle / estimated) \\",
            r"\midrule"]
    for k in keys:
        r = res[k]
        su = k.split("sigma_u")[1]
        body.append(
            f"{su} & {r['norm_C_err_mean']:.4f} & {r['naive_mean']:.3f} & "
            f"{r['oracle_cal_mean']:.3f} & {r['plugin_cal_mean']:.3f} & "
            f"{r['score_oracle_mean']:.4f} / {r['score_plugin_mean']:.4f} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_hdc.tex", "\n".join(body) + "\n")


def table_gls():
    """Theoretical GLS gain across quantile levels; values from hdc_gls_theory.py."""
    grids = {"3 levels": [0.4, 0.5, 0.6],
             "7 levels": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
             "9 levels": [round(0.1 * k, 3) for k in range(1, 10)]}
    dists = {"Laplace": "laplace", "Gaussian": "norm", "Uniform": "uniform"}
    import numpy as np
    from scipy.stats import norm, laplace, uniform
    from scipy.stats import rv_continuous
    backend = {"laplace": laplace(0, 1), "norm": norm(0, 1),
               "uniform": uniform(-1, 1)}

    def ratio(taus, dist):
        K = len(taus)
        q = dist.ppf(taus)
        f = dist.pdf(q)
        C = np.array([[(min(taus[k], taus[l]) - taus[k] * taus[l]) / (f[k] * f[l])
                       for l in range(K)] for k in range(K)])
        Ci = np.linalg.inv(C)
        one = np.ones(K)
        k0 = int(np.argmin(np.abs(np.array(taus) - 0.5)))    # reference = median
        return (1.0 / (one @ Ci @ one)) / C[k0, k0]

    body = [r"\begin{table}[ht]", r"\centering", r"\small",
            r"\caption{Theoretical gain from combining quantile levels in the "
            r"location-shift model: ratio of the GLS variance \eqref{eq:gls} to the "
            r"variance of the single level $\tau=1/2$, computed from the closed form. "
            r"A value of $1$ means combining levels recovers nothing.}",
            r"\label{tab:gls}",
            r"\begin{tabular}{lccc}", r"\toprule",
            r"error distribution & 3 levels & 7 levels & 9 levels \\", r"\midrule"]
    for name, key in dists.items():
        cells = " & ".join(f"{ratio(g, backend[key]):.4f}" for g in grids.values())
        body.append(f"{name} & {cells} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_gls.tex", "\n".join(body) + "\n")


def table_debias():
    """Two panels: p < n (exact plug-in Hessian) and p > n (nodewise inverse)."""
    with io.open("simulations/hdc_debias.json", encoding="utf-8") as fh:
        res = json.load(fh)
    with io.open("simulations/hdc_pgtn.json", encoding="utf-8") as fh:
        pg = json.load(fh)
    meta = res["_meta"]
    pg_R = pg["_meta"]["R"]
    keys = [k for k in res if k != "_meta"]
    keys.sort(key=lambda z: (float(z.split("_")[0][2:]), -float(z.split("lam")[1])))
    caption = (r"\caption{Debiased inference for the calibrated estimator, nominal $95\%$ "
               r"intervals. Panel A ($n=400>p=40$, $R=__RA__$): three intervals for the "
               r"calibrated estimator, all counted on the $s=5$ signal coordinates: the "
               r"penalised fit with its own sandwich interval (``$\ell_1$''), the one-step "
               r"debiased interval \eqref{eq:debias} (``deb.''), and the interval obtained by "
               r"refitting unpenalised quantile regression on the selected support and then "
               r"debiasing (``refit$+$deb.''). The null coverage and the mean length in each "
               r"row belong to the refit$+$deb.\ interval, as does the support column, which "
               r"gives the mean number of coordinates selected, with the fraction of "
               r"replications in which the selected set is exactly the true support in "
               r"parentheses. Panel B ($n=200<p=250$, $R=__RB__$): the plug-in Hessian is "
               r"singular, so $\hat\Theta$ is built nodewise and the penalised estimator, "
               r"which admits no usable interval, is omitted; the support column then also "
               r"reports the mean number of missed signal coordinates.}"
               ).replace("__RA__", str(meta["R"])).replace("__RB__", str(pg_R))
    body = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            caption,
            r"\label{tab:debias}",
            r"\begin{tabular}{lccccccc}", r"\toprule",
            r"& & \multicolumn{3}{c}{coverage, signal} & \multicolumn{2}{c}{refit$+$deb.} & \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
            r"setting & $\lambda$ & $\ell_1$ & deb. & refit$+$deb. & null & length & support \\",
            r"\midrule",
            r"\multicolumn{8}{l}{\emph{Panel A: $n=400$, $p=40$, $R=%d$, $\lambda_0=%.4f$}}\\" % (
                meta["R"], meta["lam0"])]
    for k in keys:
        r = res[k]
        su = k.split("_")[0][2:]
        fac = float(k.split("lam")[1])
        body.append(
            f"$\\sigma_u={su}$ & {fac:g}$\\times\\lambda_0$ & "
            f"{r['cov_sup_l1']['mean']:.3f} & "
            f"{r['cov_sup_deb']['mean']:.3f} & "
            f"\\textbf{{{r['cov_sup_rd']['mean']:.3f}}} & "
            f"{r['cov_null_rd']['mean']:.3f} & {r['len_rd']['mean']:.3f} & "
            f"{r['nsup']['mean']:.2f} ({r['exact_support']['mean']:.3f}) \\\\")
    body.append(r"\midrule")
    pm = pg["_meta"]
    body.append(r"\multicolumn{8}{l}{\emph{Panel B: $n=%d$, $p=%d$, $s=%d$, $R=%d$, "
                r"$\sigma_u=%g$, $\lambda_0=%.4f$; nodewise inverse}}\\" % (
                    pm["n"], pm["p"], pm["s"], pm["R"], pm["sigma_u"], pm["lam0"]))
    for k in sorted([z for z in pg if z != "_meta"],
                    key=lambda z: -float(z[3:])):
        r = pg[k]
        fac = float(k[3:])
        body.append(
            f"refit$+$deb. & {fac:g}$\\times\\lambda_0$ & "
            f"\\multicolumn{{3}}{{c}}{{\\textbf{{{r['cov_sup']['mean']:.3f}}}"
            f"$\\pm${r['cov_sup']['se']:.3f}}} & "
            f"{r['cov_null']['mean']:.3f} & {r['len_sup']['mean']:.3f} & "
            f"{r['nsup']['mean']:.2f} ({r['exact_support']['mean']:.2f}, "
            f"{r['n_under']['mean']:.2f}) \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_debias.tex", "\n".join(body) + "\n")


def table_plugin():
    """Known calibration against the method-of-moments plug-in (Proposition 6.2)."""
    with io.open("review/verify_plugin_calibration_bias.json", encoding="utf-8") as fh:
        res = json.load(fh)
    with io.open("review/verify_plugin_bias_scaling.json", encoding="utf-8") as fh:
        sc = json.load(fh)
    R = res["R"]
    Z95 = 1.959963984540054
    caption = (r"\caption{What estimating the calibration costs, nominal $95\%$ intervals: "
               r"coverage of the refitted and debiased estimator on the $s=5$ signal "
               r"coordinates, with the calibration known (as everywhere else in this paper) "
               r"and with the method-of-moments plug-in $\hat\Ccal=I-\Sigma_uS_W^{-1}$. The "
               r"bias is the mean of $\hat b_j-\beta^*_j$ over signal coordinates and "
               r"replications, the length is the mean interval length, and "
               r"$r_n=\max_j\|(\hat\Ccal-\Ccal)_{j\cdot}\|_1$ is the row-wise calibration "
               r"error. Panel A varies $\sigma_u$ at $n=400$, $p=40$, $R=200$; Panel B varies "
               r"$p$ at the fixed ratio $n=4p$ ($\sigma_u=0.5$, $R=100$), where the "
               r"known-calibration intervals stay at the nominal level while the plug-in "
               r"intervals do not. Proposition~\ref{prop:plugin} says why the plug-in cannot "
               r"close this gap: its row-wise error is of the order of the sample "
               r"covariance's, which is the order at which the bias stops being negligible "
               r"relative to the standard error.}").replace("__R__", str(R))
    body = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            caption,
            r"\label{tab:plugin}",
            r"\begin{tabular}{llccccc}", r"\toprule",
            r"& & & coverage & & mean & \\",
            r"setting & calibration & coverage & s.e. & bias & length & $r_n$ \\",
            r"\midrule",
            r"\multicolumn{7}{l}{\emph{Panel A: $n=400$, $p=40$, $R=%d$}}\\" % R]
    for key in sorted(res["res"]):
        r = res["res"][key]
        su = key.split("=")[1]
        for tag, lab in (("known", r"known $\Ccal$"), ("plug", r"plug-in $\hat\Ccal$")):
            cov = r[f"cov_{tag}"]["mean"]
            se = r[f"cov_{tag}"]["se"]
            bias = r[f"bias_{tag}"]["mean"]
            length = 2 * Z95 * r[f"se_{tag}"]["mean"]
            rn_s = "--" if tag == "known" else f"{r['r_n']['mean']:.3f}"
            body.append(f"$\\sigma_u={su}$ & {lab} & {cov:.3f} & {se:.3f} & {bias:+.3f} & "
                        f"{length:.3f} & {rn_s} \\\\")
    body.append(r"\midrule")
    body.append(r"\multicolumn{7}{l}{\emph{Panel B: $n=4p$, $s=5$, $\sigma_u=0.5$, $R=%d$}}\\"
                % sc["R"])
    for r in sc["rows"]:
        if r["p"] < 100:          # below this the penalised fit itself is unstable: n <= 200
            continue
        for tag, lab in (("known", r"known $\Ccal$"), ("plug", r"plug-in $\hat\Ccal$")):
            cov = r[tag]["cov"]
            bias = r[tag]["bias"]
            length = 2 * Z95 * r[tag]["se"]
            rn_s = "--" if tag == "known" else f"{r['r_n']:.3f}"
            body.append(f"$p={r['p']}$, $n={r['n']}$ & {lab} & {cov:.3f} & -- & {bias:+.3f} "
                        f"& {length:.3f} & {rn_s} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_plugin.tex", "\n".join(body) + "\n")



def table_nhanes(max_rows=10):
    with io.open("simulations/nhanes_results.json", encoding="utf-8") as fh:
        res = json.load(fh)
    m = res["meta"]
    rows = sorted(res["rows"], key=lambda r: -abs(r["calibrated"]))[:max_rows]
    body = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Real data: NHANES 2017--2018, $n=%d$ adults with two reliable "
            r"24-hour dietary recalls, $p=%d$ nutrients after removing near-duplicates, "
            r"$\tau=1/2$. Both $\Sigma_x$ and $\Sigma_u$ are estimated from the replicate "
            r"pair, with no synthetic noise; reliability is "
            r"$(\Sigma_x)_{jj}/((\Sigma_x)_{jj}+(\Sigma_u)_{jj})$. ``naive'' uses the "
            r"day-1 recall and ``calibrated'' uses $\hat\Ccal W$ with an intercept and the "
            r"debiased interval of \S\ref{sec:debiasing}; the largest $%d$ coefficients in "
            r"absolute value are shown.}" % (m["n"], m["p"], max_rows),            r"\label{tab:nhanes}",
            r"\begin{tabular}{lcccr}", r"\toprule",
            r"nutrient & reliability & naive & calibrated & $95\%$ interval \\", r"\midrule"]
    for r in rows:
        body.append(f"{r['nutrient']} & {r['reliability']:.3f} & {r['naive']:+.3f} & "
                    f"{r['calibrated']:+.3f} & "
                    f"$[{r['ci_lo']:+.2f},\\, {r['ci_hi']:+.2f}]$"
                    + (r"$^{\dagger}$" if r["significant"] else "") + r" \\")
    body += [r"\midrule",
             r"\multicolumn{5}{l}{$^{\dagger}$interval excludes zero.}\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_nhanes.tex", "\n".join(body) + "\n")


if __name__ == "__main__":
    table_targets()
    table_score()
    table_scaling()
    table_highdim()
    table_coverage()
    table_hdc()
    table_gls()
    table_debias()
    table_plugin()
    table_nhanes()
    print("all tables written")

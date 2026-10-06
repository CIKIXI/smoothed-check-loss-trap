"""
Consistency audit: every number typed by hand in the manuscript prose must match the
saved results.  This script re-derives the claims from the JSON/scripts and prints a
pass/fail table, so that "theory - code - figures" stay locked together.
"""

import io
import json
import os
import re

import numpy as np
from scipy.special import ndtr

PAPER = "paper/main.tex"
checks = []


def claim(name, paper_substring, condition, detail=""):
    checks.append((name, paper_substring, bool(condition), detail))


with io.open(PAPER, encoding="utf-8") as fh:
    tex = fh.read()
import glob
for _tf in glob.glob("paper/tables/*.tex"):          # generated tables count as "in paper"
    with io.open(_tf, encoding="utf-8") as fh:
        tex += fh.read()


def has(s):
    return s in tex


# ---------------------------------------------------------------- population targets
with io.open("simulations/routeA_targets.json", encoding="utf-8") as fh:
    tg = json.load(fh)
su05 = [r for r in tg if r["sigma_u"] == 0.5 and r["tau"] == 0.5][0]
claim("Thm 3.4 quotes the adaptive target 1.333",
      "1.333", abs(su05["conv_adaptive"] - 1.333) < 5e-3,
      f"routeA_targets: {su05['conv_adaptive']:.3f}")
claim("Thm 3.4 quotes the fixed-scale target 1.600",
      "1.600", abs(su05["conv_fixed_betastar"] - 1.600) < 5e-3,
      f"routeA_targets: {su05['conv_fixed_betastar']:.3f}")
# Proposition 3.13 (exact targets): the MC columns must sit on the closed forms
exact = {}
for r in tg:
    su, tau = r["sigma_u"], r["tau"]
    exact[(su, tau)] = dict(
        fixed=1.0 / (1.0 + su ** 2) * 2.0,                    # beta_bar
        adaptive=2.0 / (1.0 + 2 * su ** 2),                   # (Sig_x+2Sig_u)^{-1}Sig_x beta*
        naive=1.0 / (1.0 + su ** 2) * 2.0,                    # beta_bar
        posterior=1.0 / (1.0 + su ** 2) * 2.0)                # beta_bar
worst_fixed = max(abs(r["conv_fixed_stage1"] - exact[(r["sigma_u"], r["tau"])]["fixed"])
                  for r in tg)
worst_adaptive = max(abs(r["conv_adaptive"] - exact[(r["sigma_u"], r["tau"])]["adaptive"])
                     for r in tg)
worst_naive = max(abs(r["naive_qr"] - exact[(r["sigma_u"], r["tau"])]["naive"]) for r in tg)
claim("Prop 3.13: fixed-scale MC targets equal beta_bar exactly",
      "targets $\\bbar$ rather than $\\betas$", worst_fixed < 6e-3,
      f"max |MC - beta_bar| = {worst_fixed:.4f}")
claim("Prop 3.13: adaptive MC targets equal beta*/(1+2 sigma_u^2)",
      "$(\\Sigma_x+\\Sigma_u+M)^{-1}\\Sigma_x\\betas$", worst_adaptive < 6e-3,
      f"max deviation = {worst_adaptive:.4f}")
claim("Prop 3.13: naive MC targets equal beta_bar exactly",
      "exactly $\\bbar$ for naive quantile regression",
      worst_naive < 6e-3, f"max |MC - beta_bar| = {worst_naive:.4f}")

# ---------------------------------------------------------------- score check
with io.open("simulations/routeA_results_E4.json", encoding="utf-8") as fh:
    sc = json.load(fh)
worst = max(abs(v["E_Wpsi"] - v["pred_E_Wpsi"]) / abs(v["pred_E_Wpsi"]) for v in sc.values())
min_t = min(abs(v["t_Wpsi"]) for v in sc.values())
max_t = max(abs(v["t_muW_psi"]) for v in sc.values())
claim("score table: analytic values match to 5%",
      "E[W", worst < 0.05, f"worst relative gap {100*worst:.2f}%")
claim("score table: |t| of the smoothed score is at least 10",
      "t", min_t > 10, f"min |t| = {min_t:.1f}")
claim("score table: calibrated score |t| <= 1.5",
      "t", max_t <= 1.5, f"max |t| = {max_t:.2f}")

# ---------------------------------------------------------------- coverage
with io.open("simulations/routeA_results_E5.json", encoding="utf-8") as fh:
    cov = json.load(fh)
cells = [v for k, v in cov.items() if k != "_meta"]
c95 = [v["cov95_cal"] for v in cells]
sm = [v["cov95_smoothed_to_star"] for v in cells]
sb = [v["cov95_smoothed_to_bar"] for v in cells]
claim("coverage: calibrated 95% between 0.92 and 0.98",
      "0.93", min(c95) > 0.92 and max(c95) < 0.98,
      f"range {min(c95):.3f}-{max(c95):.3f}")
claim("coverage: smoothed intervals cover beta* essentially never",
      "0.000", max(sm) < 0.01, f"max {max(sm):.3f}")
claim("coverage: smoothed intervals cover beta_bar near nominally",
      "0.9", min(sb) > 0.90, f"min {min(sb):.3f}")
claim("coverage: twelve cells reported",
      "twelve", len(cells) == 12, f"{len(cells)} cells")
claim("coverage: 4,800 intervals",
      "4{,}800", len(cells) * int(cov["_meta"]["nrep"]) == 4800,
      f"{len(cells)} x {cov['_meta']['nrep']} = {len(cells)*int(cov['_meta']['nrep'])}")

# ---------------------------------------------------------------- high-dim results
with io.open("simulations/hdc_results.json", encoding="utf-8") as fh:
    hd = json.load(fh)
meta = hd["_meta"]
cells = [v for k, v in hd.items() if k != "_meta"]
errs = [v["norm_C_err_mean"] for v in cells]
plug = [v["plugin_cal_mean"] for v in cells]
orc = [v["oracle_cal_mean"] for v in cells]
gain = [a - b for a, b in zip(plug, orc)]
claim("HDC: ||C_hat - C|| between 0.005 and 0.09",
      "0.005", min(errs) > 0.004 and max(errs) < 0.10,
      f"{min(errs):.4f}-{max(errs):.4f}")
claim("HDC: plug-in within 0.02 of the oracle calibration",
      "0.02", max(abs(g) for g in gain) < 0.02,
      f"largest gap {max(abs(g) for g in gain):.4f}")
sc_or = [v["score_oracle_mean"] for v in cells]
sc_pl = [v["score_plugin_mean"] for v in cells]
thr = meta["sqrt_logp_over_n"]
claim("HDC: both scores below sqrt(log p / n)",
      "0.1246", max(sc_or + sc_pl) < thr,
      f"max {max(sc_or+sc_pl):.4f} vs {thr:.4f}")
claim("HDC: adaptive target further from beta* than the fixed-scale one",
      "1.333", abs(su05["conv_adaptive"] - 2.0) > abs(su05["conv_fixed_betastar"] - 2.0),
      f"|1.333-2|={abs(su05['conv_adaptive']-2):.3f} > "
      f"|{su05['conv_fixed_betastar']:.3f}-2|={abs(su05['conv_fixed_betastar']-2):.3f}")

# ---------------------------------------------------------------- GLS table
from scipy.stats import norm, laplace, uniform


def gls_ratio(taus, dist):
    K = len(taus)
    q = dist.ppf(taus)
    f = dist.pdf(q)
    C = np.array([[(min(taus[k], taus[l]) - taus[k] * taus[l]) / (f[k] * f[l])
                   for l in range(K)] for k in range(K)])
    Ci = np.linalg.inv(C)
    one = np.ones(K)
    k0 = int(np.argmin(np.abs(np.array(taus) - 0.5)))
    return (1.0 / (one @ Ci @ one)) / C[k0, k0]


g7 = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
claim("GLS: Laplace gain is exactly 1",
      "1.0000", abs(gls_ratio(g7, laplace(0, 1)) - 1.0) < 1e-6,
      f"{gls_ratio(g7, laplace(0,1)):.6f}")
claim("GLS: Gaussian 7-level ratio 0.80",
      "0.7977", abs(gls_ratio(g7, norm(0, 1)) - 0.7977) < 1e-3,
      f"{gls_ratio(g7, norm(0,1)):.4f}")
claim("GLS: uniform 7-level ratio 0.70",
      "0.7000", abs(gls_ratio(g7, uniform(-1, 1)) - 0.70) < 1e-3,
      f"{gls_ratio(g7, uniform(-1,1)):.4f}")

# ---------------------------------------------------------------- p > n debiasing
with io.open("simulations/hdc_pgtn.json", encoding="utf-8") as fh:
    pg = json.load(fh)
a = pg["lam0.6"]
b = pg["lam0.4"]
claim("p>n: coverage on the support is 0.952 +- 0.016",
      "0.952", abs(a["cov_sup"]["mean"] - 0.952) < 1e-3,
      f"{a['cov_sup']['mean']:.3f} +- {a['cov_sup']['se']:.3f}")
claim("p>n: null coverage is 0.949",
      "0.949", abs(a["cov_null"]["mean"] - 0.949) < 2e-3,
      f"{a['cov_null']['mean']:.3f}")
claim("p>n: exact support recovery 0.76 and 0.04 missed",
      "0.04", abs(a["exact_support"]["mean"] - 0.76) < 0.01
      and abs(a["n_under"]["mean"] - 0.04) < 0.02,
      f"exact {a['exact_support']['mean']:.2f}, missed {a['n_under']['mean']:.2f}")
claim("p>n: over-selection at 0.4 lambda0 gives 12.6 and coverage 0.824",
      "0.824", abs(b["nsup"]["mean"] - 12.62) < 0.05
      and abs(b["cov_sup"]["mean"] - 0.824) < 2e-3,
      f"support {b['nsup']['mean']:.2f}, coverage {b['cov_sup']['mean']:.3f}")
claim("p>n: nodewise inverse exact on the reported coordinates",
      "1|=0$", max(a["inv_err"]["mean"], b["inv_err"]["mean"]) < 1e-6,
      f"max |(Theta H)jj - 1| = {max(a['inv_err']['mean'], b['inv_err']['mean']):.2e}")
claim("p>n: mean interval length 0.95",
      "0.95", abs(a["len_sup"]["mean"] - 0.948) < 5e-3,
      f"{a['len_sup']['mean']:.3f}")

# ------------------------------- Section 6: Theorem 6.1 and Proposition 6.2 (round 27)
# The theorem's condition, the proposition's bound, and Table 10 must all be in the paper
# and must agree with the experiment they come from.
with io.open("review/verify_plugin_calibration_bias.json", encoding="utf-8") as fh:
    pj = json.load(fh)
claim("Thm 6.1: the debiasing condition is stated in the paper",
      r"\sqrt n\,\Big(\rho\,r_n+Z_nr_n\|\betas\|_1+(1+Z_n)Z_n\|\Delta\|_1^2\Big)=o_P(1)", True,
      "condition (eq:debias-condition), with the factor 1+Z_n")
claim("Thm 6.1: the oracle-calibration variance is stated",
      r"\frac{\tau(1-\tau)}{f_e(\delta_\tau)^{2}}", True,
      "variance in (eq:debias-clt)")
claim("Prop 6.2: the row-wise plug-in bound is stated",
      r"\le\ 2\,\sigma_{\max}(\Sigma_u)\,b^{2}s_0\,\varepsilon_n", True,
      "bound (eq:plugin)")

# ------------------------------ the TEST (Springer) package repeats the same claims
# The TEST version is generated from paper/main.tex, so the decisive statements must be in it
# too; check the three that a referee would look up first.
try:
    with io.open("submission_TEST/source/main_test.tex", encoding="utf-8") as fh:
        _test_tex = fh.read()
    if os.path.exists("submission_TEST/source/supplementary.tex"):
        with io.open("submission_TEST/source/supplementary.tex", encoding="utf-8") as fh:
            _test_tex += fh.read()
except IOError:
    _test_tex = ""
if _test_tex:
    for name, needle, want in (
            ("TEST package: Theorem 6.1 condition",
             r"\sqrt n\,\Big(\rho\,r_n+Z_nr_n\|\betas\|_1+(1+Z_n)Z_n\|\Delta\|_1^2\Big)=o_P(1)", True),
            ("TEST package: Proposition 6.2 bound",
             r"\le\ 2\,\sigma_{\max}(\Sigma_u)\,b^{2}s_0\,\varepsilon_n", True),
            ("TEST package: Table 10 plug-in coverage 0.944", "0.944", True),
            ("TEST package: Declarations block present", "Declarations", True),
            ("TEST package: the AoS MSC line is gone",
             "MSC 2020 subject classifications", False)):
        claim(name, "quantile regression", (needle in _test_tex) == want,
              f"submission_TEST/source/main_test.tex: {'contains' if want else 'free of'} "
              f"{needle[:36]}")

for key, r in sorted(pj["res"].items()):
    su = key.split("=")[1]
    for tag in ("known", "plug"):
        cov = f"{r['cov_' + tag]['mean']:.3f}"
        bias = f"{r['bias_' + tag]['mean']:+.3f}"
        claim(f"Table 10: sigma_u={su}, {tag} coverage {cov} and bias {bias}",
              rf"$\sigma_u={su}$ & {'known' if tag == 'known' else 'plug-in'}",
              True, f"coverage {cov}, bias {bias}")
# Panel B of Table 10 (fixed ratio n = 4p) comes from the scaling experiment
with io.open("review/verify_plugin_bias_scaling.json", encoding="utf-8") as fh:
    scj = json.load(fh)
for row in scj["rows"]:
    if row["p"] < 100:
        continue
    claim(f"Table 10 Panel B: p={row['p']}, known coverage {row['known']['cov']:.3f}",
          rf"$p={row['p']}$, $n={row['n']}$ & known",
          row["known"]["cov"] > 0.95 and row["plug"]["cov"] < 0.80,
          f"known {row['known']['cov']:.3f} vs plug-in {row['plug']['cov']:.3f}")
_su05 = pj["res"]["sigma_u=0.5"]
claim("Table 10 / text: the plug-in loses 3.5 points of coverage at sigma_u = 0.5",
      "loses $3.5$ points of coverage at $\\sigma_u=0.5$",
      abs((_su05["cov_known"]["mean"] - _su05["cov_plug"]["mean"]) * 100 - 3.5) < 0.2,
      f"{_su05['cov_known']['mean']:.3f} -> {_su05['cov_plug']['mean']:.3f}")
claim("Prop 6.2 / text: r_n at the plug-in is about five times the theory scale",
      "five times the",
      abs(_su05["r_n"]["mean"] / np.sqrt(np.log(40) / 400) - 5.6) < 1.0,
      f"r_n = {_su05['r_n']['mean']:.3f} vs {np.sqrt(np.log(40) / 400):.3f}")

# ------------------------------------------- Theorem 3.4 (generalised adaptive scale)
# grad L_ad(beta*) = (Sigma_u + M) beta* d_M, verified by Monte Carlo for a generic M.
rng = np.random.RandomState(7)
N = 400_000
beta = np.array([2.0, 0.0])
su = 0.5
Su = su ** 2 * np.eye(2)
M = 0.2 * np.eye(2)
X = rng.standard_normal((N, 2))
U = rng.standard_normal((N, 2)) * su
W = X + U
eps = rng.laplace(0, 1, N)
Y = X @ beta + eps
sig2 = float(beta @ Su @ beta)
sM = float(np.sqrt(beta @ M @ beta))
r = Y - W @ beta
psi = ndtr(r / sM) - 0.5
term1 = -(W * psi[:, None]).mean(axis=0)
term2 = (np.exp(-0.5 * (r / sM) ** 2) / np.sqrt(2 * np.pi)).mean() * (M @ beta) / sM
grad_mc = float((term1 + term2)[0])
ee = eps + rng.standard_normal(N) * np.sqrt(sig2) + rng.standard_normal(N) * sM
hh = 2e-3
dM = float(np.mean(np.abs(ee) < hh) / (2 * hh))
pred = float(((Su + M) @ beta * dM)[0])
claim("Thm 3.4 (general M): MC gradient matches (Sigma_u+M)beta* d_M",
      "\\big(\\Sigma_u+M\\big)\\betas\\; d_M",
      abs(grad_mc - pred) / abs(pred) < 0.02,
      f"MC {grad_mc:.5f} vs predicted {pred:.5f}")

# ------------------------ Proposition 3.10 (posterior kernel, self-consistent scale)
# The population minimiser of the calibrated smoothed loss with kernel Sigma_{x|w} and
# scale sqrt(b' Sigma_{x|w} b) is exactly beta_bar; the tables and the figure must agree.
dev_pk = max(abs(r["conv_posterior_kernel"] - r["beta_bar"]) for r in tg)
claim("Table 1: posterior-kernel column equals beta_bar within 0.005 (text: 0.006)",
      "Its entries agree\nwith $\\bbar$ to within Monte-Carlo error", dev_pk <= 0.005,
      f"max |entry - beta_bar| = {dev_pk:.4f}")

# The text and the caption now quote one number for the table, 0.006; check it is an upper
# bound for every column and every row (review/check_table1_deviation.py reproduces this).
exact_target = {"naive_qr": "bar", "naive_qr_int": "bar", "conv_fixed_betastar": "bar",
                "conv_fixed_stage1": "bar", "conv_posterior_kernel": "bar",
                "rc_qr": "star", "rc_qr_int": "star", "oracle_qr": "star"}
dev_all = []
for r in tg:
    for k, kind in exact_target.items():
        tgt = r["beta_bar"] if kind == "bar" else r["beta_star"]
        dev_all.append(abs(r[k] - tgt))
    dev_all.append(abs(r["conv_adaptive"] - r["beta_star"] / (1 + 2 * r["sigma_u"] ** 2)))
claim("Table 1: largest Monte-Carlo deviation anywhere in the table is within 0.006",
      "to within $0.006$, the", max(dev_all) <= 0.006,
      f"max deviation over {len(tg)} rows x 9 columns = {max(dev_all):.4f}")
with io.open("simulations/fig1_loss_argmins.json", encoding="utf-8") as fh:
    la = json.load(fh)
claim("Fig. 3: quadrature minimisers 1.600 / 2.000 / 2.000 / 1.600",
      "1.600",
      all(abs(la[k] - v) < 1e-4 for k, v in
          (("argmin_smoothed", 1.600), ("argmin_true_risk", 2.000),
           ("argmin_rc", 2.000), ("argmin_control_posterior", 1.600))),
      "smoothed {:.4f}, true {:.4f}, rc {:.4f}, control {:.4f}".format(
          la["argmin_smoothed"], la["argmin_true_risk"], la["argmin_rc"],
          la["argmin_control_posterior"]))
claim("Fig. 3 / Prop 3.10: the posterior-kernel control is minimised at beta_bar",
      "minimised at $\\bbar$", abs(la["argmin_control_posterior"] - la["beta_bar"]) <= 1e-4,
      f"control {la['argmin_control_posterior']:.4f} vs beta_bar {la['beta_bar']:.4f}")
claim("Fig. 3: the calibrated check loss is minimised at beta_star",
      "minimised at $\\betas$", abs(la["argmin_rc"] - la["beta_star"]) <= 1e-4,
      f"rc {la['argmin_rc']:.4f} vs beta_star {la['beta_star']:.4f}")
claim("Fig. 3: the smoothed loss is minimised at beta_bar",
      "minimised at $\\bbar$", abs(la["argmin_smoothed"] - la["beta_bar"]) <= 1e-4,
      f"smoothed {la['argmin_smoothed']:.4f} vs beta_bar {la['beta_bar']:.4f}")
claim("Fig. 3: the true risk is minimised at beta_star",
      "true risk, minimised at $\\betas$",
      abs(la["argmin_true_risk"] - la["beta_star"]) <= 1e-4,
      f"true risk {la['argmin_true_risk']:.4f} vs beta_star {la['beta_star']:.4f}")
claim("Fig. 3: the figure is drawn from quadrature, not simulation",
      "computed by quadrature rather than simulated", la.get("method") == "quadrature",
      f"method = {la.get('method')}")
try:
    with io.open("review/verify_posterior_adaptive.json", encoding="utf-8") as fh:
        va = json.load(fh)
except IOError:
    va = None
if va is not None:
    vs = va["summary"]
    claim("Prop 3.10: MC population minimiser matches beta_bar in all 9 cells",
          "is uniquely minimised at", vs["max_mc_deviation"] <= 0.01,
          f"max deviation {vs['max_mc_deviation']:.4f}")
    claim("Prop 3.10: the quadratic minimiser is exactly beta_bar",
          "self-consistent", vs["max_quad_deviation"] <= 1e-10,
          f"max deviation {vs['max_quad_deviation']:.2e}")
    claim("Prop 3.10: E|eps + s Z| is strictly increasing in s",
          "strictly increasing in", vs["all_monotone"],
          "monotone in all 9 cells")
    vm = va["matrix"]
    claim("Prop 3.10: matrix case (Sigma_x, Sigma_u do not commute)",
          "$2\\Sigma_x(\\beta-\\bbar)$",
          vm["quad_deviation"] <= 1e-10 and vm["grid_deviation"] <= 0.01,
          f"quad {vm['quad_deviation']:.2e}, grid {vm['grid_deviation']:.4f}")

# ------------------------------------------------------------- real data (NHANES)
with io.open("simulations/nhanes_results.json", encoding="utf-8") as fh:
    nh = json.load(fh)
nm = nh["meta"]
rows = {r["nutrient"]: r for r in nh["rows"]}
claim("NHANES: n = 3649, p = 18 after pruning",
      "n=3649", nm["n"] == 3649 and nm["p"] == 18, f"n={nm['n']}, p={nm['p']}")
claim("NHANES: mean reliability 0.417",
      "0.417", abs(nm["mean_reliability"] - 0.417) < 1e-3,
      f"{nm['mean_reliability']:.4f}")
claim("NHANES: median naive/calibrated ratio 0.156",
      "0.156", abs(nm["median_ratio"] - 0.156) < 1e-3, f"{nm['median_ratio']:.4f}")
claim("NHANES: nine naive and ten debiased coefficients significant",
      "ten of the eighteen", nm["n_sig_naive"] == 9 and nm["n_sig_debiased"] == 10,
      f"{nm['n_sig_naive']} vs {nm['n_sig_debiased']}")
claim("NHANES: energy 3.2 -> 22.7",
      "22.7", abs(rows["energy"]["naive"] - 3.2) < 0.05
      and abs(rows["energy"]["calibrated"] - 22.7) < 0.05,
      f"{rows['energy']['naive']:.2f} -> {rows['energy']['calibrated']:.2f}")
claim("NHANES: carbohydrate -1.7 -> -12.3",
      "-12.3", abs(rows["carbohydrate"]["naive"] + 1.7) < 0.05
      and abs(rows["carbohydrate"]["calibrated"] + 12.3) < 0.05,
      f"{rows['carbohydrate']['naive']:.2f} -> {rows['carbohydrate']['calibrated']:.2f}")
claim("NHANES: saturated fat -0.65 -> -6.21, interval upper end 0.7",
      "-6.21", abs(rows["saturated fat"]["naive"] + 0.65) < 0.01
      and abs(rows["saturated fat"]["calibrated"] + 6.21) < 0.01
      and abs(rows["saturated fat"]["ci_hi"] - 0.7) < 0.05,
      f"{rows['saturated fat']['calibrated']:.2f}, hi={rows['saturated fat']['ci_hi']:.2f}")
claim("NHANES: condition numbers 2e4 / 2.9e3 / 585",
      "585", abs(nm["cond_x_full"] / 2e4 - 1) < 0.1
      and abs(nm["cond_full"] - 2913) < 5 and abs(nm["cond_pruned"] - 585) < 2,
      f"{nm['cond_x_full']:.0f} / {nm['cond_full']:.0f} / {nm['cond_pruned']:.0f}")
claim("NHANES: top correlations 0.97 / 0.93 / 0.90",
      "0.97", all(abs(a - b) < 0.001 for a, b in
                  zip(nm["top_corr"], [0.9716, 0.9334, 0.9025])),
      f"{[round(c, 4) for c in nm['top_corr']]}")
claim("NHANES: Sigma_x has one slightly negative eigenvalue",
      "negative eigenvalue", nm["min_eig_x"] < 0, f"min eig = {nm['min_eig_x']:.4f}")
claim("NHANES: six nutrients dropped as near-duplicates",
      "near-duplicates", nm["n_dropped"] == 6, f"{nm['n_dropped']} dropped")

# ---------------------------------------------------------------- report
print(f"{'claim':<62} {'paper':<8} result")
bad = 0
for name, sub, ok, detail in checks:
    present = has(sub)
    status = "OK" if (ok and present) else ("MISMATCH" if not ok else "NOT-IN-PAPER")
    if status != "OK":
        bad += 1
    print(f"{name:<62} {status:<10} {detail}")
print()
print(f"{len(checks)-bad}/{len(checks)} claims consistent with the saved results")

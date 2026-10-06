"""
Real-data illustration: dietary intake measurement error and BMI.

Data: NHANES 2017-2018 (public), files DEMO_J, BMX_J, DR1TOT_J, DR2TOT_J.
The two 24-hour dietary recalls give GENUINE replicates of the same quantity, so both
    Sigma_x = Cov(recall1, recall2)        and     Sigma_u = (1/2) Var(recall1 - recall2)
are estimable from the data --- no synthetic noise is added anywhere.  This is the
"test-retest" route to the measurement-error covariance that the paper invokes.

Analysis
  * naive quantile regression of BMI on the day-1 recalls (the error-prone surrogate);
  * calibrated quantile regression on mu_W = C W with C = Sigma_x (Sigma_x + Sigma_u)^{-1};
  * debiased intervals for the calibrated estimator (refit + exact-score one-step
    debiasing), as validated in simulations.
Reported: estimated reliability ratios, the two coefficient vectors, and how many
nutrients change significance.

Caveats (reported in the paper): BMI is itself measured with error; the two recalls are
not iid across days (day-of-week and under-reporting effects); survey weights are not used
in this methodological illustration.
"""

import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdc_debias import debias, Z95
from hdc_debias_final import sandwich, refit

from sklearn.linear_model import QuantileRegressor

DATA = "data/nhanes/"
LOG = "simulations/nhanes_output.txt"
OUT = []

NUTRIENTS = [
    ("DR1TKCAL", "energy"), ("DR1TPROT", "protein"), ("DR1TCARB", "carbohydrate"),
    ("DR1TSUGR", "total sugars"), ("DR1TFIBE", "fibre"), ("DR1TTFAT", "total fat"),
    ("DR1TSFAT", "saturated fat"), ("DR1TMFAT", "mono fat"), ("DR1TPFAT", "poly fat"),
    ("DR1TCHOL", "cholesterol"), ("DR1TSODI", "sodium"), ("DR1TPOTA", "potassium"),
    ("DR1TCALC", "calcium"), ("DR1TPHOS", "phosphorus"), ("DR1TMAGN", "magnesium"),
    ("DR1TIRON", "iron"), ("DR1TZINC", "zinc"), ("DR1TVC", "vitamin C"),
    ("DR1TVD", "vitamin D"), ("DR1TFOLA", "folate"), ("DR1TVB12", "vitamin B12"),
    ("DR1TALCO", "alcohol"), ("DR1TCAFF", "caffeine"), ("DR1TMOIS", "moisture"),
]


def log(m=""):
    print(m, flush=True)
    OUT.append(str(m))
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))


def load():
    d1 = pd.read_sas(DATA + "DR1TOT_J.xpt")
    d2 = pd.read_sas(DATA + "DR2TOT_J.xpt")
    demo = pd.read_sas(DATA + "DEMO_J.xpt")
    bmx = pd.read_sas(DATA + "BMX_J.xpt")
    v1 = [c for c, _ in NUTRIENTS]
    v2 = ["DR2" + c[3:] for c in v1]
    keep1 = ["SEQN", "DR1DRSTZ"] + v1
    keep2 = ["SEQN", "DR2DRSTZ"] + v2
    df = d1[keep1].merge(d2[keep2], on="SEQN", how="inner")
    df = df.merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR"]], on="SEQN")
    df = df.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN")
    df = df[(df["DR1DRSTZ"] == 1) & (df["DR2DRSTZ"] == 1)]
    df = df[df["RIDAGEYR"] >= 20]
    df = df.dropna(subset=v1 + v2 + ["BMXBMI", "INDFMPIR"])
    return df, v1, v2


def prune_collinear(R1, names, Sigma_x, Sigma_u, kappa_max=100.0):
    """Remove near-duplicate nutrients by hand, then report the conditioning.

    A greedy rule that drops whichever variable is most correlated with another removes
    the exposures of interest first (in this dataset it discards energy and protein in
    favour of phosphorus and carbohydrate), so the exclusions are listed explicitly:
    total/mono/poly fat are near-duplicates of one another (r = 0.97, 0.93, 0.87) and
    saturated fat is retained as the exposure of interest; phosphorus duplicates protein
    (r = 0.90), potassium duplicates magnesium (r = 0.87), and moisture is not a nutrient.
    """
    drop = {"mono fat", "poly fat", "total fat", "phosphorus", "potassium", "moisture"}
    keep = [j for j, nm in enumerate(names) if nm not in drop]
    return keep, sorted(nm for nm in names if nm in drop)


def main(tau=0.5, R_boot=0):
    df, v1, v2 = load()
    R1_all = np.log1p(df[v1].to_numpy(float))
    R2_all = np.log1p(df[v2].to_numpy(float))
    y = df["BMXBMI"].to_numpy(float)
    names_all = [n for _, n in NUTRIENTS]
    p_all = R1_all.shape[1]

    # ---- measurement-error covariance from the two replicates (no synthetic noise)
    Su_all = 0.5 * np.cov(R1_all - R2_all, rowvar=False)
    Sx_all = np.cov(R1_all, R2_all, rowvar=False)[:p_all, p_all:]
    log("=" * 100)
    log("NHANES 2017-2018: dietary intake measurement error and BMI")
    log("=" * 100)
    log(f"n = {R1_all.shape[0]} adults (age >= 20, two reliable 24-h recalls, complete "
        f"cases), {p_all} nutrients on the log scale")
    log(f"unregularised calibration: cond(Sigma_x) = {np.linalg.cond(Sx_all):.0f}, "
        f"cond(Sigma_x+Sigma_u) = {np.linalg.cond(Sx_all + Su_all):.0f} "
        f"(the plug-in is numerically explosive at this condition number, so the design "
        f"is pruned before the correction is computed)")
    keep, dropped = prune_collinear(R1_all, names_all, Sx_all, Su_all)
    Corr = np.corrcoef(R1_all, rowvar=False)
    iu = np.triu_indices(p_all, 1)
    top = sorted(zip(Corr[iu], [(names_all[i], names_all[j]) for i, j in zip(*iu)]),
                 reverse=True)[:3]
    log("most collinear pairs: " + "; ".join(f"{a}/{b} {r:.2f}" for r, (a, b) in top))
    log(f"dropped near-duplicate nutrients (largest absolute correlation first): "
        f"{', '.join(dropped) if dropped else 'none'}")
    log(f"retained p = {len(keep)}: {', '.join(names_all[j] for j in keep)}")
    log("")

    R1, R2 = R1_all[:, keep], R2_all[:, keep]
    names = [names_all[j] for j in keep]
    p = R1.shape[1]
    Sigma_u = Su_all[np.ix_(keep, keep)]
    Sigma_x = Sx_all[np.ix_(keep, keep)]
    n = R1.shape[0]
    log(f"after pruning: cond(Sigma_x+Sigma_u) = {np.linalg.cond(Sigma_x + Sigma_u):.1f}; "
        f"eigenvalues of Sigma_x: min {np.linalg.eigvalsh(Sigma_x).min():.4f}, "
        f"max {np.linalg.eigvalsh(Sigma_x).max():.4f}")
    C = Sigma_x @ np.linalg.inv(Sigma_x + Sigma_u)
    rel = np.diag(Sigma_x) / (np.diag(Sigma_x) + np.diag(Sigma_u))
    log("Estimated reliability Sigma_x,jj / (Sigma_x,jj + Sigma_u,jj) from the replicate pair:")
    for j, name in enumerate(names):
        log(f"   {name:<15} {rel[j]:.3f}")
    log(f"   mean reliability {rel.mean():.3f}")
    log("")

    muW = R1 @ C.T

    # ---- quantile regressions
    def fit(D0, tau0, lam=0.0):
        q = QuantileRegressor(quantile=tau0, alpha=lam, solver="highs")
        q.fit(D0, y)
        return q.coef_, float(q.intercept_)

    b_nv, d_nv = fit(R1, tau)
    b_ca, d_ca = fit(muW, tau)
    # debiased intervals for the calibrated estimator, with a refitted initial estimator
    support = np.abs(b_ca) > 1e-8
    b_rf, d_rf = refit(y, muW, support, tau)
    b_db, _, se_db = debias(y, muW, b_rf, d_rf, tau, score="exact")

    log(f"tau = {tau}; coefficients (change in BMI per unit of log intake)")
    log(f"{'nutrient':<16} {'reliab.':>8} {'naive':>9} {'calibrated':>11} "
        f"{'95% CI (debiased)':>24} {'sig?':>5}")
    rows = []
    for j, name in enumerate(names):
        lo, hi = b_db[j] - Z95 * se_db[j], b_db[j] + Z95 * se_db[j]
        sig = (lo > 0) or (hi < 0)
        rows.append(dict(nutrient=name, reliability=float(rel[j]),
                         naive=float(b_nv[j]), calibrated=float(b_ca[j]),
                         ci_lo=float(lo), ci_hi=float(hi), significant=bool(sig)))
        log(f"{name:<16} {rel[j]:>8.3f} {b_nv[j]:>9.3f} {b_ca[j]:>11.3f} "
            f"{('[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']'):>24} "
            f"{('yes' if sig else 'no'):>5}")
    log("")
    n_sig_db = sum(r["significant"] for r in rows)
    # significance of the naive estimates under the same (debiased) standard errors
    n_sig_nv = 0
    se_nv = sandwich(y, R1, b_nv, d_nv, tau)[:p]
    for j in range(p):
        lo, hi = b_nv[j] - Z95 * se_nv[j], b_nv[j] + Z95 * se_nv[j]
        n_sig_nv += (lo > 0) or (hi < 0)
    # attenuation: how much smaller is the naive coefficient than the calibrated one
    ratio = np.array([r["naive"] / r["calibrated"] for r in rows
                      if abs(r["calibrated"]) > 1e-8])
    log(f"naive coefficients significant: {n_sig_nv} of {p}; "
        f"calibrated (debiased) significant: {n_sig_db} of {p}")
    log(f"median naive/calibrated coefficient ratio: {np.median(ratio):.3f} "
        f"(a value near the mean reliability {rel.mean():.3f} would indicate pure attenuation)")
    log(f"||b_naive - b_calibrated||_2 = {np.linalg.norm(b_nv - b_ca):.4f}; "
        f"||b_calibrated||_2 = {np.linalg.norm(b_ca):.4f}")

    out = dict(meta=dict(n=n, p=p, tau=tau, mean_reliability=float(rel.mean()),
                         n_sig_naive=int(n_sig_nv), n_sig_debiased=int(n_sig_db),
                         median_ratio=float(np.median(ratio)),
                         norm_diff=float(np.linalg.norm(b_nv - b_ca)),
                         norm_cal=float(np.linalg.norm(b_ca)),
                         cond_x_full=float(np.linalg.cond(Sx_all)),
                         cond_full=float(np.linalg.cond(Sx_all + Su_all)),
                         cond_pruned=float(np.linalg.cond(Sigma_x + Sigma_u)),
                         n_dropped=int(len(dropped)), dropped=dropped,
                         top_corr=[float(r) for r, _ in top],
                         min_eig_x=float(np.linalg.eigvalsh(Sigma_x).min())),
               rows=rows)
    with io.open("simulations/nhanes_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    t0 = __import__("time").time()
    main()
    log(f"[total {__import__('time').time()-t0:.1f}s]")
    print("\n[saved] simulations/nhanes_results.json")

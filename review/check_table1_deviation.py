"""Largest Monte-Carlo deviation in Table 1 (tab_targets) from the exact targets."""
import io
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = json.load(io.open("simulations/routeA_targets.json", encoding="utf-8"))

# exact targets: beta_bar for naive and every fixed-scale smoothed loss (Prop. 3.13(i),(ii)),
# (Sigma_x+Sigma_u+M)^{-1} Sigma_x beta* for the adaptive kernel M = Sigma_u (Prop. 3.13(iii)),
# beta* for the calibration-based estimators, beta_bar for the posterior-kernel control.
exact = {
    "naive_qr": "bar", "naive_qr_int": "bar",
    "conv_fixed_betastar": "bar", "conv_fixed_stage1": "bar",
    "conv_posterior_kernel": "bar",
    "rc_qr": "star", "rc_qr_int": "star", "oracle_qr": "star",
}

devs = {}
for r in rows:
    su = r["sigma_u"]
    adaptive_exact = r["beta_star"] / (1.0 + 2.0 * su ** 2)
    for k, kind in exact.items():
        tgt = r["beta_bar"] if kind == "bar" else r["beta_star"]
        devs.setdefault(k, []).append(abs(r[k] - tgt))
    devs.setdefault("conv_adaptive", []).append(abs(r["conv_adaptive"] - adaptive_exact))

print(f"{'column':24s} {'n':>3s} {'max |dev|':>10s} {'mean |dev|':>10s}")
worst = 0.0
for k, v in devs.items():
    v = np.array(v)
    print(f"{k:24s} {len(v):3d} {v.max():10.4f} {v.mean():10.4f}")
    worst = max(worst, v.max())
print(f"\nlargest deviation over all {len(rows)} rows and all columns: {worst:.4f}")
bar_dev = max(max(devs[k]) for k in ("conv_posterior_kernel",))
print(f"posterior-kernel column vs beta_bar: {bar_dev:.4f}")

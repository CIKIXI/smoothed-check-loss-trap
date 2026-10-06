r"""
Determinism check for the debiasing experiment (Panel A of Table 9).

Every reported number must be reproducible from the shipped code and seeds.  This script
recomputes a subset of the Panel A design and prints a hash of the per-replication
coverage indicators; running it twice must give the same hash.

Run:  python review/verify_debias_determinism.py 40
"""

import hashlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations"))

from hdc_debias import fit_l1, debias, Z95                     # noqa: E402
from hdc_debias_final import refit, sandwich                   # noqa: E402
from routeA_lib import calibration_matrix, draw_eps            # noqa: E402


def main(R=40, n=400, p=40, s=5, beta_val=2.0, sigma_u=0.3, tau=0.5, seed0=24680):
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    lam0 = float(np.sqrt(np.log(p) / n))
    C = calibration_matrix(np.eye(p), sigma_u ** 2 * np.eye(p))
    bits = []
    for rep in range(R):
        rng = np.random.RandomState(seed0 + rep * 7919)
        X = rng.standard_normal((n, p))
        W = X + rng.standard_normal((n, p)) * sigma_u
        Y = X @ beta_star + draw_eps(n, tau, rng)
        muW = W @ C.T
        b_l1, d_l1 = fit_l1(Y, muW, tau, lam0)
        b_db, _, se_db = debias(Y, muW, b_l1, d_l1, tau, score="exact")
        support = np.abs(b_l1) > 1e-8
        b_rf, d_rf = refit(Y, muW, support, tau)
        b_rd, _, se_rd = debias(Y, muW, b_rf, d_rf, tau, score="exact")
        sup = np.arange(s)
        bits.append(np.concatenate([
            np.abs(b_l1[sup] - beta_star[sup]) <= Z95 * sandwich(Y, muW, b_l1, d_l1, tau)[:p][sup],
            np.abs(b_db[sup] - beta_star[sup]) <= Z95 * se_db[sup],
            np.abs(b_rd[sup] - beta_star[sup]) <= Z95 * se_rd[sup],
            [float(support.sum())],
        ]))
    arr = np.concatenate(bits)
    h = hashlib.sha256(np.round(arr, 6).tobytes()).hexdigest()[:32]
    print(f"R = {R}  hash = {h}")
    print(f"  l1 {arr[0:R*5].mean():.4f}  deb {arr[R*5:R*10].mean():.4f}  "
          f"rd {arr[R*10:R*15].mean():.4f}  support {arr[R*15:].mean():.3f}")
    with io.open("review/verify_debias_determinism.txt", "a", encoding="utf-8") as fh:
        fh.write(f"R={R} hash={h} l1={arr[0:R*5].mean():.4f} "
                 f"deb={arr[R*5:R*10].mean():.4f} rd={arr[R*10:R*15].mean():.4f}\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)

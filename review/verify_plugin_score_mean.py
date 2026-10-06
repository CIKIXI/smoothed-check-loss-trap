r"""Population mean of the estimated-design score at the truth, plug-in vs known C."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations"))

from routeA_lib import calibration_matrix, draw_eps, phi_std   # noqa: E402


def main(N=400_000, p=40, s=5, beta_val=2.0, sigma_u=0.5, tau=0.5, seed=11):
    beta_star = np.zeros(p)
    beta_star[:s] = beta_val
    Su = sigma_u ** 2 * np.eye(p)
    C = calibration_matrix(np.eye(p), Su)
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((N, p))
    U = rng.standard_normal((N, p)) * sigma_u
    W = X + U
    Y = X @ beta_star + draw_eps(N, tau, rng)
    delta = float(np.quantile(Y - W @ C.T @ beta_star, tau))
    SW = W.T @ W / N
    C_hat = np.eye(p) - Su @ np.linalg.inv(SW)
    D = SW - Su - np.eye(p)          # Sigma_x_hat - Sigma_x with Sigma_x = I

    sup = np.arange(s)
    for tag, Cu in (("known", C), ("plug", C_hat)):
        muW = W @ Cu.T
        e = Y - muW @ beta_star
        # centre the indicator at the *design-specific* quantile so that the comparison is
        # about the score's mean and not about a location shift
        d = float(np.quantile(e, tau))
        psi = tau - (e - d < 0).astype(float)
        S = muW.T @ psi / N
        print(f"{tag:6s}: mean score on the signal coords {np.round(S[sup], 5)}")
        print(f"        mean |score| on the other coords    "
              f"{np.abs(S[s:]).mean():.5f}")

    # predicted mean of the estimated-design score at beta*, up to the f_e factor:
    #   E[S] = -f_e(delta) (Sigma_x + D) Delta' beta*,  Delta = C_hat - C
    Delta = C_hat - C
    pred = -(np.eye(p) + D) @ Delta.T @ beta_star
    print(f"\npredicted (up to -f_e(delta)) (Sigma_x+D)Delta'beta* on the signal coords "
          f"{np.round(pred[sup], 5)}")
    print(f"f_e(delta) = {phi_std(0.0)/1.0:.4f} for Laplace (density at 0 = 1/2)")


if __name__ == "__main__":
    main()

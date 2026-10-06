# Claim to be checked (statement only — no proof is provided on purpose)

## Setting

Data $(Y_i, W_i)$, $i=1,\dots,n$, with

* $X_i \sim \mathcal N(0, \Sigma_x)$, $\Sigma_x \succ 0$, $\mathbb E[X_i] = 0$;
* $U_i \sim \mathcal N(0, \Sigma_u)$, $\Sigma_u \succeq 0$, independent of $X_i$;
* $W_i = X_i + U_i$;
* $Y_i = X_i^\top \beta^\* + \varepsilon_i$ where $\beta^\* \in \mathbb R^p$ is fixed and
  $\varepsilon_i$ is independent of $(X_i, U_i)$ with **any** distribution (only a density
  and $\mathbb E|\varepsilon_i| < \infty$ are assumed).

Notation: $\tau \in (0,1)$; $\rho_\tau(r) = r(\tau - \mathbf 1\{r<0\})$ is the check loss;

$$\tilde\rho_\tau(r,\sigma) \;=\; \mathbb E_Z[\rho_\tau(r - \sigma Z)], \qquad Z \sim \mathcal N(0,1) \text{ independent},$$

is the check loss convolved with $\mathcal N(0,\sigma^2)$ (the "smoothed" loss);

$$\bar\beta \;=\; (\Sigma_x + \Sigma_u)^{-1}\Sigma_x \beta^\* .$$

## Claim

Fix $\tau \in (0,1)$. Then each of the following population criteria has a **unique**
minimiser, and it is as stated.

**(i) Naive check loss.** $\beta \mapsto \mathbb E[\rho_\tau(Y_i - \beta^\top W_i)]$ is
uniquely minimised at $\beta = \bar\beta$.

**(ii) Fixed-scale smoothed loss.** For **every** $\sigma > 0$,
$\beta \mapsto \mathbb E[\tilde\rho_\tau(Y_i - \beta^\top W_i, \sigma)]$ is uniquely
minimised at $\beta = \bar\beta$ — i.e. the bandwidth does not move the target.

**(iii) Adaptive scale.** For any $M \succeq 0$, with $\sigma_M(\beta) = \sqrt{\beta^\top M \beta}$,
the criterion $\beta \mapsto \mathbb E[\tilde\rho_\tau(Y_i - \beta^\top W_i, \sigma_M(\beta))]$
is uniquely minimised at

$$\beta^\dagger(M) \;=\; (\Sigma_x + \Sigma_u + M)^{-1}\Sigma_x \beta^\* .$$

In the scalar case $p = 1$, $\Sigma_x = 1$, $\Sigma_u = \sigma_u^2$, (ii) says the minimiser
is $\beta^\*/(1+\sigma_u^2)$ and (iii) with $M = \Sigma_u$ says it is
$\beta^\*/(1+2\sigma_u^2)$.

## What is being asked

1. Re-derive these statements from scratch.
2. Look for counterexamples: vary $\tau$, the error distribution (including asymmetric and
   heavy-tailed ones), $\Sigma_x, \Sigma_u$ (including singular $\Sigma_u$ and
   non-commuting matrices), $M$ (including $M$ not commuting with $\Sigma_x$ or $\Sigma_u$,
   and $M$ singular), the bandwidth $\sigma$, and the mean of $X$ (note: $\mathbb E[X_i]=0$
   is assumed — check whether the claim survives if it fails).
3. State clearly whether each of (i), (ii), (iii) is true, false, or true only under extra
   assumptions, and give numerical evidence (your own code) for whatever you conclude.

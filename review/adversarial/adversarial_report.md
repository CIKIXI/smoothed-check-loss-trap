# Adversarial check of `statement_prop313.md`

**Verdict: parts (i), (ii) and (iii) are TRUE exactly as stated.** I could not
construct a counterexample, and every apparent counterexample the numerics
produced turned out to be a floating-point artefact that dissolves in
high-precision arithmetic. Two of the stated assumptions are nevertheless
essential, one is superfluous, and one part (the "irrelevance of the bandwidth")
is true but much weaker numerically than it reads.

Everything below is reproducible with

```
python adversarial_check.py --stage all --n-mc 8000 --n-mc-grad 2000000 --n-sm 200000
python make_report_tables.py            # -> results/tables.md (all raw tables)
```

`results/tables.md` holds the per-configuration numbers; this file holds the
derivation, the verdicts and the interpretation.

---

## 0. Verdict per part

| part | criterion | claimed minimiser | verdict | why |
|---|---|---|---|---|
| (i) | $E[\rho_\tau(Y_i-\beta'W_i)]$ | $\bar\beta=(\Sigma_x+\Sigma_u)^{-1}\Sigma_x\beta^\*$ | **TRUE** | the criterion equals $\Lambda(\sqrt{q_0(\beta)})$ with $\Lambda$ strictly increasing and $q_0$ a strictly convex quadratic (§2.4) |
| (ii) | $E[\tilde\rho_\tau(Y_i-\beta'W_i,\sigma)]$, every $\sigma>0$ | $\bar\beta$ | **TRUE** | smoothing only adds $\sigma^2$ to the residual variance, so the argmin is unchanged; but the *minimum value* moves and the contrast decays like $\varphi(0)/(2\sigma)$ (§2.5, §7) |
| (iii) | $E[\tilde\rho_\tau(Y_i-\beta'W_i,\sigma_M(\beta))]$, $M\succeq0$ | $\beta^\dagger=(\Sigma_x+\Sigma_u+M)^{-1}\Sigma_x\beta^\*$ | **TRUE** | same reduction with $q_0(\beta)+\beta'M\beta$; no commutativity, no non-singular $\Sigma_u$, no non-singular $M$ needed (§2.6) |
| scalar special case 1 | $p=1$, $\Sigma_x=1$, $\Sigma_u=\sigma_u^2$ | $\beta^\*/(1+\sigma_u^2)$ | **TRUE** | 3000 configurations (12 laws × 5 $\sigma_u$ × 5 $\tau$ × 10 $\sigma$): max deviation $2.4\cdot10^{-9}$, median $0$ over the ten non-degenerate laws |
| scalar special case 2 | (iii) with $M=\Sigma_u$ | $\beta^\*/(1+2\sigma_u^2)$ | **TRUE** | 300 configurations (12 laws × 5 $\sigma_u$ × 5 $\tau$): max deviation $2.3\cdot10^{-9}$, median $0$ over the ten non-degenerate laws |

Role of each assumption in the statement (all tested, §6):

| assumption | role |
|---|---|
| $E[X]=0$ | **essential** — drop it and the criterion is no longer a function of the residual variance and $\bar\beta$ is not the minimiser |
| $\Sigma_x\succ0$ | **essential as written** — the true condition is $\Sigma_x+\Sigma_u+M\succ0$; with singular $\Sigma_x$ and $\Sigma_u+M$ vanishing on a common direction the minimiser is not unique |
| $E\lvert\varepsilon\rvert<\infty$ | **needed for finiteness** — with Cauchy errors every value is $+\infty$ |
| $\varepsilon$ has a density | **not needed** (statement too weak here) — the proof only uses that $\varepsilon$ is a random variable; atomic and degenerate $\varepsilon$ pass |
| $M\succeq0$ | needed so that $\sigma_M(\beta)$ is real; no relation to $\Sigma_x,\Sigma_u$ required |

---

## 1. Setting and notation

$X\sim N(0,\Sigma_x)$, $\Sigma_x\succ0$; $U\sim N(0,\Sigma_u)$, $\Sigma_u\succeq0$,
$U\perp X$; $W=X+U$; $Y=X'\beta^\*+\varepsilon$ with $\varepsilon\perp(X,U)$;
$\rho_\tau(r)=r(\tau-\mathbf 1\{r<0\})$;
$\tilde\rho_\tau(r,\sigma)=E_Z[\rho_\tau(r-\sigma Z)]$ with $Z\sim N(0,1)$
independent;

$$\bar\beta=(\Sigma_x+\Sigma_u)^{-1}\Sigma_x\beta^\*,\qquad
\beta^\dagger(M)=(\Sigma_x+\Sigma_u+M)^{-1}\Sigma_x\beta^\*.$$

## 2. Derivation from first principles

### 2.1 The residual is $\varepsilon$ plus a centred Gaussian with variance a quadratic form in $\beta$

$$r(\beta):=Y-\beta'W
=\varepsilon+\underbrace{X'(\beta^\*-\beta)-\beta'U}_{=:\,\xi(\beta)} .$$

Because $X$ and $U$ are independent centred Gaussians, for each fixed $\beta$

$$\xi(\beta)\sim N\!\big(0,q_0(\beta)\big),\qquad
q_0(\beta)=(\beta^\*-\beta)'\Sigma_x(\beta^\*-\beta)+\beta'\Sigma_u\beta\ \ge0,
\tag{2.1}$$

$\xi(\beta)\perp\varepsilon$, and — because $E[X]=E[U]=0$ —

$$E[r(\beta)]=\mu_x'(\beta^\*-\beta)=0\quad\text{for every }\beta. \tag{2.2}$$

### 2.2 The check loss is "linear plus half-absolute", so $\tau$ cancels

Since $\rho_\tau(r)=\tau r+(-r)^{+}$ and $E[r(\beta)]\equiv0$,

$$E[\rho_\tau(r(\beta))]=\tau\underbrace{E[r(\beta)]}_{=\,0}
+E\big[(-r(\beta))^{+}\big]=E\big[(-\big(\varepsilon+\xi(\beta)\big))^{+}\big].$$

With $\xi(\beta)\stackrel{d}{=}\sqrt{q_0(\beta)}Z$ and independence,

$$\boxed{\,R(\beta)=\Lambda\!\big(\sqrt{q_0(\beta)}\big)\,},\qquad
\Lambda(s):=E\big[(-\varepsilon-sZ)^{+}\big]. \tag{2.3}$$

**The entire $\tau$-dependence cancels.** This is the structural reason why all
three parts are true: $\beta$ enters the criterion *only* through the residual
variance. Nothing about signs, directions, or commutation can matter.

### 2.3 Key lemma: $\Lambda$ is strictly increasing

For $W\sim N(m,s^2)$, $E[(-W)^{+}]=s\varphi(m/s)-m\Phi(-m/s)=:g(m,s)$, hence
$\Lambda(s)=E[g(\varepsilon,s)]$ and, since $\partial_s g(m,s)=\varphi(m/s)>0$
pointwise (equivalently Stein's lemma: $\Lambda'(s)=s\,E[f_{\varepsilon+sZ}(0)]$),

$$\Lambda'(s)=E\big[\varphi(\varepsilon/s)\big]>0\qquad(s>0). \tag{2.4}$$

$\varphi>0$ everywhere and $\varepsilon$ is a genuine random variable, so (2.4)
holds for **every** law of $\varepsilon$ — atomic, discrete, heavy-tailed.
Moreover $\Lambda'(0)=0$ and $\Lambda(s)=\Lambda(0)+\tfrac12f_\varepsilon(0)s^2+O(s^3)$
when a density exists at $0$; $\Lambda$ is convex, $C^1$, and asymptotically
$\Lambda(s)=s\varphi(0)+O(1)$.

### 2.4 Part (i)

$R=\Lambda\circ\sqrt{q_0}$ with $\Lambda$ strictly increasing $\Rightarrow$
$\arg\min R=\arg\min q_0$. The latter is a strictly convex quadratic with Gram
matrix $\Sigma_x+\Sigma_u\succ0$ (automatic from $\Sigma_x\succ0$), so

$$\nabla q_0(\beta)=0\iff(\Sigma_x+\Sigma_u)\beta=\Sigma_x\beta^\*
\iff\beta=\bar\beta\quad\text{(unique)} .$$

### 2.5 Part (ii)

For fixed $\beta$, $r(\beta)-\sigma Z=\varepsilon+\xi(\beta)-\sigma Z$ and
$\xi(\beta)-\sigma Z\sim N(0,q_0(\beta)+\sigma^2)$ (sum of independent centred
Gaussians), independent of $\varepsilon$. Hence for every $\sigma>0$

$$R_\sigma(\beta)=E[\tilde\rho_\tau(r(\beta),\sigma)]
=\Lambda\!\Big(\sqrt{q_0(\beta)+\sigma^2}\Big). \tag{2.5}$$

Smoothing adds $\sigma^2$ to the variance and *nothing else*: the argmin is again
the unique minimiser of $q_0$, i.e. $\bar\beta$, at every bandwidth. Note that
$R_\sigma(\bar\beta)=\Lambda(\sqrt{q_0(\bar\beta)+\sigma^2})$ **does** increase with
$\sigma$, and
$R_\sigma(\beta)-R_\sigma(\bar\beta)=\frac{\varphi(0)}{2\sigma}\big(q_0(\beta)-q_0(\bar\beta)\big)+O(\sigma^{-3})$
— the target is invariant, the contrast is not (§7).

### 2.6 Part (iii)

For fixed $\beta$, $\xi(\beta)-\sigma_M(\beta)Z\sim N(0,q_0(\beta)+\beta'M\beta)$, so

$$R_M(\beta)=\Lambda\!\Big(\sqrt{q_M(\beta)}\Big),\quad
q_M(\beta)=q_0(\beta)+\beta'M\beta
=\beta'(\Sigma_x{+}\Sigma_u{+}M)\beta-2\beta'\Sigma_x\beta^\*
+\beta^{*\prime}\Sigma_x\beta^\*,$$

whose unique minimiser solves $(\Sigma_x+\Sigma_u+M)\beta=\Sigma_x\beta^\*$, i.e.
$\beta=\beta^\dagger(M)$.

**No commutativity is used anywhere.** No simultaneous diagonalisation, no
eigenvalue ordering, no $[\Sigma_x,\Sigma_u]=0$ or $[M,\Sigma_x]=0$: the criterion
depends on $\beta$ only through a scalar quadratic form, so matrix products never
enter it. (Numerically confirmed on 1512 configurations with
$\lVert[\Sigma_x,\Sigma_u]\rVert_F>10^{-8}$ or $\lVert[M,\Sigma_x]\rVert_F>10^{-8}$:
max deviation $6.8\cdot10^{-9}$.)

**No non-singularity of $\Sigma_u$ is used:** the coefficient matrix is
$\Sigma_x+\Sigma_u+M$, positive definite from $\Sigma_x\succ0$ alone. (Numerically
confirmed on 648 configurations with rank-1 or zero $\Sigma_u$/$M$: max deviation
$6.8\cdot10^{-9}$.) Likewise $M$ may be rank-deficient, or zero (which reduces
(iii) to (i)).

### 2.7 The two scalar special cases

$p=1$, $\Sigma_x=1$, $\Sigma_u=\sigma_u^2$: $\bar\beta=\beta^\*/(1+\sigma_u^2)$ ✔.
$M=\Sigma_u$: $\beta^\dagger=\beta^\*/(1+2\sigma_u^2)$ ✔.

### 2.8 Minimal conditions

$$\Sigma_x+\Sigma_u+M\succ0,\qquad E[X]=0,\qquad E\lvert\varepsilon\rvert<\infty,
\qquad \varepsilon\perp(X,U).$$

"$\Sigma_x\succ0$" can be weakened to "$\Sigma_x\succeq0$ and
$\Sigma_x+\Sigma_u+M\succ0$"; "$\varepsilon$ has a density" is unnecessary.

---

## 3. Numerical strategy (three independent routes)

* **Route 1 (reduction).** Uses only (2.1) and evaluates
  $F(\beta)=\tau\mu_x'(\beta^\*-\beta)+\Lambda(\sqrt{q_M(\beta)+\sigma^2})$.
  $\Lambda$ is tabulated per error law from adaptive quadrature against the exact
  density (exactly, in closed form, for atomic laws), with a geometric ladder of
  break points around $0$ when $s$ is small, then interpolated by a clamped cubic
  spline using the exact $\Lambda'(0)=0$. **Measured accuracy vs. fresh adaptive
  quadrature: max $8.3\cdot10^{-10}$, typically $\sim10^{-11}$** (table §1).
* **Route 2 (direct, no reduction).** Tensor Gauss–Hermite quadrature over the
  model variables $(X,U)$ (through a PSD square root, so singular $\Sigma_u$
  works): $F(\beta)=\sum_n w_n J(t_n(\beta),\sigma(\beta))$,
  $t_n(\beta)=x_n'(\beta^\*-\beta)-u_n'\beta$, with
  $J(t,s)=E_\varepsilon[(-(\varepsilon+t-sZ))^{+}]$ from a density-based rule
  ($s>0$) or the exact cdf antiderivative ($s=0$).
* **Route 3 (Monte-Carlo).** Raw simulation; (a) the *exact* minimiser of the
  empirical check-loss criterion by a sparse LP (quantile regression, HiGHS) over
  5 seeds of $n=8000$; (b) the population subgradient at the claimed minimiser,
  $\hat g=n^{-1}\sum_iW_i(\tau-\mathbf 1\{Y_i-\hat\beta'W_i<0\})$ with $n=2\cdot10^6$,
  which must be statistically zero.

## 4. Counterexample search

### 4.1 What was computed

**7320** configurations in the broad sweep (route 1), **241** re-checked with the
reduction-free route 2, **46** with the Monte-Carlo routes, plus uniqueness,
stress and high-precision diagnostics. The axes crossed:

* $\tau\in\{0.1,0.3,0.5,0.7,0.9\}$;
* twelve error laws (§4.5);
* $\sigma_u\in\{0,0.1,0.5,1,2\}$ (scalar) and $\Sigma_u\in\{0,\;0.7I,\;$random PSD
  non-commuting,$\;$rank-1,$\;0.5\Sigma_x\}$ (multivariate);
* $p\in\{1,2,3,4\}$, $\Sigma_x$ diagonal / strongly correlated / ill-conditioned
  (cond 50);
* $M\in\{0,\;cI,\;$random PSD non-commuting,$\;$rank-1,$\;\Sigma_u\}$;
* $\sigma\in\{0,10^{-6},10^{-3},0.01,0.1,0.5,1,2,10,100\}$, plus
  $10^3\ldots10^{12}$ in 50-digit arithmetic;
* and $E[X]\ne0$ as a stress test.

### 4.2 Broad sweep (route 1)

| quantity | value |
|---|---|
| configurations | 7320 |
| median $\lVert\hat\beta-\beta_{\rm claim}\rVert_\infty$ | $0$ |
| 99th percentile | $2.5\cdot10^{-9}$ |
| max $\lVert\hat\beta-\beta_{\rm claim}\rVert_\infty$ | $0.99$ — only for the two degenerate error laws of §4.6 |
| configurations with deviation $<10^{-6}$ | 7259 / 7320 |
| the 61 exceptions | 52 × $\varepsilon\equiv10$, 9 × $\varepsilon=3+e$ (all with objective gap $=0$: ties) |
| max $F(\hat\beta)-F(\beta_{\rm claim})$ | $0$ |
| min $F(\hat\beta)-F(\beta_{\rm claim})$ | $-2.2\cdot10^{-16}$ (no configuration found a strictly better point) |

By block (ten non-degenerate laws / two degenerate laws):

| block | $n$ | max deviation (non-degenerate) | median | max (degenerate) |
|---|---|---|---|---|
| scalar (i)/(ii) | 3000 | $2.4\cdot10^{-9}$ | 0 | $0.64$ |
| scalar (iii), $M\in\{0,0.09,0.25,1,4\}$ | 1500 | $3.0\cdot10^{-9}$ | 0 | $0.99$ |
| scalar (iii), $M=\Sigma_u$ | 300 | $2.3\cdot10^{-9}$ | 0 | $0.85$ |
| matrix (ii), $p=2,3,4$ | 1620 | $6.8\cdot10^{-9}$ | 0 | — |
| matrix (iii), $p=2,3,4$ | 900 | $6.1\cdot10^{-9}$ | 0 | — |

By bandwidth (non-degenerate): $\sigma=0$: $6.1\cdot10^{-9}$; $10^{-6}$:
$4.8\cdot10^{-10}$; $10^{-3}$: $2.9\cdot10^{-10}$; $0.01$: $5.4\cdot10^{-10}$;
$0.1$: $7.0\cdot10^{-10}$; $0.5$: $6.8\cdot10^{-9}$; $1,10,100$: $0$; $2$:
$1.7\cdot10^{-9}$ — no trend in $\sigma$, consistent with (2.5). By $\tau$: maxima
between $2.3\cdot10^{-9}$ and $6.8\cdot10^{-9}$ for every $\tau$, i.e. no
$\tau$-dependence beyond numerical noise, consistent with §2.2.
Non-commuting subset (1512 configs): max $6.8\cdot10^{-9}$. Singular
$\Sigma_u$/$M$ subset (648): max $6.8\cdot10^{-9}$.

### 4.3 Reduction-free re-check (route 2)

241 configurations. For the *smoothed* criteria ($\sigma>0$: parts (ii) and (iii))
route 2 reproduces the claimed minimiser with
$\max\lVert\hat\beta-\beta_{\rm claim}\rVert_\infty$ of order
$10^{-9}\dots10^{-14}$, and the two routes agree on the criterion *value* at the
claim to $10^{-10}\dots10^{-16}$. This is an independent confirmation of the
reductions (2.3), (2.5), (2.6) — i.e. of the entire mathematical content of the
claim, since the rest is linear algebra.

For the non-smooth case $\sigma=0$, Gauss–Hermite integrates a kinked function and
converges only algebraically: the measured order-convergence is
$5.1\cdot10^{-6}$ ($n_{\rm gh}=10$), $8.3\cdot10^{-7}$ (20), $3.7\cdot10^{-7}$
(30), $-1.9\cdot10^{-7}$ (45). This produces apparent minimiser offsets of
$10^{-8}\dots10^{-3}$ in a handful of $\sigma=0$ configurations, while the claimed
point has the same objective value to $\sim10^{-14}$ — quadrature artefacts, not
counterexamples. At $\sigma=1$ the same comparison is $2\cdot10^{-10}$ at every
order.

### 4.4 Monte-Carlo routes

* **Exact empirical LP minimiser** (5 seeds × $n=8000$, 46 configurations): all
  deviations lie within Monte-Carlo error; the largest $|z|$ is 3.26 (a $p=2$
  configuration) and the deviations themselves are $|{\rm mean\ dev}|\le0.015$.
* **Population subgradient at the claim** ($n=2\cdot10^6$): largest $|z|=3.03$
  over all components and configurations (typical $|z|<2$). There is no
  systematic deviation — exactly what the claim predicts.
* **Smoothed empirical fits** (L-BFGS, $n=2\cdot10^5$): deviations
  $-3.7\cdot10^{-13}$ (Gaussian, $\sigma=1$), $-6.4\cdot10^{-14}$ (Laplace,
  $\sigma=0.3$), $-1.1\cdot10^{-14}$ ((iii), Gaussian), $1.7\cdot10^{-14}$ ((iii),
  $p=2$, non-commuting $M$) — and $6\cdot10^{-4}$/$3\cdot10^{-4}$ for $t_3$ and the
  bimodal law, which is the Monte-Carlo noise of the empirical minimiser at
  $n=2\cdot10^5$.

### 4.5 The twelve error laws used

standard Gaussian; Laplace (var 1); asymmetric Laplace ($\kappa=3$); uniform;
centred exponential ($e-1$, mass touching 0); shifted exponential ($3+e$, mass
far from 0); Student $t_3$ (heavy tail); skew-normal $\alpha=8$; split normal
($\sigma_-=1,\sigma_+=4$); two-component Gaussian mixture
$\tfrac12N(-2.5,.5^2)+\tfrac12N(2.5,.5^2)$; purely atomic $0.4\delta_{-1}+0.6\delta_{+2}$
(**no density**); and degenerate $\varepsilon\equiv10$ (**no density**).

### 4.6 Uniqueness

* $\Lambda$ monotone: the largest negative increment of the tabulated $\Lambda$ on
  a $3\cdot10^5$-point grid over $[0,30]$ is $5.6\cdot10^{-17}$ — numerical zero —
  for all twelve laws.
* Dense scalar grid scans ($\pm8$, down to step $0.001$): 33 "failures", all of
  them with $\varepsilon\equiv10$, $\varepsilon=3+e$ or the atomic law, i.e. the
  degenerate cases where the criterion is flat to $10^{-200}$ or below (§8). For
  every other law the global grid minimum coincides with the claim.
* 200 random line scans through the claim for $p=2,3,4$ for each of (i), (ii),
  (iii): **0 violations**.
* Numerical Hessian at the claim for $p=2,3,4$: positive definite in all cases.

### 4.7 Where the large numbers come from (and why they are not counterexamples)

If the error law is bounded away from $0$ (or atomic), the criterion's slope
$\Lambda'(s)=E[\varphi(\varepsilon/s)]$ is $\exp(-\Theta(1/s^2))$-small: for
$\varepsilon\equiv10$ at $s=0.01$ the criterion
increases by $2.3\cdot10^{-217156}$ when $\beta$ moves by $0.01$. In double
precision such a criterion is *identical* over a whole ellipsoid, so any
gradient-based optimiser stops at its starting point and reports an arbitrary
$\hat\beta$ — while `gap = F(beta_num) - F(beta_claim) = 0` shows that it found a
*tie*, not a better point. Redone in 300-digit arithmetic (§8) the criterion is
strictly increasing on both sides of the claim and the high-precision grid argmin
is the claim exactly. The mathematical statement is intact; what fails is the
*identifiability* of the population minimiser from any finite-precision
computation. That is a real practical caveat, and it is not visible in the
statement.

## 5. What is stated too strongly / too weakly

* **Too weak: "$\varepsilon$ has a density".** Never used. Verified with atomic and
  degenerate $\varepsilon$ (including $\varepsilon\equiv10$).
* **Too weak: "$\Sigma_x\succ0$".** The proof only needs
  $\Sigma_x+\Sigma_u+M\succ0$.
* **Too weak: "fix $\tau\in(0,1)$".** The $\tau$-dependence cancels identically
  (§2.2), so the result is uniform in $\tau$ — and in fact holds for any loss of
  the form $a r+(-r)^{+}$.
* **Too strong in spirit: "for every $\sigma>0$ … the bandwidth does not move the
  target".** True, but the criterion's contrast is
  $\frac{\varphi(0)}{2\sigma}\big(q_0(\beta)-q_0(\bar\beta)\big)+O(\sigma^{-3})$:
  the *value* at the optimum does move, and for $\sigma\gg1$ the minimiser is
  numerically undecidable in double precision even though it is exactly
  $\bar\beta$ (verified up to $\sigma=10^{12}$ in 50-digit arithmetic, §7). A
  finite-sample or float64 fit will *not* appear bandwidth-invariant.
* **Too strong: "unique minimiser".** True only because $\Sigma_x+\Sigma_u+M\succ0$
  (which the statement's $\Sigma_x\succ0$ supplies). Relax $\Sigma_x$ to PSD and
  uniqueness silently disappears (§6b). And when $\Lambda$ underflows, uniqueness
  is vacuous in practice (§4.7).
* **Under-stated: $E[X]=0$.** It is used essentially (it is what kills the
  $\tau$-term), and it is easy to lose: with $E[X]=\mu_x\ne0$ the criterion
  becomes $E[\rho_\tau(\mu_x'(\beta^\*-\beta)+\varepsilon+\xi(\beta))]$, which is
  not a function of $q_0$, and $\bar\beta$ is *not* the minimiser (§6a).

## 6. Stress tests

### (a) $E[X]\ne0$ — the claim fails, as it must

| $p$ | $\mu_x$ | $\varepsilon$ | $\tau$ | $\beta_{\rm claim}$ | $\beta_{\rm numeric}$ (route 2) | deviation | $F(\text{claim})-F(\text{min})$ |
|---|---|---|---|---|---|---|---|
| 1 | (1.0) | Gaussian | 0.5 | 0.8000 | 0.8891 | 0.089 | $+3.2\cdot10^{-3}$ |
| 1 | (0.3) | Gaussian | 0.5 | 0.8000 | 0.8134 | 0.013 | $+4.4\cdot10^{-5}$ |
| 1 | (2.0) | Laplace | 0.5 | 0.8000 | 0.9525 | 0.153 | $+2.7\cdot10^{-2}$ |
| 1 | (1.0), $\Sigma_u=0$ | Gaussian | 0.5 | 1.0000 | 1.0000 | $2.7\cdot10^{-9}$ | $+4.8\cdot10^{-15}$ |
| 1 | (1.0), $\Sigma_u=0$ | Gaussian | 0.3 | 1.0000 | 0.7350 | 0.265 | $+2.6\cdot10^{-2}$ |
| 1 | (1.0), $\Sigma_u=0.25$ | Gaussian | 0.3 | 0.8000 | 0.6335 | 0.167 | $+1.0\cdot10^{-2}$ |
| 2 | (0.8,−0.6) | Gaussian | 0.5 | (0.756,0.435) | (0.821,0.374) | 0.065 | $+2.4\cdot10^{-3}$ |

The single row with (numerically) zero deviation is an accident worth naming: with
$\Sigma_u=0$ the derivative of the criterion at $\beta^\*$ is
$-\mu_x\big(\tau-F_\varepsilon(0)\big)$, which vanishes exactly when $\tau$ equals
the error median — the row with $\tau=0.3$ shows that the claim fails as soon as
that coincidence is broken.

### (b) Singular $\Sigma_x$ — uniqueness fails, as expected

$\Sigma_x=\mathrm{diag}(1,0)$, $\beta^\*=(1,0)$. With $\Sigma_u=0$ the criterion is
*exactly constant* along $e_2$ (values `0.398942` at $\beta_2\in\{-5,-1,0,1,5\}$,
spread $0.0$): the minimiser is an affine set, $\bar\beta$ is only one of its
points, and "unique" is false. With $\Sigma_u=0.5I$ the criterion is again strictly
convex (values $1.484,0.540,0.461,0.540,1.484$) and the claim — in its sharpened
form — holds.

### (c) $E\lvert\varepsilon\rvert=\infty$

Cauchy errors: the sample criterion means diverge ($3.64,3.56,3.55,\dots$ at
$n=4\cdot10^5$) because the population value is $+\infty$ for every $\beta$; there
is no minimiser at all. The moment assumption is doing real work; the density
assumption is not.

### (d) Indefinite $M$

For $M=\mathrm{diag}(1,-0.4)$ and $b=(0,1)$ we get $b'Mb=-0.4<0$, so
$\sqrt{\beta'M\beta}$ is not real and (iii)'s criterion is undefined. The
statement's $M\succeq0$ is exactly the needed condition.

## 7. Huge bandwidths (50-digit arithmetic)

Scalar model $\Sigma_x=1$, $\Sigma_u=1/4$, $\beta^\*=1$, so the claim is
$\bar\beta=0.8$. With $R_\sigma(\beta)=\Lambda(\sqrt{(1-\beta)^2+\beta^2/4+\sigma^2})$:

| $\sigma$ | $R_\sigma(\bar\beta)$ | $\Delta R$ at $\bar\beta\pm10^{-4}$ | $\Delta R$ at $\bar\beta\pm0.5$ | curvature |
|---|---|---|---|---|
| $10^3$ | 398.942519766729109 | $+2.49\cdot10^{-12}$ | $+6.23\cdot10^{-5}$ | $4.987\cdot10^{-4}$ |
| $10^6$ | 398942.280401672043 | $+2.49\cdot10^{-15}$ | $+6.23\cdot10^{-8}$ | $4.987\cdot10^{-7}$ |
| $10^9$ | 398942280.401432678 | $+2.49\cdot10^{-18}$ | $+6.23\cdot10^{-11}$ | $4.987\cdot10^{-10}$ |
| $10^{12}$ | 398942280401.432678 | $+2.49\cdot10^{-21}$ | $+6.23\cdot10^{-14}$ | $4.987\cdot10^{-13}$ |

(Gaussian and Laplace errors give identical entries to the printed digits; the
difference is $O(\sigma^{-2})$.) All increments are strictly positive and
symmetric, the curvature at the claim is positive, and it decays exactly like
$\varphi(0)/(2\sigma)$ — a factor $10^{-3}$ per factor $10^3$ in $\sigma$. So (ii)
holds at every bandwidth, but the numerical content of "the minimiser is
$\bar\beta$" evaporates as $\sigma\to\infty$.

## 8. The degenerate error laws in 300-digit arithmetic

For $\varepsilon\equiv10$ the criterion at the claim is as small as
$6.9\cdot10^{-91}$ ($\sigma=0.5$, $\sigma_u=0$) and the increments away from it are
e.g. $+5.75\cdot10^{-92}$ at $\pm0.01$ — strictly positive, symmetric, and
therefore a genuine minimum. The 300-digit grid argmin is the claim exactly
(1.0, 0.8, 0.5 for $\sigma_u=0,0.5,1$) in all six configurations tested, including
the two where double precision sees the criterion as the constant $0.0$ and the
increment as $0$ instead of $2.29\cdot10^{-217156}$.

For the atomic law $0.4\delta_{-1}+0.6\delta_{+2}$ the values are ordinary
($R(\bar\beta)=0.4,\,0.400789,\,0.410344$ for $\sigma_u=0,0.5,1$) but the slope is
$\Lambda'(s)=E[\varphi(\varepsilon/s)]\approx10^{-217}$ at $s=10^{-3}$: strict
monotonicity is real, provable and completely unobservable. This is the sharpest
illustration of §4.7: for such error laws the *claim* is true and the *estimator*
is not numerically identifiable.

## 9. Limitations of this check

* $p\le4$ in the multivariate blocks (nothing in the derivation is
  dimension-dependent, but $p\ge5$ was not run).
* Route 2 agrees with route 1 to $\sim10^{-10}$ for $\sigma>0$; at $\sigma=0$ its
  Gauss–Hermite error is $10^{-7}\dots10^{-5}$, so the independent evidence for
  $\sigma=0$ rests on route 3 (exact LP minimiser and the $n=2\cdot10^6$
  subgradient test) rather than on route 2.
* The LP replicates use $n=8000$, so they detect only deviations above
  $\sim10^{-2}$; the sharp Monte-Carlo instrument is the subgradient test.
* The statement is about population criteria; no finite-sample claim was tested.
* The "too strong in spirit" remarks in §5 are about numerical identifiability,
  not about the truth of the stated mathematical claim.

# Mathematical Background

This document derives the key causal, statistical, and fairness concepts used in the project. Written for an audience with a pure mathematics background (measure theory, probability, linear algebra).

---

## 1. Causal Inference

### Potential Outcomes Framework (Rubin)

For each user $i$, let $Y_i(1)$ be the revenue if treated (credit limit increase) and $Y_i(0)$ if not treated. The individual treatment effect is $\tau_i = Y_i(1) - Y_i(0)$. The **Average Treatment Effect** (ATE) is $\tau = \mathbb{E}[\tau_i]$.

We observe only one potential outcome: $Y_i = T_i Y_i(1) + (1-T_i)Y_i(0)$.

**Selection bias**:
$$
\mathbb{E}[Y_i|T_i=1] - \mathbb{E}[Y_i|T_i=0] = \underbrace{\mathbb{E}[Y_i(1)-Y_i(0)|T_i=1]}_{\text{ATT}} + \underbrace{\mathbb{E}[Y_i(0)|T_i=1] - \mathbb{E}[Y_i(0)|T_i=0]}_{\text{bias}}
$$
The bias term is non‑zero if confounders affect both treatment and outcome.

### Propensity Score & IPW

Define $e(X) = P(T=1|X)$. Under unconfoundedness ($Y(1),Y(0) \perp T | X$), the IPW estimator is:
$$
\hat{\tau}_{\text{IPW}} = \frac{1}{n}\sum_{i=1}^n \left( \frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i)Y_i}{1-e(X_i)} \right)
$$
Taking expectation:
$$
\mathbb{E}\left[\frac{T Y}{e(X)}\right] = \mathbb{E}\left[\mathbb{E}\left[\frac{T Y}{e(X)}\middle|X\right]\right] = \mathbb{E}\left[\frac{e(X) \mathbb{E}[Y(1)|X]}{e(X)}\right] = \mathbb{E}[Y(1)]
$$
Similarly $\mathbb{E}\left[\frac{(1-T)Y}{1-e(X)}\right] = \mathbb{E}[Y(0)]$. Hence IPW recovers ATE.

### Double Machine Learning (DML)

DML uses Neyman‑orthogonal scores to reduce bias from regularised nuisance estimators. Let $g(X)=\mathbb{E}[Y|X]$ and $m(X)=\mathbb{E}[T|X]$. The orthogonal score is:
$$
\psi(Y,T,X;\theta) = (Y - g(X) - \theta (T - m(X))) \cdot (T - m(X))
$$
The moment condition $\mathbb{E}[\psi] = 0$ yields the ATE $\theta$.

We estimate $\hat{g}$ and $\hat{m}$ via cross‑fitting to avoid overfitting bias. The final estimator is:
$$
\hat{\theta} = \frac{\sum_{i=1}^n (Y_i - \hat{g}_{-i}(X_i))(T_i - \hat{m}_{-i}(X_i))}{\sum_{i=1}^n (T_i - \hat{m}_{-i}(X_i))^2}
$$
where $\hat{g}_{-i}$ is trained on all data except fold $i$.

---

## 2. Difference‑in‑Differences (DiD)

Assumes parallel trends: in the absence of treatment, the difference between treatment and control groups would remain constant over time.

Let $Y_{i,t}$ be revenue, $Post_t$ indicator for post‑treatment, $Treat_i$ indicator for treatment group. DiD estimator:
$$
\delta = \mathbb{E}[Y_{i,t}|Treat=1, Post=1] - \mathbb{E}[Y_{i,t}|Treat=1, Post=0] \\
\quad - \left( \mathbb{E}[Y_{i,t}|Treat=0, Post=1] - \mathbb{E}[Y_{i,t}|Treat=0, Post=0] \right)
$$
We test parallel trends using a placebo test on pre‑treatment periods.

---

## 3. Sequential Probability Ratio Test (SPRT)

Let $H_0: \theta = 0$ (no effect) and $H_1: \theta = \delta$ (minimum detectable effect). After $n$ observations, the likelihood ratio is:
$$
\Lambda_n = \frac{L(\delta)}{L(0)}
$$
SPRT stops at the first $n$ where $\Lambda_n \ge A$ (reject $H_0$) or $\Lambda_n \le B$ (accept $H_0$), with:
$$
A = \frac{1-\beta}{\alpha}, \quad B = \frac{\beta}{1-\alpha}
$$
$\alpha$ is Type I error, $\beta$ Type II error. This controls error rates at any stopping time.

---

## 4. Fairness: Equalized Odds

A predictor $\hat{Y}$ satisfies equalized odds if:
$$
P(\hat{Y}=1 | Y=y, A=0) = P(\hat{Y}=1 | Y=y, A=1) \quad \forall y \in \{0,1\}
$$
where $A$ is a protected attribute (e.g., income group). This ensures equal True Positive Rate (TPR) and False Positive Rate (FPR) across groups.

**Impossibility Theorem (Chouldechova, 2017)**: For a binary predictor, one cannot simultaneously satisfy equalized odds, calibration ( $\hat{Y}=p$ implies $P(Y=1|\hat{Y}=p)=p$ ), and positive predictive value parity unless base rates are equal. We prioritise equalized odds because it directly addresses disparate impact in credit decisions.

We mitigate by adjusting the decision threshold per group to equalise TPR.

---

## 5. Power Analysis

For a two‑sample t‑test with significance $\alpha$, power $1-\beta$, and effect size $d = \frac{\text{MDE}}{\sigma}$, the required sample size per group is:
$$
n = \frac{2 (z_{1-\alpha/2} + z_{1-\beta})^2}{d^2}
$$
where $z$ is the standard normal quantile. Our power curve visualises this relationship.

---

## References

- Rubin (1974) – “Estimating causal effects of treatments”
- Rosenbaum & Rubin (1983) – “The central role of the propensity score”
- Chernozhukov et al. (2016) – “Double/debiased machine learning”
- Johari et al. (2017) – “Always valid inference”
- Chouldechova (2017) – “Fair prediction with disparate impact”
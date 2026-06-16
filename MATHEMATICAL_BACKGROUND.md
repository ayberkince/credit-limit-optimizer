# Mathematical Background

This document derives the key mathematical concepts used in the Credit Limit Optimizer project. Written for an audience with a background in probability, statistics, and linear algebra.

---

## Table of Contents

1. [Causal Inference Framework](#1-causal-inference-framework)
2. [Propensity Score & Inverse Probability Weighting](#2-propensity-score--inverse-probability-weighting)
3. [Double Machine Learning](#3-double-machine-learning)
4. [Difference-in-Differences](#4-difference-in-differences)
5. [Sequential Probability Ratio Test](#5-sequential-probability-ratio-test)
6. [Survival Analysis (Cox PH)](#6-survival-analysis-cox-ph)
7. [Fairness: Equalized Odds](#7-fairness-equalized-odds)
8. [Rosenbaum Sensitivity Analysis](#8-rosenbaum-sensitivity-analysis)
9. [Population Stability Index (Drift Detection)](#9-population-stability-index-drift-detection)
10. [Net Incremental Value (Policy Decision)](#10-net-incremental-value-policy-decision)
11. [Summary of Methods](#11-summary-of-methods)

---

## 1. Causal Inference Framework

### Potential Outcomes (Rubin Causal Model)

For each user $i$, let:

- $Y_i(1)$ = revenue if **given** a credit limit increase
- $Y_i(0)$ = revenue if **not given** the increase

The **Individual Treatment Effect** is:

$$\tau_i = Y_i(1) - Y_i(0)$$

The **Average Treatment Effect (ATE)** is:

$$\tau = \mathbb{E}[\tau_i] = \mathbb{E}[Y_i(1) - Y_i(0)]$$

### The Fundamental Problem of Causal Inference

We never observe both potential outcomes for the same user. We only see:

$$Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)$$

where $T_i \in \{0,1\}$ is the treatment indicator.

### Selection Bias

If we simply compare treated vs control group means:

$$\mathbb{E}[Y_i | T_i=1] - \mathbb{E}[Y_i | T_i=0] = \underbrace{\mathbb{E}[Y_i(1)-Y_i(0)|T_i=1]}_{\text{ATT}} + \underbrace{\bigl(\mathbb{E}[Y_i(0)|T_i=1] - \mathbb{E}[Y_i(0)|T_i=0]\bigr)}_{\text{selection bias}}$$

The second term is **non-zero** when confounders (e.g., income, credit score) affect both treatment assignment and outcome.

### Unconfoundedness (Ignorability)

We assume that conditional on observed covariates $X$:

$$Y(1), Y(0) \perp T \mid X$$

This means that within strata of $X$, treatment is as good as random. This is the core assumption enabling causal identification.

---

## 2. Propensity Score & Inverse Probability Weighting

### Propensity Score

The **propensity score** is the probability of treatment given covariates:

$$e(X) = P(T=1 \mid X)$$

**Balancing Property** (Rosenbaum & Rubin, 1983):

$$X \perp T \mid e(X)$$

This means that within strata of the propensity score, treated and control users have the same distribution of covariates.

### Inverse Probability Weighting (IPW) Estimator

The IPW estimator reweights each observation by the inverse of its probability of receiving the treatment it actually received:

$$\hat{\tau}_{\text{IPW}} = \frac{1}{n}\sum_{i=1}^n \left( \frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i)Y_i}{1-e(X_i)} \right)$$

**Why this works:** Taking expectation:

$$\mathbb{E}\left[\frac{T Y}{e(X)}\right] = \mathbb{E}\left[\mathbb{E}\left[\frac{T Y}{e(X)} \middle| X\right]\right] = \mathbb{E}\left[\frac{e(X) \mathbb{E}[Y(1)|X]}{e(X)}\right] = \mathbb{E}[Y(1)]$$

Similarly, $\mathbb{E}\left[\frac{(1-T)Y}{1-e(X)}\right] = \mathbb{E}[Y(0)]$.

Therefore the difference is the ATE:

$$\mathbb{E}[\hat{\tau}_{\text{IPW}}] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)] = \tau$$

**Implementation:** Propensity scores are estimated using logistic regression. Extreme weights are clipped to $[0.025, 0.975]$ to avoid instability.

---

## 3. Double Machine Learning

### Nuisance Functions

Define:

- $g(X) = \mathbb{E}[Y \mid X]$ — outcome model
- $m(X) = \mathbb{E}[T \mid X]$ — treatment model (same as propensity score)

### Neyman-Orthogonal Score

The orthogonal score is:

$$\psi(Y,T,X;\theta) = \bigl(Y - g(X) - \theta \cdot (T - m(X))\bigr) \cdot (T - m(X))$$

where $\theta$ is the ATE.

**Key Property (Neyman Orthogonality):**

$$\frac{\partial}{\partial \eta} \mathbb{E}[\psi] \bigg|_{\eta = \eta_0} = 0$$

where $\eta = (g, m)$ are the nuisance functions. This means that **first-order errors in estimating $g$ and $m$ do not bias $\theta$** — only second-order errors remain, which vanish faster as sample size grows.

### Cross-Fitting Procedure

1. Split data into $K$ folds.
2. For each fold $k$:
   - Train $\hat{g}$ and $\hat{m}$ on all other folds.
   - Predict on fold $k$: $\hat{Y}_i = \hat{g}_{-k}(X_i)$, $\hat{T}_i = \hat{m}_{-k}(X_i)$.
   - Compute residuals: $\tilde{Y}_i = Y_i - \hat{Y}_i$, $\tilde{T}_i = T_i - \hat{T}_i$.
3. Combine all residuals and estimate:

$$\hat{\theta}_{\text{DML}} = \frac{\sum_i \tilde{Y}_i \tilde{T}_i}{\sum_i \tilde{T}_i^2}$$

This is the **Double Machine Learning (DML)** estimator. It is consistent and asymptotically normal even when $\hat{g}$ and $\hat{m}$ are learned using high-dimensional or non-parametric methods.

**Why we use it:** DML is robust to model misspecification and can handle complex non-linear relationships via gradient boosting.

---

## 4. Difference-in-Differences

### Setup

We have panel data: repeated observations of the same users over time.

Let:

- $Y_{it}$ = revenue for user $i$ at time $t$
- $T_i$ = treatment group indicator (1 if treated, 0 if control)
- $Post_t$ = indicator for post-treatment period

### DiD Estimator

$$\delta_{\text{DiD}} = \bigl(\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}\bigr) - \bigl(\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}}\bigr)$$

This compares the **change** in revenue for the treatment group to the **change** in the control group.

### Parallel Trends Assumption

The key identifying assumption is:

> In the absence of treatment, the treatment and control groups would have followed parallel trends.

This is **testable** using a placebo test: we pretend treatment happened at an earlier time and check if the DiD estimate is significantly different from zero.

**Statistical test:** We bootstrap the placebo DiD effect and test if it is significantly different from zero ($p > 0.05$ suggests parallel trends holds).

---

## 5. Sequential Probability Ratio Test

### Problem with Fixed-Sample Tests

In classical A/B testing, you cannot "peek" at results without inflating the Type I error rate. This is the **multiple testing problem**.

### SPRT Framework

Let:

- $H_0: \theta = 0$ (no treatment effect)
- $H_1: \theta = \delta$ (minimum detectable effect)

After $n$ observations, the **likelihood ratio** is:

$$\Lambda_n = \frac{L(\delta \mid \text{data})}{L(0 \mid \text{data})}$$

### Decision Boundaries

Stop at the first $n$ where:

- If $\Lambda_n \ge A = \frac{1-\beta}{\alpha}$ → reject $H_0$ (treatment works)
- If $\Lambda_n \le B = \frac{\beta}{1-\alpha}$ → accept $H_0$ (no effect)
- Otherwise → continue sampling

where:
- $\alpha$ = Type I error (false positive rate)
- $\beta$ = Type II error (false negative rate)

### Error Control

SPRT guarantees:

$$P(\text{reject } H_0 \mid H_0) \le \alpha$$
$$P(\text{accept } H_0 \mid H_1) \le \beta$$

**Important:** The test can be stopped at any time without inflating error rates. This is the key advantage over fixed-sample tests.

---

## 6. Survival Analysis (Cox PH)

### Survival Function

For each user, let $T$ be the time (in months) until default. The **survival function** is:

$$S(t) = P(T > t)$$

### Hazard Function

The **hazard function** is the instantaneous risk of default at time $t$:

$$\lambda(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t \mid T \ge t)}{\Delta t}$$

### Cox Proportional Hazards Model

The Cox model assumes:

$$\lambda(t \mid X) = \lambda_0(t) \exp(\beta^T X)$$

where:
- $\lambda_0(t)$ is the **baseline hazard** (unspecified)
- $\beta$ is the vector of coefficients
- $X$ are covariates (treatment, income, credit score)

### Hazard Ratio

The **hazard ratio** for treatment vs control is:

$$\text{HR} = \frac{\lambda(t \mid T=1)}{\lambda(t \mid T=0)} = \exp(\beta_{\text{treatment}})$$

- If $\text{HR} > 1$, treatment increases default risk.
- If $\text{HR} < 1$, treatment decreases default risk.

### Partial Likelihood

The Cox model estimates $\beta$ by maximizing the **partial likelihood**:

$$L(\beta) = \prod_{i: \text{event}_i=1} \frac{\exp(\beta^T X_i)}{\sum_{j: t_j \ge t_i} \exp(\beta^T X_j)}$$

This allows inference without specifying the baseline hazard.

---

## 7. Fairness: Equalized Odds

### Definition

A predictor $\hat{Y}$ satisfies **equalized odds** if:

$$P(\hat{Y}=1 \mid Y=y, A=a) = P(\hat{Y}=1 \mid Y=y, A=b)$$

for all $y \in \{0,1\}$ and all protected groups $a, b$.

This means the **True Positive Rate** (TPR) and **False Positive Rate** (FPR) are equal across groups:

$$\text{TPR}_a = \text{TPR}_b \quad \text{and} \quad \text{FPR}_a = \text{FPR}_b$$

### Why We Use It

Equalized odds directly addresses **disparate impact** in lending decisions. It ensures that users with the same actual risk are treated similarly, regardless of their income group.

### Mitigation Strategy

We adjust the decision threshold per group to equalize TPR:

$$\tau_g = \arg\min_{\tau} \left| \text{TPR}_g(\tau) - \text{TPR}_{\text{target}} \right|$$

This is implemented via grid search over $\tau \in [0,1]$.

### Impossibility Theorem (Chouldechova, 2017)

It is **impossible** to simultaneously satisfy equalized odds, calibration, and positive predictive value parity unless base rates are identical across groups. We prioritise equalized odds as it most directly addresses discriminatory lending.

---

## 8. Rosenbaum Sensitivity Analysis

### Purpose

Quantify how strong an unmeasured confounder would need to be to overturn our causal conclusion.

### Gamma Parameter

Let $\Gamma$ be the odds ratio of treatment assignment between two matched individuals with identical observed covariates but different unmeasured confounders:

$$\Gamma = \frac{\text{Odds}(T=1 \mid X, U=1)}{\text{Odds}(T=1 \mid X, U=0)}$$

- $\Gamma = 1$: No unmeasured confounding
- $\Gamma > 1$: Unmeasured confounding exists

### Interpretation

The critical gamma $\Gamma_{\text{critical}}$ is the value at which the p-value crosses 0.05:

- $\Gamma_{\text{critical}} < 1.5$: Highly sensitive to unmeasured confounding
- $1.5 < \Gamma_{\text{critical}} < 2.0$: Moderately robust
- $\Gamma_{\text{critical}} > 2.0$: Highly robust

### Implementation

We approximate the Rosenbaum bounds using the t-test p-value and a logistic transformation:

$$p_{\text{inflated}} = \frac{\text{odds}_{\text{obs}}}{1 + \Gamma \cdot \text{odds}_{\text{obs}}}$$

where $\text{odds}_{\text{obs}} = \frac{p_{\text{obs}}}{1 - p_{\text{obs}}}$.

---

## 9. Population Stability Index (Drift Detection)

### Purpose

Monitor whether the distribution of input features shifts over time, which can degrade model performance.

### Formula

For a given feature $X$, the **Population Stability Index (PSI)** is:

$$\text{PSI} = \sum_{b=1}^{B} \left( p_{\text{current}, b} - p_{\text{ref}, b} \right) \cdot \ln\left(\frac{p_{\text{current}, b}}{p_{\text{ref}, b}}\right)$$

where:
- $B$ = number of bins
- $p_{\text{ref}, b}$ = proportion of reference sample in bin $b$
- $p_{\text{current}, b}$ = proportion of current sample in bin $b$

### Interpretation

| PSI Range | Interpretation |
|-----------|----------------|
| $< 0.1$ | No significant drift |
| $0.1 - 0.2$ | Moderate drift (investigate) |
| $> 0.2$ | Significant drift (retrain model) |

---

## 10. Net Incremental Value (Policy Decision)

### Definition

The **Net Incremental Value (NIV)** for treating a user with treatment $t$ is:

$$\text{NIV}(t) = \underbrace{\text{CATE}_t}_{\text{gross benefit}} - \underbrace{\Delta_{\text{default}} \times \text{loss}}_{\text{expected default loss}} - \underbrace{\text{cost}_t}_{\text{intervention cost}}$$

where:
- $\text{CATE}_t$ = Conditional Average Treatment Effect for the user's income quintile
- $\Delta_{\text{default}}$ = increase in default probability due to treatment
- $\text{loss}$ = average loss per default (e.g., $500)
- $\text{cost}_t$ = cost of implementing the treatment

### Decision Rule

Treat user $i$ if:

$$\text{NIV}_i > 0$$

For multi-product uplift, choose the treatment with the highest NIV:

$$t_i^* = \arg\max_{t \in \mathcal{T}} \text{NIV}_i(t)$$

---

## 11. Summary of Methods

| Method | What It Does | Key Assumption | Formula |
|--------|--------------|----------------|---------|
| **IPW** | Removes confounding | Unconfoundedness | $\hat{\tau} = \frac{1}{n}\sum \left(\frac{T Y}{e(X)} - \frac{(1-T)Y}{1-e(X)}\right)$ |
| **DML** | Robust to ML misspecification | Neyman orthogonality | $\hat{\theta} = \frac{\sum \tilde{Y}\tilde{T}}{\sum \tilde{T}^2}$ |
| **DiD** | Uses time as control | Parallel trends | $\delta = (Y_{T,post} - Y_{T,pre}) - (Y_{C,post} - Y_{C,pre})$ |
| **SPRT** | Sequential testing | Known variance | $\Lambda_n = \frac{L(\delta)}{L(0)}$, stop at $A$ or $B$ |
| **Cox PH** | Default timing | Proportional hazards | $\lambda(t\|X) = \lambda_0(t) \exp(\beta^T X)$ |
| **Equalized Odds** | Fairness | Group parity | $\text{TPR}_a = \text{TPR}_b$, $\text{FPR}_a = \text{FPR}_b$ |
| **Rosenbaum** | Sensitivity | None (detects violation) | $\Gamma = \frac{\text{Odds}(T\|X,U=1)}{\text{Odds}(T\|X,U=0)}$ |
| **PSI** | Drift detection | None (detects shift) | $\text{PSI} = \sum (p_{\text{cur}} - p_{\text{ref}}) \ln\left(\frac{p_{\text{cur}}}{p_{\text{ref}}}\right)$ |
| **NIV** | Business decision | Cost-benefit | $\text{NIV} = \text{CATE} - \Delta_{\text{default}} \cdot \text{loss} - \text{cost}$ |

---

## References

1. Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*.

2. Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*.

3. Chernozhukov, V., et al. (2016). Double/debiased machine learning for treatment and causal parameters. *Econometrics Journal*.

4. Johari, R., et al. (2017). Always valid inference: Bringing sequential analysis to A/B testing. *arXiv*.

5. Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big Data*.

6. Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society*.

7. Rosenbaum, P. R. (2002). *Observational Studies*. Springer.

8. European Commission. (2024). EU Artificial Intelligence Act. *Regulation (EU) 2024/1689*.
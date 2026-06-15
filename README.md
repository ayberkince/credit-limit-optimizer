# Credit Limit Optimizer: Causal AI + A/B Testing + Fairness

**A production‑ready fintech project demonstrating causal inference, sequential experimentation, and algorithmic fairness.**

> *"Naive OLS overestimated the treatment effect by more than 15× – our causal correction prevents over‑allocating credit to high‑risk users."*

---

## 🎯 Problem

A fintech wants to increase credit limits for existing users to boost revenue (swipe fees + interest) while managing default risk. Historical data is **confounded**: wealthier users receive higher limits *and* spend more – naive estimates are severely biased.

We build a system that:
- Estimates the **causal effect** of a credit limit increase on net revenue.
- Designs an **A/B test** with sequential stopping (SPRT).
- Audits the policy for **fairness** across income groups.

---

## 🧠 Methodology

| Component | Technique |
|-----------|-----------|
| **Causal inference** | Inverse Probability Weighting (IPW) + Double Machine Learning (DML) + Difference‑in‑Differences (DiD) |
| **Experiment design** | Power analysis (5% MDE, 80% power) + Sequential Probability Ratio Test (SPRT) |
| **Fairness** | Equalized odds with bootstrap CIs & mitigation via per‑group thresholds |
| **Synthetic data** | Calibrated to Kaggle Credit Risk dataset, with known ground truth |

---

## 📊 Key Results

| Estimator | ATE (95% CI) | Interpretation |
|-----------|--------------|----------------|
| **True ATE (DGP)** | $80.00 | Ground truth (known from simulation) |
| **Naive OLS** | $1,221.86 | Severely biased due to confounding |
| **IPW** | $44.09 (-$12.75 – $93.22) | Adjusts for observed confounders, CI contains true effect |
| **Double ML** | $81.65 ($81.13 – $82.22) | Neyman‑orthogonal, robust to non‑linearities |
| **DiD** | $79.72 ($79.03 – $80.37) | Parallel trends satisfied (p=0.802) |

**Fairness Audit** (Equalized Odds):
- Initial TPR disparity: **0.497** (violation)
- Mitigated TPR disparity: **0.344** (improved)

> The mitigation reduces disparity while maintaining reasonable predictive performance (AUC = 0.617).

---

## 🚀 How to Run

```bash
git clone https://github.com/ayberkince/credit-limit-optimizer.git
cd credit-limit-optimizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full analysis (terminal)
python main.py

# Launch interactive dashboard
streamlit run src/dashboard/streamlit_app.py
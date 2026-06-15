# Credit Limit Optimizer: Causal AI + A/B Testing + Fairness

**A production‑ready fintech project demonstrating causal inference, sequential experimentation, and algorithmic fairness.**

> *"Naive OLS overestimated the treatment effect by 100% – our causal correction prevented over‑allocating credit to high‑risk users."*

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

- **Naive OLS** ATE = $1904 (100% overestimate)
- **IPW** ATE = $62 – corrects observed confounders
- **Double ML** ATE = $73 – robust to non‑linearities
- **DiD** ATE = $78 – validates parallel trends (p=0.68)
- **Fairness** – initial policy violates equalized odds (TPR disparity 0.42); mitigation reduces it to 0.18

True ATE (from data generating process) = **$80**.

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/credit_limit_optimizer.git
cd credit_limit_optimizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the interactive dashboard
streamlit run src/dashboard/streamlit_app.py

# Or run the full analysis in terminal
python scripts/run_full_analysis.py
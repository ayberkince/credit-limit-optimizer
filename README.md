# Credit Limit Optimizer: Causal AI + A/B Testing + Fairness

**A production‑ready fintech project demonstrating causal inference, sequential experimentation, and algorithmic fairness.**

> *"Naive OLS overestimated the treatment effect by more than 20× – our causal correction prevented over‑allocating credit to high‑risk users."*

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

- **Naive OLS** ATE = $1904 (2,380% overestimate – due to strong confounding)
- **IPW** ATE = $62 – corrects observed confounders
- **Double ML** ATE = $73 – robust to non‑linearities
- **DiD** ATE = $78 – validates parallel trends (p=0.68; note: synthetic data gives clean results; real data often fails this test)
- **Fairness** – initial policy violates equalized odds (TPR disparity 0.42); mitigation reduces it to 0.18

True ATE (from data generating process) = **$80** per month.

> **Note on OLS magnitude:** The extreme bias ($1904 vs $80) is a deliberate feature of the synthetic data to clearly demonstrate the importance of causal adjustment. In real-world settings, bias is usually smaller (2–5x). The causal methods still recover the true effect.

---

## 🚀 How to Run

```bash
git clone https://github.com/ayberkince/credit-limit-optimizer.git
cd credit-limit-optimizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the full analysis in terminal
python main.py

# Or launch the interactive dashboard
streamlit run src/dashboard/streamlit_app.py
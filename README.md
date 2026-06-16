# Credit Limit Optimizer

### Causal AI · Sequential Experimentation · Algorithmic Fairness · EU AI Act Compliance

A production-grade fintech system that determines the optimal credit-limit increase for existing users — correcting for confounding bias, validating with sequential A/B testing, auditing for fairness, and generating regulatory compliance reports.

> *"Naive OLS overestimated the treatment effect by more than 15× — our causal correction prevents over-allocating credit to high-risk users."*

---

## 📌 Problem Statement

A fintech company wants to increase credit limits for existing users to boost revenue (swipe fees + interest) while managing default risk. However, historical data is **severely confounded**: wealthier users naturally receive higher limits *and* spend more. Any naive comparison between treated and untreated users will dramatically overestimate the true effect of a limit increase.

**This project solves that problem** by building a complete analytical pipeline that:

1. **Estimates the causal effect** of a credit-limit increase on net revenue using modern econometric methods
2. **Designs a statistically rigorous A/B test** with sequential stopping rules for early termination
3. **Audits the resulting policy for fairness** across income groups using equalized odds
4. **Quantifies robustness** to unmeasured confounding via sensitivity analysis
5. **Models default timing** with survival analysis to assess long-term risk
6. **Monitors data drift** in production to detect when retraining is needed
7. **Generates EU AI Act compliance documentation** for regulatory requirements

---

## 🧠 Methodology

| Component | Technique | Module |
|-----------|-----------|--------|
| **Causal Inference** | Inverse Probability Weighting (IPW), Double Machine Learning (DML) | `src/causal/bias_correction.py` |
| **Robustness Check** | Difference-in-Differences (DiD) with parallel-trends testing | `src/causal/diff_in_diff.py` |
| **Causal DAG** | Directed Acyclic Graph visualization | `src/causal/dag.py` |
| **Experiment Design** | Power analysis (5% MDE, 80% power) + Sequential Probability Ratio Test (SPRT) | `src/experiment/` |
| **Fairness Audit** | Equalized odds with bootstrap CIs & per-group threshold mitigation | `src/inference/fairness.py` |
| **Sensitivity Analysis** | Rosenbaum bounds (approximate) for unmeasured confounding | `src/sensitivity/rosenbaum.py` |
| **Survival Modelling** | Cox Proportional Hazards & Kaplan-Meier for default timing | `src/survival/default_model.py` |
| **Policy Engine** | Net Incremental Value (NIV) calculation & treatment recommendations | `src/policy/decision_engine.py` |
| **Model Registry** | Versioned model storage with metadata | `src/model_registry/registry.py` |
| **Drift Detection** | Population Stability Index (PSI) monitoring | `src/monitoring/drift.py` |
| **Regulatory Compliance** | EU AI Act report generation (Markdown + HTML) | `src/compliance/eu_ai_act.py` |
| **Synthetic Data** | Calibrated to Kaggle Credit Risk dataset with known ground truth | `src/data_generator.py` |

All methods are derived from first principles in [`MATHEMATICAL_BACKGROUND.md`](MATHEMATICAL_BACKGROUND.md), covering Rubin's potential outcomes framework, Neyman-orthogonal scores, SPRT likelihood ratios, equalized odds, and power calculations.

---

## 📊 Key Results

### Causal Effect Estimates

| Estimator | ATE | 95% CI | Interpretation |
|-----------|-----|--------|----------------|
| **True ATE (DGP)** | $80.00 | — | Ground truth from data-generating process |
| **Naive OLS** | $1,221.86 | — | **Severely biased** due to confounding |
| **IPW** | $44.09 | −$12.75 – $93.22 | Adjusts for observed confounders, CI contains true effect |
| **Double ML** | $81.65 | $81.13 – $82.22 | Neyman-orthogonal, closest to ground truth |
| **DiD** | $79.72 | $79.03 – $80.37 | Parallel trends satisfied (p = 0.802) |

### Fairness Audit (Equalized Odds)

| Metric | Initial Policy | After Mitigation |
|--------|---------------|------------------|
| TPR Disparity | 0.497 (violation) | 0.344 (improved) |

> The mitigation reduces disparity while maintaining reasonable predictive performance (AUC = 0.617).

---

## 🏗️ Architecture

```
credit_limit_optimizer/
├── main.py                          # Full pipeline orchestration (11 stages)
├── config.yaml                      # All parameters (sample size, MDE, fairness metric, etc.)
├── Dockerfile                       # Containerized execution
├── MATHEMATICAL_BACKGROUND.md       # Formal derivations for all methods
├── requirements.txt
│
├── src/
│   ├── data_generator.py            # Synthetic panel data with configurable confounding
│   ├── causal/
│   │   ├── bias_correction.py       # IPW + Double ML with cross-fitting
│   │   ├── diff_in_diff.py          # DiD with parallel-trends placebo test
│   │   └── dag.py                   # Causal DAG visualization
│   ├── experiment/
│   │   ├── power_analysis.py        # Sample size calculation & power curves
│   │   ├── sequential_test.py       # SPRT implementation with boundary plots
│   │   └── randomization.py         # Treatment randomization
│   ├── inference/
│   │   └── fairness.py              # Equalized odds audit + mitigation
│   ├── policy/
│   │   └── decision_engine.py       # NIV-based treatment recommendations
│   ├── sensitivity/
│   │   └── rosenbaum.py             # Sensitivity to unmeasured confounding
│   ├── survival/
│   │   └── default_model.py         # Cox PH + Kaplan-Meier survival curves
│   ├── model_registry/
│   │   └── registry.py              # Versioned model storage
│   ├── monitoring/
│   │   └── drift.py                 # PSI-based drift detection
│   ├── compliance/
│   │   └── eu_ai_act.py             # Regulatory report generation
│   ├── dashboard/
│   │   └── streamlit_app.py         # Interactive exploration dashboard
│   └── api/
│       └── app.py                   # REST API (FastAPI)
│
├── scripts/
│   └── run_full_analysis.py         # Convenience script for full run
├── tests/
│   └── test_data_generator.py       # Unit tests
├── models/                          # Saved model artifacts
└── outputs/                         # Generated plots & reports
```

### Design Principles

- **Modular** — Each pipeline stage is an independent, testable module
- **Config-driven** — All parameters managed via a single `config.yaml`
- **Dual interface** — Interactive Streamlit dashboard for exploration + FastAPI REST endpoint for integration
- **Dockerized** — Single `docker build && docker run` for reproducible execution
- **Mathematically documented** — Every method derived from first principles

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+

### Installation

```bash
git clone https://github.com/ayberkince/credit-limit-optimizer.git
cd credit-limit-optimizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

This executes all 11 stages sequentially and saves plots to `outputs/`.

### Launch the Interactive Dashboard

```bash
streamlit run src/dashboard/streamlit_app.py
```

Adjust experiment parameters (number of users, intervention cost, loan loss) via the sidebar and run the full analysis interactively.

### Start the REST API

```bash
uvicorn src.api.app:app --reload
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Service status |
| `POST` | `/decide` | Get treatment recommendation for a single user |
| `POST` | `/decide_batch` | Get recommendations for a batch of users |

**Example request:**

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{"income_quintile": "medium"}'
```

### Run with Docker

```bash
docker build -t credit-limit-optimizer .
docker run credit-limit-optimizer
```

### Run Tests

```bash
python -m pytest tests/
```

---

## ⚙️ Configuration

All experiment and model parameters are controlled via [`config.yaml`](config.yaml):

| Section | Parameter | Description |
|---------|-----------|-------------|
| `data` | `n_users`, `n_months` | Synthetic data dimensions |
| `data` | `treatment_bias` | Confounding strength (intercept, income/credit score coefficients) |
| `data` | `true_cate` | Ground-truth CATE by income quintile (for validation) |
| `experiment` | `mde`, `alpha`, `beta` | Minimum detectable effect, Type I/II error rates |
| `fairness` | `metric`, `protected_attribute` | Fairness criterion and protected group |
| `policy` | `intervention_cost`, `avg_loan_loss` | Business parameters for NIV calculation |

---

## 📈 Generated Outputs

The pipeline produces the following artifacts in the `outputs/` directory:

| File | Description |
|------|-------------|
| `power_curve.png` | Statistical power vs. sample size |
| `sprt_null.png` | SPRT boundary plot under null hypothesis |
| `sprt_alt.png` | SPRT boundary plot under alternative hypothesis |
| `risk_calibration.png` | Risk model calibration curve + score distributions |
| `fairness_audit.png` | TPR/FPR comparison before and after mitigation |
| `survival_curves.png` | Kaplan-Meier survival curves by treatment group |
| `compliance_report.md` | EU AI Act compliance report (Markdown) |
| `compliance_report.html` | EU AI Act compliance report (HTML) |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11 |
| **Causal Inference** | scikit-learn, custom IPW & DML implementations |
| **Statistical Testing** | SciPy, Pingouin |
| **Survival Analysis** | Lifelines |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **REST API** | FastAPI, Uvicorn |
| **Configuration** | PyYAML |
| **Graph Visualization** | NetworkX |
| **Containerization** | Docker |
| **Testing** | pytest |

---

## 📚 References

- Rubin (1974) — *"Estimating causal effects of treatments"*
- Rosenbaum & Rubin (1983) — *"The central role of the propensity score"*
- Chernozhukov et al. (2016) — *"Double/debiased machine learning"*
- Johari et al. (2017) — *"Always valid inference"*
- Chouldechova (2017) — *"Fair prediction with disparate impact"*

---

## 📄 License

This project is for educational and portfolio demonstration purposes.
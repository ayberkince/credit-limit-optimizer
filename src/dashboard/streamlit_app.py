import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import tempfile
from src.data_generator import CreditDataGenerator
from src.causal.bias_correction import BiasCorrection
from src.causal.diff_in_diff import DiffInDifferences
from src.experiment.power_analysis import PowerAnalysis
from src.experiment.sequential_test import SequentialTest
from src.inference.fairness import FairnessAuditor

st.set_page_config(page_title="Credit Limit Optimizer", layout="wide")

st.title("📊 Credit Limit Optimizer: Causal AI & Experimentation Dashboard")
st.markdown("---")

# Load base config
with open("config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# Sidebar controls
st.sidebar.header("Experiment Parameters")
n_users = st.sidebar.slider("Number of users", 1000, 50000, base_config['data']['n_users'], step=1000)

run_button = st.sidebar.button("Run Full Analysis")
st.sidebar.markdown("---")
st.sidebar.markdown("**Modules:**\n- Causal (OLS, IPW, DML, DiD)\n- Power Analysis\n- SPRT\n- Fairness Audit")

def generate_data(n_users):
    # Create a modified config
    config = base_config.copy()
    config['data']['n_users'] = n_users
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    generator = CreditDataGenerator(temp_path)
    users, panel = generator.generate()
    os.unlink(temp_path)  # clean up
    return users, panel

if run_button:
    st.info(f"Generating synthetic data with {n_users} users...")
    users, panel = generate_data(n_users)
    st.success(f"Generated {len(users)} users and {len(panel)} panel rows.")
    
    # Causal Inference
    st.header("🔬 Causal Inference: Effect of Credit Limit Increase")
    bc = BiasCorrection(panel)
    results = bc.compare_estimators()
    st.dataframe(results.round(2))
    
    # True ATE
    true_cate_map = base_config['data']['true_cate']
    users['true_cate'] = users['income_quintile'].map(true_cate_map).astype(float)
    true_ate = users['true_cate'].mean()
    st.metric("True ATE (from DGP)", f"${true_ate:.2f}")
    
    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(results['method'], results['ate'], color=['red', 'blue', 'green'])
    ax.axhline(y=true_ate, color='black', linestyle='--', label=f'True ATE = ${true_ate:.2f}')
    ax.set_ylabel("ATE ($)")
    ax.set_title("Treatment Effect Estimates")
    ax.legend()
    st.pyplot(fig)
    
    # DiD
    st.subheader("Robustness Check: Difference-in-Differences")
    did = DiffInDifferences(panel, treatment_month=base_config['data']['treatment_month'])
    did_results = did.run_full_analysis()
    st.write(f"Parallel Trends Test: {did_results['parallel_trends_assumption']}")
    st.metric("DiD Estimate", f"${did_results['ate']:.2f}", delta=f"95% CI: [{did_results['ci_lower']:.2f}, {did_results['ci_upper']:.2f}]")
    
    # Experiment Design
    st.header("📐 Experiment Design")
    pre_control = panel[(panel['month'] < base_config['data']['treatment_month']) & (panel['treatment_assigned'] == 0)]['revenue']
    control_mean = pre_control.mean()
    control_std = pre_control.std()
    pa = PowerAnalysis(alpha=0.05, beta=0.20, mde=0.05, control_mean=control_mean, control_std=control_std)
    n_per_group = pa.required_sample_size()
    st.metric("Required sample size per group (MDE=5%)", n_per_group)
    st.metric("Total users needed", 2*n_per_group)
    
    # Power curve
    fig2, ax2 = plt.subplots()
    sample_sizes, powers = pa.power_curve()
    ax2.plot(sample_sizes, powers)
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80% power')
    ax2.set_xlabel("Sample size per group")
    ax2.set_ylabel("Power")
    ax2.set_title("Power Curve")
    ax2.legend()
    st.pyplot(fig2)
    
    # Sequential test (SPRT)
    st.subheader("Sequential Testing (SPRT)")
    spr = SequentialTest(alpha=0.05, beta=0.20, mde=0.05, control_mean=control_mean, control_std=control_std)
    fig3 = spr.plot_sprt(true_effect=0)
    st.pyplot(fig3)
    st.caption("Under null hypothesis (true effect = 0), the likelihood ratio stays below the reject boundary.")
    
    # Fairness Audit
    st.header("⚖️ Fairness Audit")
    auditor = FairnessAuditor(panel)
    fig_cal = auditor.plot_risk_calibration()
    st.pyplot(fig_cal)
    fig_fair = auditor.plot_fairness_comparison()
    st.pyplot(fig_fair)
    report = auditor.generate_fairness_report()
    st.text(report)
    
    st.markdown("---")
    st.success("Analysis complete.")
else:
    st.info("Click 'Run Full Analysis' to start.")
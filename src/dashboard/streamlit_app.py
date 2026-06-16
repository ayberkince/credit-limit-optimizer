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
from src.policy.decision_engine import DecisionEngine   # <-- NEW IMPORT

st.set_page_config(page_title="Credit Limit Optimizer", layout="wide")

st.title("📊 Credit Limit Optimizer: Causal AI & Experimentation Dashboard")
st.markdown("---")

# Load base config
with open("config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# Sidebar controls
st.sidebar.header("Experiment Parameters")
n_users = st.sidebar.slider("Number of users", 1000, 50000, base_config['data']['n_users'], step=1000)

# Policy parameters (sidebar)
st.sidebar.header("Policy Parameters")
intervention_cost = st.sidebar.number_input("Intervention Cost ($)", value=50, min_value=0, step=5)
avg_loan_loss = st.sidebar.number_input("Avg Loan Loss ($)", value=500, min_value=100, step=50)

run_button = st.sidebar.button("Run Full Analysis")
st.sidebar.markdown("---")
st.sidebar.markdown("**Modules:**\n- Causal (OLS, IPW, DML, DiD)\n- Power Analysis\n- SPRT\n- Fairness Audit\n- Policy Decision")

def generate_data(n_users):
    config = base_config.copy()
    config['data']['n_users'] = n_users
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    generator = CreditDataGenerator(temp_path)
    users, panel = generator.generate()
    os.unlink(temp_path)
    return users, panel

if run_button:
    st.info(f"Generating synthetic data with {n_users} users...")
    users, panel = generate_data(n_users)
    st.success(f"Generated {len(users)} users and {len(panel)} panel rows.")
    
    # ============ CAUSAL INFERENCE ============
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
    
    # ============ POLICY DECISION ENGINE ============
    st.header("📋 Policy Decision Engine")
    st.markdown("**Translates causal estimates into business recommendations**")
    
    cate_map = base_config['data']['true_cate']
    default_risk_increase = base_config.get('policy', {}).get('default_risk_increase', 0.02)
    
    engine = DecisionEngine(
        cate_map=cate_map,
        intervention_cost=intervention_cost,
        avg_loan_loss=avg_loan_loss,
        default_risk_increase=default_risk_increase
    )
    
    # Apply policy and get summary
    df_policy = engine.apply_policy(users)
    n_treat = df_policy['recommended_treatment'].sum()
    total_niv = df_policy['expected_niv'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Users to Treat", f"{n_treat:,}", delta=f"{100*n_treat/len(users):.1f}% of users")
    col2.metric("Total Expected NIV", f"${total_niv:,.2f}")
    col3.metric("Avg NIV per Treated User", f"${df_policy[df_policy['recommended_treatment']]['expected_niv'].mean():.2f}")
    
    # NIV by quintile bar chart
    fig_policy, ax_policy = plt.subplots(figsize=(10, 5))
    quintile_summary = df_policy.groupby('income_quintile').agg(
        count=('income_quintile', 'size'),
        total_niv=('expected_niv', 'sum'),
        avg_niv=('expected_niv', 'mean'),
        treat_count=('recommended_treatment', 'sum')
    ).reset_index()
    
    # Sort by avg_niv for better visualization
    quintile_summary = quintile_summary.sort_values('avg_niv', ascending=False)
    
    x = np.arange(len(quintile_summary))
    width = 0.35
    ax_policy.bar(x - width/2, quintile_summary['avg_niv'], width, color='steelblue', label='Avg NIV per user')
    ax_policy.bar(x + width/2, quintile_summary['treat_count'] / quintile_summary['count'] * 100, width, color='darkorange', label='% Treated')
    ax_policy.set_xlabel('Income Quintile')
    ax_policy.set_ylabel('Value ($ or %)')
    ax_policy.set_title('Policy Recommendation by Income Quintile')
    ax_policy.set_xticks(x)
    ax_policy.set_xticklabels(quintile_summary['income_quintile'], rotation=45)
    ax_policy.legend()
    ax_policy.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    st.pyplot(fig_policy)
    
    # Show detailed breakdown table
    with st.expander("View detailed breakdown by income quintile"):
        st.dataframe(quintile_summary.round(2))

    # ... after showing the policy bar chart ...

# --- CATE vs NIV comparison chart ---
    if 'true_cate' not in df_policy.columns:
        # Merge true_cate from users into df_policy
        df_policy = df_policy.merge(users[['user_id', 'true_cate']], on='user_id', how='left')
    
    cate_vs_niv = df_policy.groupby('income_quintile').agg(
        avg_cate=('true_cate', 'mean'),
        avg_niv=('expected_niv', 'mean')
    ).reset_index()
    
    # Sort by avg_cate for better visualization
    cate_vs_niv = cate_vs_niv.sort_values('avg_cate', ascending=False)
    
    fig_cate, ax_cate = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cate_vs_niv))
    width = 0.35
    ax_cate.bar(x - width/2, cate_vs_niv['avg_cate'], width, label='CATE (gross benefit)', color='steelblue')
    ax_cate.bar(x + width/2, cate_vs_niv['avg_niv'], width, label='NIV (net benefit)', color='darkorange')
    ax_cate.set_xlabel('Income Quintile')
    ax_cate.set_ylabel('Dollars per month')
    ax_cate.set_title('CATE vs NIV by Income Quintile')
    ax_cate.set_xticks(x)
    ax_cate.set_xticklabels(cate_vs_niv['income_quintile'], rotation=45)
    ax_cate.legend()
    ax_cate.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    st.pyplot(fig_cate)
    
    # ============ DiD ============
    st.subheader("Robustness Check: Difference-in-Differences")
    did = DiffInDifferences(panel, treatment_month=base_config['data']['treatment_month'])
    did_results = did.run_full_analysis()
    st.write(f"Parallel Trends Test: {did_results['parallel_trends_assumption']}")
    st.metric("DiD Estimate", f"${did_results['ate']:.2f}", delta=f"95% CI: [{did_results['ci_lower']:.2f}, {did_results['ci_upper']:.2f}]")
    
    # ============ EXPERIMENT DESIGN ============
    st.header("📐 Experiment Design")
    pre_control = panel[(panel['month'] < base_config['data']['treatment_month']) & (panel['treatment_assigned'] == 0)]['revenue']
    control_mean = pre_control.mean()
    control_std = pre_control.std()
    pa = PowerAnalysis(alpha=0.05, beta=0.20, mde=0.05 * control_mean, control_mean=control_mean, control_std=control_std)
    n_per_group = pa.required_sample_size()
    st.metric("Required sample size per group (MDE=5%)", n_per_group)
    st.metric("Total users needed", 2*n_per_group)
    
    fig2, ax2 = plt.subplots()
    sample_sizes, powers = pa.power_curve()
    ax2.plot(sample_sizes, powers)
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80% power')
    ax2.set_xlabel("Sample size per group")
    ax2.set_ylabel("Power")
    ax2.set_title("Power Curve")
    ax2.legend()
    st.pyplot(fig2)
    
    # ============ SPRT ============
    st.subheader("Sequential Testing (SPRT)")
    spr = SequentialTest(alpha=0.05, beta=0.20, mde=0.05, control_mean=control_mean, control_std=control_std)
    fig3 = spr.plot_sprt(true_effect=0)
    st.pyplot(fig3)
    st.caption("Under null hypothesis (true effect = 0), the likelihood ratio stays below the reject boundary.")
    
    # ============ FAIRNESS ============
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
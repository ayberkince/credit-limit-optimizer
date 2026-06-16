#!/usr/bin/env python
"""
Main entry point for the credit limit optimization pipeline.
Runs: data generation → causal inference → DiD → experiment design → fairness audit
      → sensitivity analysis → survival model → model registry → drift detection.
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import CreditDataGenerator
from src.causal.bias_correction import BiasCorrection
from src.causal.diff_in_diff import DiffInDifferences
from src.experiment.power_analysis import PowerAnalysis
from src.experiment.sequential_test import SequentialTest
from src.inference.fairness import FairnessAuditor
from src.policy.decision_engine import DecisionEngine
from src.sensitivity.rosenbaum import RosenbaumSensitivity
from src.survival.default_model import DefaultSurvivalModel
from src.model_registry.registry import ModelRegistry
from src.monitoring.drift import DriftDetector
from sklearn.linear_model import LogisticRegression
from src.compliance.eu_ai_act import EUAICompliance

def main(config_path="config.yaml"):
    logger.info("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create outputs directory for saving plots
    os.makedirs("outputs", exist_ok=True)

    # ============================================================
    # STEP 1: Data Generation
    # ============================================================
    logger.info("Step 1: Generating synthetic panel data...")
    generator = CreditDataGenerator(config_path)
    users, panel = generator.generate()
    logger.info(f"Generated {len(users)} users, {len(panel)} panel rows")

    # Check default events
    n_defaults = panel[panel['default'] == 1].shape[0]
    logger.info(f"Total default events in panel: {n_defaults}")
    n_users_with_default = panel[panel['default'] == 1]['user_id'].nunique()
    logger.info(f"Users with at least one default: {n_users_with_default}")

    # Compute true ATE from DGP for validation
    true_cate_map = config['data']['true_cate']
    true_cate_map = {str(k): float(v) for k, v in true_cate_map.items()}
    users['true_cate'] = users['income_quintile'].map(true_cate_map).astype(float)
    n_missing = users['true_cate'].isna().sum()
    if n_missing > 0:
        unique_labels = users['income_quintile'].unique()
        raise ValueError(
            f"Missing mapping for {n_missing} users. Quintile labels found: {unique_labels}\n"
            f"Expected keys: {list(true_cate_map.keys())}"
        )
    true_ate = users['true_cate'].mean()
    logger.info(f"True ATE (from DGP): ${true_ate:.2f}")

    # ============================================================
    # STEP 2: Causal Inference
    # ============================================================
    logger.info("Step 2: Causal Inference - Estimating Treatment Effect...")
    bc = BiasCorrection(panel)
    results = bc.compare_estimators()
    print("\n=== Causal Effect Estimates ===")
    print(results.round(2).to_string(index=False))

    # ============================================================
    # STEP 3: Policy Decision Engine
    # ============================================================
    logger.info("Step 2.5: Policy Decision Engine - converting CATE to business recommendations...")
    intervention_cost = config.get('policy', {}).get('intervention_cost', 50)
    avg_loan_loss = config.get('policy', {}).get('avg_loan_loss', 500)
    default_risk_increase = config.get('policy', {}).get('default_risk_increase', 0.02)

    engine = DecisionEngine(
        cate_map=config['data']['true_cate'],
        intervention_cost=intervention_cost,
        avg_loan_loss=avg_loan_loss,
        default_risk_increase=default_risk_increase
    )
    report = engine.summary_report(users)
    print(report)

    # ============================================================
    # STEP 4: Difference‑in‑Differences (Robustness)
    # ============================================================
    logger.info("Step 3: Robustness Check - Difference-in-Differences...")
    did = DiffInDifferences(panel, treatment_month=config['data']['treatment_month'])
    did_results = did.run_full_analysis()
    print(f"\nParallel Trends Test: {did_results['parallel_trends_assumption']}")
    print(f"DiD Estimate: ${did_results['ate']:.2f} (95% CI: ${did_results['ci_lower']:.2f} - ${did_results['ci_upper']:.2f})")

    # ============================================================
    # STEP 5: Experiment Design (Power + SPRT)
    # ============================================================
    logger.info("Step 4: Experiment Design - Power Analysis & Sequential Testing...")
    pre_control_revenue = panel[
        (panel['month'] < config['data']['treatment_month']) &
        (panel['treatment_assigned'] == 0)
    ]['revenue']
    control_mean = pre_control_revenue.mean()
    control_std = pre_control_revenue.std()
    mde_fraction = config['experiment']['mde']
    mde_dollars = mde_fraction * control_mean
    logger.info(f"Control mean revenue = ${control_mean:.2f}, MDE = {mde_fraction*100}% = ${mde_dollars:.2f}")

    pa = PowerAnalysis(
        alpha=config['experiment']['alpha'],
        beta=config['experiment']['beta'],
        mde=mde_dollars,
        control_mean=control_mean,
        control_std=control_std
    )
    n_per_group = pa.required_sample_size()
    print(f"\nRequired sample size per group (MDE=5%): {n_per_group}")
    print(f"Total users needed: {2*n_per_group}")
    pa.plot_power_curve(save_path="outputs/power_curve.png")

    spr = SequentialTest(
        alpha=config['experiment']['alpha'],
        beta=config['experiment']['beta'],
        mde=mde_fraction,
        control_mean=control_mean,
        control_std=control_std
    )
    print("\nSimulating SPRT under null hypothesis (true effect = 0)...")
    spr.plot_sprt(true_effect=0, save_path="outputs/sprt_null.png")
    print("SPRT plot saved to outputs/sprt_null.png")
    print("\nSimulating SPRT under alternative (true effect = MDE)...")
    spr.plot_sprt(true_effect=mde_dollars, save_path="outputs/sprt_alt.png")
    print("SPRT plot saved to outputs/sprt_alt.png")

    # ============================================================
    # STEP 6: Fairness Audit
    # ============================================================
    logger.info("Step 5: Fairness Audit...")
    auditor = FairnessAuditor(panel)
    auditor.plot_risk_calibration(save_path="outputs/risk_calibration.png")
    auditor.plot_fairness_comparison(save_path="outputs/fairness_audit.png")
    print(auditor.generate_fairness_report())

    # ============================================================
    # STEP 7: Sensitivity Analysis (Rosenbaum)
    # ============================================================
    logger.info("Step 5.5: Sensitivity Analysis...")
    sens = RosenbaumSensitivity(panel)
    print(sens.interpret())

    # ============================================================
    # STEP 8: Survival Model for Default Timing
    # ============================================================
    logger.info("Step 5.6: Survival Model for Default Timing...")
    surv = DefaultSurvivalModel(panel)
    surv.fit()
    if surv.model is not None:
        print(surv.interpret())
        surv.plot_survival(save_path="outputs/survival_curves.png")
    else:
        logger.warning("Survival model not fitted – skipping interpretation and plot.")

    # ============================================================
    # STEP 9: Model Registry
    # ============================================================
    logger.info("Step 5.7: Model Registry...")
    registry = ModelRegistry()

    # Save propensity score model (trained during causal inference)
    ps_model = LogisticRegression(max_iter=500, random_state=42)
    ps_model.fit(bc.X_norm, bc.T)
    registry.save_model(ps_model, "propensity_model",
                        n_users=int(len(bc.X_norm)),
                        n_features=int(bc.X_norm.shape[1]))

    if surv.model is not None:
        registry.save_model(surv.model, "survival_model",
                            n_events=int(surv.survival_df['event_observed'].sum()),
                            n_users=int(len(surv.survival_df)))
    else:
        logger.warning("Survival model not available for registry.")

    # ============================================================
    # STEP 10: Drift Detection
    # ============================================================
    logger.info("Step 5.8: Drift Detection...")
    reference = panel[panel['month'] < config['data']['treatment_month']]
    current = panel[panel['month'] >= config['data']['treatment_month']]
    detector = DriftDetector(reference)
    drift_report = detector.generate_report(current)
    print(drift_report)

    
    logger.info("Step 6: Generating EU AI Act Compliance Report...")
    
    # Safely extract estimates, handling missing methods gracefully
    ate_estimates = {}
    methods_to_extract = {
        'Naive OLS': 'Naive OLS',
        'IPW': 'IPW',
        'Double ML': 'Double ML (manual)'
    }
    
    ci_dict = {}
    
    for label, method_name in methods_to_extract.items():
        row = results[results['method'] == method_name]
        
        # Check if the method actually exists in our results DataFrame
        if not row.empty:
            ate_estimates[label] = row['ate'].iloc[0]
            
            # Check if confidence intervals exist and aren't null
            if 'ci_lower' in row.columns and 'ci_upper' in row.columns:
                if not pd.isna(row['ci_lower'].iloc[0]) and not pd.isna(row['ci_upper'].iloc[0]):
                    ci_dict[label] = (row['ci_lower'].iloc[0], row['ci_upper'].iloc[0])
                else:
                    ci_dict[label] = (None, None)
            else:
                ci_dict[label] = (None, None)
        else:
            logger.warning(f"Causal method '{method_name}' was not found in bias correction results.")
            ate_estimates[label] = 0.0  # Fallback baseline
            ci_dict[label] = (None, None)

    results_dict = {'ate_estimates': ate_estimates, 'ci': ci_dict}

    # Fairness audit results (extract from auditor if available)
    fairness_dict = {
        'initial': {
            'tpr_disparity': auditor.audit_policy("initial")['tpr_disparity'],
            'fpr_disparity': auditor.audit_policy("initial")['fpr_disparity'],
            'violated': auditor.audit_policy("initial")['equalized_odds_violation']
        },
        'mitigated': {
            'tpr_disparity': auditor.mitigate()['tpr_disparity'],
            'fpr_disparity': auditor.mitigate()['fpr_disparity']
        }
    }

    # Survival model (if available)
    surv_model = surv if surv.model is not None else None

    # Policy report (already printed)
    policy_report = report

    # Generate compliance report
    compliance = EUAICompliance(
        config=config,
        results=results_dict,
        fairness_audit=fairness_dict,
        survival_model=surv_model,
        policy_report=policy_report,
        output_dir="outputs"
    )
    compliance.generate_report()
    compliance.generate_html()
    logger.info("Compliance report generated.")

    logger.info("Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
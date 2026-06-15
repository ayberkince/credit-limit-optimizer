#!/usr/bin/env python
"""
Main entry point for the credit limit optimization pipeline.
Runs: data generation → causal inference → DiD → experiment design → fairness audit.
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.data_generator import CreditDataGenerator
from src.causal.bias_correction import BiasCorrection
from src.causal.diff_in_diff import DiffInDifferences
from src.experiment.power_analysis import PowerAnalysis
from src.experiment.sequential_test import SequentialTest
from src.inference.fairness import FairnessAuditor

def main(config_path="config.yaml"):
    logger.info("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create outputs directory for saving plots
    os.makedirs("outputs", exist_ok=True)

    logger.info("Step 1: Generating synthetic panel data...")
    generator = CreditDataGenerator(config_path)
    users, panel = generator.generate()
    logger.info(f"Generated {len(users)} users, {len(panel)} panel rows")

    # Compute true ATE from DGP for validation
    true_cate_map = config['data']['true_cate']
    # Ensure keys are strings and values float
    true_cate_map = {str(k): float(v) for k, v in true_cate_map.items()}
    users['true_cate'] = users['income_quintile'].map(true_cate_map).astype(float)
    # Guard: check that mapping didn't produce NaN (i.e., quintile labels match config)
    n_missing = users['true_cate'].isna().sum()
    if n_missing > 0:
        unique_labels = users['income_quintile'].unique()
        raise ValueError(f"Missing mapping for {n_missing} users. Quintile labels found: {unique_labels}\n"
                         f"Expected keys: {list(true_cate_map.keys())}")
    true_ate = users['true_cate'].mean()
    logger.info(f"True ATE (from DGP): ${true_ate:.2f}")

    logger.info("Step 2: Causal Inference - Estimating Treatment Effect...")
    bc = BiasCorrection(panel)
    results = bc.compare_estimators()
    print("\n=== Causal Effect Estimates ===")
    print(results.round(2).to_string(index=False))

    logger.info("Step 3: Robustness Check - Difference-in-Differences...")
    did = DiffInDifferences(panel, treatment_month=config['data']['treatment_month'])
    did_results = did.run_full_analysis()
    print(f"\nParallel Trends Test: {did_results['parallel_trends_assumption']}")
    print(f"DiD Estimate: ${did_results['ate']:.2f} (95% CI: ${did_results['ci_lower']:.2f} - ${did_results['ci_upper']:.2f})")

    logger.info("Step 4: Experiment Design - Power Analysis & Sequential Testing...")
    # Use pre-treatment control group revenue to estimate baseline mean and std
    pre_control_revenue = panel[(panel['month'] < config['data']['treatment_month']) &
                                (panel['treatment_assigned'] == 0)]['revenue']
    control_mean = pre_control_revenue.mean()
    control_std = pre_control_revenue.std()
    # Convert MDE from fraction to dollars
    mde_fraction = config['experiment']['mde']
    mde_dollars = mde_fraction * control_mean
    logger.info(f"Control mean revenue = ${control_mean:.2f}, MDE = {mde_fraction*100}% = ${mde_dollars:.2f}")

    pa = PowerAnalysis(alpha=config['experiment']['alpha'],
                       beta=config['experiment']['beta'],
                       mde=mde_dollars,      # now in dollars
                       control_mean=control_mean,
                       control_std=control_std)
    n_per_group = pa.required_sample_size()
    print(f"\nRequired sample size per group (MDE=5%): {n_per_group}")
    print(f"Total users needed: {2*n_per_group}")
    pa.plot_power_curve(save_path="outputs/power_curve.png")

    # Sequential test (SPRT)
    spr = SequentialTest(alpha=config['experiment']['alpha'],
                         beta=config['experiment']['beta'],
                         mde=mde_fraction,
                         control_mean=control_mean)
    print("\nSimulating SPRT under null hypothesis (true effect = 0)...")
    spr.plot_sprt(true_effect=0, save_path="outputs/sprt_null.png")
    print("SPRT plot saved to outputs/sprt_null.png")
    print("\nSimulating SPRT under alternative (true effect = MDE)...")
    spr.plot_sprt(true_effect=mde_dollars, save_path="outputs/sprt_alt.png")
    print("SPRT plot saved to outputs/sprt_alt.png")

    logger.info("Step 5: Fairness Audit...")
    auditor = FairnessAuditor(panel)
    auditor.plot_risk_calibration(save_path="outputs/risk_calibration.png")
    auditor.plot_fairness_comparison(save_path="outputs/fairness_audit.png")
    print(auditor.generate_fairness_report())

    logger.info("Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
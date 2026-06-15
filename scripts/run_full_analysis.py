#!/usr/bin/env python
"""
Run the full credit limit optimization pipeline from the terminal.
Generates data, estimates causal effects, runs DiD, power analysis, SPRT, and fairness audit.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data_generator import CreditDataGenerator
from src.causal.bias_correction import BiasCorrection
from src.causal.diff_in_diff import DiffInDifferences
from src.experiment.power_analysis import PowerAnalysis
from src.experiment.sequential_test import SequentialTest
from src.inference.fairness import FairnessAuditor

def main():
    print("Loading configuration...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Generating synthetic panel data...")
    gen = CreditDataGenerator("config.yaml")
    users, panel = gen.generate()
    print(f"Generated {len(users)} users, {len(panel)} panel rows.\n")

    # True ATE
    true_cate_map = config['data']['true_cate']
    users['true_cate'] = users['income_quintile'].map(true_cate_map).astype(float)
    true_ate = users['true_cate'].mean()
    print(f"True ATE (from DGP): ${true_ate:.2f}\n")

    # Causal estimators
    bc = BiasCorrection(panel)
    results = bc.compare_estimators()
    print("Causal Effect Estimates:")
    print(results.round(2).to_string(index=False))
    print("\n" + "-"*50 + "\n")

    # DiD
    did = DiffInDifferences(panel, treatment_month=config['data']['treatment_month'])
    did_results = did.run_full_analysis()
    print("Difference-in-Differences:")
    print(f"  Parallel trends test: {did_results['parallel_trends_assumption']}")
    print(f"  DiD estimate: ${did_results['ate']:.2f} (95% CI: [{did_results['ci_lower']:.2f}, {did_results['ci_upper']:.2f}])")
    print("\n" + "-"*50 + "\n")

    # Power analysis
    pre_control = panel[(panel['month'] < config['data']['treatment_month']) & (panel['treatment_assigned'] == 0)]['revenue']
    control_mean = pre_control.mean()
    control_std = pre_control.std()
    pa = PowerAnalysis(alpha=0.05, beta=0.20, mde=0.05, control_mean=control_mean, control_std=control_std)
    n_per_group = pa.required_sample_size()
    print("Experiment Design (Power Analysis):")
    print(f"  Required sample size per group (MDE=5%): {n_per_group}")
    print(f"  Total users needed: {2*n_per_group}\n")

    # SPRT simulation (under null)
    spr = SequentialTest(alpha=0.05, beta=0.20, mde=0.05, control_mean=control_mean)
    print("Sequential Testing (SPRT) – simulation under null:")
    spr.plot_sprt(true_effect=0, save=True)  # saves sprt_trajectory_effect_0.png
    print("  SPRT plot saved as 'sprt_trajectory_effect_0.png'\n")

    # Fairness audit
    auditor = FairnessAuditor(panel)
    auditor.plot_risk_calibration(save=True)
    auditor.plot_fairness_comparison(save=True)
    report = auditor.generate_fairness_report()
    print(report)

    print("\nPipeline complete. All plots saved as PNG files.")

if __name__ == "__main__":
    main()
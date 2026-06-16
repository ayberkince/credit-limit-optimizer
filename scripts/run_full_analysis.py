#!/usr/bin/env python
"""
Run the full credit limit optimization pipeline from the terminal.
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
from src.policy.decision_engine import DecisionEngine
from src.sensitivity.rosenbaum import RosenbaumSensitivity
from src.survival.default_model import DefaultSurvivalModel
from src.model_registry.registry import ModelRegistry
from src.monitoring.drift import DriftDetector
from sklearn.linear_model import LogisticRegression

def main():
    os.makedirs("outputs", exist_ok=True)

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

    # Policy Decision Engine
    intervention_cost = config.get('policy', {}).get('intervention_cost', 50)
    avg_loan_loss = config.get('policy', {}).get('avg_loan_loss', 500)
    default_risk_increase = config.get('policy', {}).get('default_risk_increase', 0.02)
    engine = DecisionEngine(
        cate_map=config['data']['true_cate'],
        intervention_cost=intervention_cost,
        avg_loan_loss=avg_loan_loss,
        default_risk_increase=default_risk_increase
    )
    print(engine.summary_report(users))
    print("-"*50 + "\n")

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
    mde_dollars = config['experiment']['mde'] * control_mean
    pa = PowerAnalysis(alpha=config['experiment']['alpha'], beta=config['experiment']['beta'],
                       mde=mde_dollars, control_mean=control_mean, control_std=control_std)
    n_per_group = pa.required_sample_size()
    print("Experiment Design (Power Analysis):")
    print(f"  Required sample size per group (MDE=5%): {n_per_group}")
    print(f"  Total users needed: {2*n_per_group}\n")
    pa.plot_power_curve(save_path="outputs/power_curve.png")

    # SPRT simulation
    spr = SequentialTest(
        alpha=config['experiment']['alpha'],
        beta=config['experiment']['beta'],
        mde=config['experiment']['mde'],
        control_mean=control_mean,
        control_std=control_std
    )
    print("Sequential Testing (SPRT) – simulation under null:")
    spr.plot_sprt(true_effect=0, save_path="outputs/sprt_null.png")
    print("  SPRT null plot saved to outputs/sprt_null.png")
    print("Sequential Testing (SPRT) – simulation under alternative:")
    spr.plot_sprt(true_effect=mde_dollars, save_path="outputs/sprt_alt.png")
    print("  SPRT alt plot saved to outputs/sprt_alt.png\n")

    # Fairness audit
    auditor = FairnessAuditor(panel)
    auditor.plot_risk_calibration(save_path="outputs/risk_calibration.png")
    auditor.plot_fairness_comparison(save_path="outputs/fairness_audit.png")
    print(auditor.generate_fairness_report())
    print("-"*50 + "\n")

    # Sensitivity analysis
    sens = RosenbaumSensitivity(panel)
    print(sens.interpret())
    print("-"*50 + "\n")

    # Survival model
    surv = DefaultSurvivalModel(panel)
    surv.fit()
    if surv.model is not None:
        print(surv.interpret())
        surv.plot_survival(save_path="outputs/survival_curves.png")
    else:
        print("Survival model not fitted – skipping.")
    print("-"*50 + "\n")

    # Model registry
    registry = ModelRegistry()
    ps_model = LogisticRegression(max_iter=500, random_state=42)
    ps_model.fit(bc.X_norm, bc.T)
    registry.save_model(ps_model, "propensity_model",
                        n_users=int(len(bc.X_norm)),
                        n_features=int(bc.X_norm.shape[1]))
    if surv.model is not None:
        registry.save_model(surv.model, "survival_model",
                            n_events=int(surv.survival_df['event_observed'].sum()),
                            n_users=int(len(surv.survival_df)))
    print("Model registry updated.\n")

    # Drift detection
    reference = panel[panel['month'] < config['data']['treatment_month']]
    current = panel[panel['month'] >= config['data']['treatment_month']]
    detector = DriftDetector(reference)
    print(detector.generate_report(current))

    print("\nPipeline complete. All plots saved to outputs/")

if __name__ == "__main__":
    main()

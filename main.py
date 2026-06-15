from pandas.core.window import expanding
import yaml
import argparse
import pandas as pd
from src.data_generator import CreditDataGenerator
from src.causal.bias_correction import BiasCorrection
from src.causal.diff_in_diff import DiffInDifferences
from src.experiment.power_analysis import PowerAnalysis
from src.experiment.sequential_test import SequentialTest
from src.inference.fairness import FairnessAuditor

def main(config_path="config.yaml"):
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Step 1: Generating synthetic panel data...")
    generator = CreditDataGenerator(config_path)
    users, panel = generator.generate()
    
    print(f"Generated {len(users)} users, {len(panel)} panel rows")
    
    # Compute true ATE from DGP for validation
    true_cate_map = config['data']['true_cate']
    # Ensure numeric mapping
    true_cate_map = {k: float(v) for k, v in true_cate_map.items()}
    
    # Map quintile to CATE, then ensure numeric (not categorical)
    users['true_cate'] = users['income_quintile'].map(true_cate_map).astype(float)
    true_ate = users['true_cate'].mean()
    print(f"\nTrue ATE (from DGP): ${true_ate:.2f}")
    
    print("\nStep 2: Causal Inference - Estimating Treatment Effect...")
    bc = BiasCorrection(panel)
    results = bc.compare_estimators()
    print("\n=== Causal Effect Estimates ===")
    # Display with nicer formatting
    print(results.round(2).to_string(index=False))

    print("\nStep 3: Robustness Check - Difference-in-Differences...")
    did = DiffInDifferences(panel, treatment_month=config['data']['treatment_month'])
    did_results = did.run_full_analysis()
    print(f"\nParallel Trends Test: {did_results['parallel_trends_assumption']}")
    print(f"DiD Estimate: ${did_results['ate']:.2f} (95% CI: ${did_results['ci_lower']:.2f} - ${did_results['ci_upper']:.2f})")
    
    print("\nPipeline complete (additional modules to be added).")

    print("\nStep 4: Experiment Design - Power Analysis & Sequential Testing...")
    # Estimate control mean and std from panel data (pre-treatment period)
    pre_control_revenue = panel[(panel['month'] < config['data']['treatment_month']) & 
                            (panel['treatment_assigned'] == 0)]['revenue']
    control_mean = pre_control_revenue.mean()
    control_std = pre_control_revenue.std()

    pa = PowerAnalysis(alpha=0.05, beta=0.20, mde=0.05, 
                   control_mean=control_mean, control_std=control_std)
    n_per_group = pa.required_sample_size()
    print(f"\nRequired sample size per group (MDE=5%): {n_per_group}")
    print(f"Total users needed: {2*n_per_group}")

    # Generate power curve
    pa.plot_power_curve()
    print("Power curve saved as 'power_curve.png'")

    # Sequential test example
    st = SequentialTest(alpha=0.05, beta=0.20, mde=0.05, 
                    control_mean=control_mean)
    # Simulate under null (no effect)
    print("\nSimulating sequential test under null hypothesis...")
    st.plot_sprt(true_effect=0)
    print("SPRT plot saved as 'sprt_trajectory_effect_0.png'")


    print("\nStep 5: Fairness Audit...")
    auditor = FairnessAuditor(panel, protected_attr='income_quintile')
    auditor.plot_fairness_comparison()
    print(auditor.generate_fairness_report())   

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
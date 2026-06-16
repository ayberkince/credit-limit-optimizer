"""
Rosenbaum sensitivity analysis for unmeasured confounding.
Quantifies how strong an unmeasured confounder would need to be
to invalidate the causal estimate.

For large datasets, this implementation subsamples to avoid memory issues.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# NOTE: This module implements an *approximate* sensitivity analysis inspired by Rosenbaum
# bounds, but does NOT implement the classical Rosenbaum (2002) bounds. The classical
# method requires matched pairs and the Wilcoxon signed-rank statistic. Here we use a
# two-sample approximation: a Gamma-level confounder can shift the z-statistic by up to
# log(Gamma), following the large-sample approximation in Rosenbaum & Rubin (1983). The
# resulting "critical gamma" is an approximate lower bound on how strong unmeasured
# confounding would need to be to overturn significance. Results should be interpreted
# as indicative, not as exact Rosenbaum bounds.

class RosenbaumSensitivity:
    def __init__(self, df_panel: pd.DataFrame, treatment_col: str = 'treatment_active',
                 outcome_col: str = 'revenue', max_sample_per_group: int = 2000):
        """
        Parameters:
        df_panel: Panel data
        treatment_col: Column name for treatment indicator
        outcome_col: Column name for outcome
        max_sample_per_group: Maximum number of observations per group for sensitivity analysis
                              (subsampling to avoid memory issues)
        """
        self.df = df_panel.copy()
        self.treatment = treatment_col
        self.outcome = outcome_col
        self.max_sample = max_sample_per_group
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data: use post-treatment period, drop missing values, subsample"""
        df_post = self.df[self.df[self.treatment].isin([0,1])].copy()
        df_post = df_post.dropna(subset=[self.outcome, self.treatment])
        
        # Separate treated and control
        treated = df_post[df_post[self.treatment] == 1]
        control = df_post[df_post[self.treatment] == 0]
        
        # Subsample to avoid memory issues
        if len(treated) > self.max_sample:
            treated = treated.sample(self.max_sample, random_state=42)
        if len(control) > self.max_sample:
            control = control.sample(self.max_sample, random_state=42)
        
        self.Y_treat = treated[self.outcome].values
        self.Y_control = control[self.outcome].values
        self.n_treat = len(self.Y_treat)
        self.n_control = len(self.Y_control)
        
    def compute_gamma_range(self, gamma_min=1.0, gamma_max=2.5, steps=10):
        """
        Compute the sensitivity parameter gamma for a range of values.
        Uses the t-test and a correction factor to approximate Rosenbaum bounds.
        
        Gamma is the odds ratio of treatment assignment between two matched individuals
        with identical covariates but different unmeasured confounders.
        """
        # Compute observed t-statistic and two-tailed p-value
        t_stat, p_obs = ttest_ind(self.Y_treat, self.Y_control)
        z_obs = abs(t_stat)

        gamma_vals = np.linspace(gamma_min, gamma_max, steps)
        p_values = []

        for gamma in gamma_vals:
            # Approximate worst-case z-statistic under Gamma-level confounding.
            # A confounder with odds ratio Gamma can shift the test statistic by up to
            # log(Gamma) in the large-sample limit (Rosenbaum & Rubin 1983 approximation).
            # As gamma increases, z_adj decreases → p_inflated increases (less significant).
            z_adj = max(0.0, z_obs - np.log(gamma))
            p_inflated = float(2 * (1 - norm.cdf(z_adj)))
            p_inflated = np.clip(p_inflated, 0.0, 1.0)
            p_values.append(p_inflated)
        
        return {
            'gamma': gamma_vals,
            'p_values': p_values,
            'critical_gamma': self._find_critical_gamma(gamma_vals, p_values),
            'n_treat': self.n_treat,
            'n_control': self.n_control,
            'observed_p_value': p_obs,
            'observed_z': z_obs
        }
    
    def _find_critical_gamma(self, gamma_vals, p_values, threshold=0.05):
        """Find the gamma value where p-value crosses the significance threshold"""
        for g, p in zip(gamma_vals, p_values):
            if p > threshold:
                return g
        return gamma_vals[-1]
    
    def interpret(self):
        """Generate interpretation of sensitivity analysis"""
        result = self.compute_gamma_range()
        critical = result['critical_gamma']
        p_obs = result['observed_p_value']
        
        interpretation = f"""
        ========== APPROXIMATE SENSITIVITY ANALYSIS ==========
        (Two-sample approximation inspired by Rosenbaum bounds; see module note for caveats)

        Sample sizes: Treated = {result['n_treat']}, Control = {result['n_control']}
        Observed |z|-statistic: {result['observed_z']:.3f}
        Observed p-value (two-sample t-test): {p_obs:.4f}

        Gamma is the odds ratio by which an unmeasured confounder could preferentially
        assign treatment. Gamma=1 means no unmeasured confounding.

        Critical Gamma: {critical:.2f}
        (At this level of confounding the result becomes non-significant at p=0.05)

        Interpretation:
        - Gamma < 1.5: Result is sensitive to moderate unmeasured confounding
        - Gamma 1.5–2.0: Moderately robust
        - Gamma > 2.0: Robust to substantial unmeasured confounding

        An unmeasured confounder would need odds ratio >= {critical:.2f} to overturn
        the observed result.
        """
        return interpretation

if __name__ == "__main__":
    from src.data_generator import CreditDataGenerator
    gen = CreditDataGenerator()
    _, panel = gen.generate()
    sens = RosenbaumSensitivity(panel)
    print(sens.interpret())
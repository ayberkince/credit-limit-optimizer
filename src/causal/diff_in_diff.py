"""
Difference-in-Differences estimator for robustness check.
Assumes parallel trends in pre-treatment period.
"""

import numpy as np
import pandas as pd
from scipy import stats

class DiffInDifferences:
    def __init__(self, df_panel: pd.DataFrame, treatment_month: int = 4):
        """
        Parameters:
        df_panel: DataFrame with columns: user_id, month, treatment_active, revenue
        treatment_month: month when treatment starts (0-indexed)
        """
        self.df = df_panel.copy()
        self.treatment_month = treatment_month
        self._prepare_data()
    
    def _prepare_data(self):
        """Split into pre and post periods, and treatment/control groups"""
        # Treatment group: users who ever receive treatment (treatment_assigned = 1)
        # But careful: in DiD, we compare those who got treatment vs those who didn't
        # Use treatment_assigned (not active) because assignment is at user level
        treat_users = self.df.groupby('user_id')['treatment_assigned'].max()
        self.treated_users = set(treat_users[treat_users == 1].index)
        self.control_users = set(treat_users[treat_users == 0].index)
        
        # Pre period: months < treatment_month
        pre_mask = self.df['month'] < self.treatment_month
        # Post period: months >= treatment_month
        post_mask = self.df['month'] >= self.treatment_month
        
        # Average revenue in each group and period
        self.pre_treat = self.df[pre_mask & self.df['user_id'].isin(self.treated_users)]['revenue'].mean()
        self.pre_control = self.df[pre_mask & self.df['user_id'].isin(self.control_users)]['revenue'].mean()
        self.post_treat = self.df[post_mask & self.df['user_id'].isin(self.treated_users)]['revenue'].mean()
        self.post_control = self.df[post_mask & self.df['user_id'].isin(self.control_users)]['revenue'].mean()
        
    def parallel_trends_test(self):
        """
        Test parallel trends assumption using pre-treatment periods.
        For simplicity, we test whether trends are parallel in the 2 periods before treatment.
        """
        # Use months before treatment: e.g., month 2 vs month 1 (if treatment_month=4)
        # We'll do a simple placebo test: assume treatment happened at a false earlier time
        placebo_month = self.treatment_month - 2
        if placebo_month < 1:
            return {'parallel_trends_assumption': 'Cannot test (insufficient pre-periods)', 'p_value': None}
        
        # For the placebo, we consider months before placebo as 'pre' and between placebo and treatment as 'post'
        pre_placebo = (self.df['month'] < placebo_month)
        post_placebo = (self.df['month'] >= placebo_month) & (self.df['month'] < self.treatment_month)
        
        # Compute DiD for placebo
        pre_treat = self.df[pre_placebo & self.df['user_id'].isin(self.treated_users)]['revenue'].mean()
        pre_control = self.df[pre_placebo & self.df['user_id'].isin(self.control_users)]['revenue'].mean()
        post_treat = self.df[post_placebo & self.df['user_id'].isin(self.treated_users)]['revenue'].mean()
        post_control = self.df[post_placebo & self.df['user_id'].isin(self.control_users)]['revenue'].mean()
        
        placebo_did = (post_treat - pre_treat) - (post_control - pre_control)
        
        # Bootstrap to test if placebo effect is significantly different from zero
        n_boot = 500
        boot_effects = []
        np.random.seed(42)
        for _ in range(n_boot):
            # Resample users with replacement
            treat_users_list = list(self.treated_users)
            control_users_list = list(self.control_users)
            boot_treat = np.random.choice(treat_users_list, len(treat_users_list), replace=True)
            boot_control = np.random.choice(control_users_list, len(control_users_list), replace=True)
            # Compute placebo DiD for bootstrap sample
            pre_treat_b = self.df[pre_placebo & self.df['user_id'].isin(boot_treat)]['revenue'].mean()
            pre_control_b = self.df[pre_placebo & self.df['user_id'].isin(boot_control)]['revenue'].mean()
            post_treat_b = self.df[post_placebo & self.df['user_id'].isin(boot_treat)]['revenue'].mean()
            post_control_b = self.df[post_placebo & self.df['user_id'].isin(boot_control)]['revenue'].mean()
            boot_effects.append((post_treat_b - pre_treat_b) - (post_control_b - pre_control_b))
        
        p_value = (np.abs(boot_effects) >= np.abs(placebo_did)).mean()
        return {'parallel_trends_assumption': f'Placebo DiD effect = {placebo_did:.2f}, p={p_value:.3f} (should be >0.05 to assume parallel trends)',
                'p_value': p_value}
    
    def estimate(self):
        """Compute DiD estimator: (T_post - T_pre) - (C_post - C_pre)"""
        diff_treat = self.post_treat - self.pre_treat
        diff_control = self.post_control - self.pre_control
        did = diff_treat - diff_control
        
        # Bootstrap confidence interval
        n_boot = 500
        boot_dids = []
        treat_users_list = list(self.treated_users)
        control_users_list = list(self.control_users)
        np.random.seed(42)
        for _ in range(n_boot):
            boot_treat = np.random.choice(treat_users_list, len(treat_users_list), replace=True)
            boot_control = np.random.choice(control_users_list, len(control_users_list), replace=True)
            pre_treat_b = self.df[self.df['month'] < self.treatment_month]['revenue'][self.df['user_id'].isin(boot_treat)].mean()
            pre_control_b = self.df[self.df['month'] < self.treatment_month]['revenue'][self.df['user_id'].isin(boot_control)].mean()
            post_treat_b = self.df[self.df['month'] >= self.treatment_month]['revenue'][self.df['user_id'].isin(boot_treat)].mean()
            post_control_b = self.df[self.df['month'] >= self.treatment_month]['revenue'][self.df['user_id'].isin(boot_control)].mean()
            boot_dids.append((post_treat_b - pre_treat_b) - (post_control_b - pre_control_b))
        
        ci_lower, ci_upper = np.percentile(boot_dids, [2.5, 97.5])
        
        return {
            'method': 'Difference-in-Differences',
            'ate': did,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': 'Assumes parallel trends; requires validation'
        }
    
    def run_full_analysis(self):
        """Run parallel trends test and DiD estimate"""
        trends = self.parallel_trends_test()
        did_result = self.estimate()
        return {**trends, **did_result}

if __name__ == "__main__":
    from src.data_generator import CreditDataGenerator
    gen = CreditDataGenerator()
    _, panel = gen.generate()
    did = DiffInDifferences(panel, treatment_month=4)
    results = did.run_full_analysis()
    print("Parallel Trends Test:", results['parallel_trends_assumption'])
    print(f"DiD Estimate: {results['ate']:.2f} (95% CI: {results['ci_lower']:.2f} - {results['ci_upper']:.2f})")
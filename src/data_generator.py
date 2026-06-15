"""
Synthetic panel data generator for credit limit optimization.
Calibrated to public dataset patterns, with known ground truth causal effects.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from typing import Tuple
import yaml

class CreditDataGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.n_users = self.config['data']['n_users']
        self.n_months = self.config['data']['n_months']
        self.treatment_month = self.config['data']['treatment_month']
        self.random_seed = self.config['data']['random_seed']
        np.random.seed(self.random_seed)

    def _generate_user_attributes(self) -> pd.DataFrame:
        """Generate static user attributes (income, credit score, etc.)"""
        inc_params = self.config['data']['income']
        income = np.random.lognormal(
            mean=inc_params['mu'],
            sigma=inc_params['sigma'],
            size=self.n_users
        )
        cs_params = self.config['data']['credit_score']
        credit_score = np.random.normal(
            loc=cs_params['mean'],
            scale=cs_params['sd'],
            size=self.n_users
        )
        credit_score = np.clip(credit_score, cs_params['min'], cs_params['max'])
        income_quintile = pd.qcut(income, q=5, labels=['low', 'medium_low', 'medium', 'medium_high', 'high'])
        return pd.DataFrame({
            'user_id': range(self.n_users),
            'income': income,
            'credit_score': credit_score,
            'income_quintile': income_quintile
        })

    def _assign_treatment(self, df_users: pd.DataFrame) -> pd.Series:
        bias = self.config['data']['treatment_bias']
        logit = (bias['intercept'] +
                 bias['income_coef'] * (df_users['income'] - df_users['income'].mean()) / df_users['income'].std() +
                 bias['credit_score_coef'] * (df_users['credit_score'] - df_users['credit_score'].mean()) / df_users['credit_score'].std())
        propensity = expit(logit)
        treatment = np.random.binomial(1, propensity)
        return treatment

    def _generate_panel(self, df_users: pd.DataFrame, treatment: pd.Series) -> pd.DataFrame:
        # <-- This line defines the method. Everything below MUST be indented.
        records = []
        cate_map = self.config['data']['true_cate']
        
        # Precompute income statistics for default model
        mean_income = df_users['income'].mean()
        std_income = df_users['income'].std()
        
        for _, user in df_users.iterrows():
            user_id = user['user_id']
            income_qt = user['income_quintile']
            base_revenue = 50 + 0.1 * user['income'] + np.random.normal(0, 10)
            cate = cate_map[income_qt]
            
            for month in range(self.n_months):
                time_trend = month * 2
                if treatment[user_id] == 1 and month >= self.treatment_month:
                    treat_effect = cate
                else:
                    treat_effect = 0
                revenue = base_revenue + time_trend + treat_effect + np.random.normal(0, self.config['data']['revenue_noise_sd'])
                revenue = max(0, revenue)
                default = 0  # placeholder, will set final month later
                records.append({
                    'user_id': user_id,
                    'month': month,
                    'treatment_assigned': treatment[user_id],
                    'treatment_active': 1 if (treatment[user_id] == 1 and month >= self.treatment_month) else 0,
                    'revenue': revenue,
                    'default': default,
                    'income': user['income'],
                    'credit_score': user['credit_score'],
                    'income_quintile': income_qt,
                    'monthly_spending_momentum': time_trend
                })
        
        df_panel = pd.DataFrame(records)
        
        # --- New default logic (logistic model) ---
        credit_norm = (df_users['credit_score'] - 600) / 100
        income_norm = (df_users['income'] - mean_income) / std_income
        risk_score = 0.3 * credit_norm - 0.2 * income_norm + 0.5 * treatment
        default_prob = 1 / (1 + np.exp(-risk_score))
        default_prob = np.clip(default_prob, 0.01, 0.5)
        
        default_final = (np.random.rand(self.n_users) < default_prob).astype(int)
        df_users['default_final'] = default_final
        
        # Assign default only in the last month
        df_panel.loc[df_panel['month'] == self.n_months-1, 'default'] = df_panel['user_id'].map(df_users.set_index('user_id')['default_final'])
        
        return df_panel

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_users = self._generate_user_attributes()
        treatment = self._assign_treatment(df_users)
        df_panel = self._generate_panel(df_users, treatment)
        return df_users, df_panel

if __name__ == "__main__":
    generator = CreditDataGenerator()
    users, panel = generator.generate()
    print(f"Generated {len(users)} users, {len(panel)} panel rows")
    print(panel.head())
    print("\nTreatment assignment summary:")
    print(panel['treatment_assigned'].value_counts())
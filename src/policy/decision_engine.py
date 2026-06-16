"""
Policy Decision Engine: Converts CATE estimates into business actions.
Computes expected Net Incremental Value (NIV) and optimal treatment policy.
"""

import numpy as np
import pandas as pd

class DecisionEngine:
    def __init__(self, cate_map, intervention_cost=50, avg_loan_loss=500, default_risk_increase=0.02):
        """
        Parameters:
        cate_map: dict mapping income_quintile (string) -> CATE (dollars per month)
        intervention_cost: cost of implementing the limit increase
        avg_loan_loss: average loss per default event
        default_risk_increase: absolute increase in default probability due to treatment
        """
        # Ensure keys are strings for robust lookup
        self.cate_map = {str(k): float(v) for k, v in cate_map.items()}
        self.cost = intervention_cost
        self.loss = avg_loan_loss
        self.default_risk_delta = default_risk_increase

    def expected_niv(self, income_quintile):
        """
        Compute expected Net Incremental Value for a user of given quintile.
        NIV = CATE - (default_risk_increase * avg_loan_loss) - intervention_cost
        """
        # Convert to string in case it's categorical or other type
        key = str(income_quintile)
        cate = self.cate_map.get(key, 0)
        expected_default_loss = self.default_risk_delta * self.loss
        niv = cate - expected_default_loss - self.cost
        return niv

    def recommend(self, income_quintile):
        """
        Return whether to treat (increase limit) and the expected NIV.
        """
        niv = self.expected_niv(income_quintile)
        return {
            'recommend_treatment': niv > 0,
            'expected_niv': niv,
            'cate': self.cate_map.get(str(income_quintile), 0),
            'default_loss_expected': self.default_risk_delta * self.loss
        }

    def apply_policy(self, df_users):
        """
        Apply policy to a DataFrame of users (must have 'income_quintile' column).
        Returns DataFrame with added columns: 'recommended_treatment', 'expected_niv'.
        """
        df = df_users.copy()
        # Ensure quintile column is string for consistent mapping
        df['income_quintile'] = df['income_quintile'].astype(str)
        df['recommended_treatment'] = df['income_quintile'].apply(
            lambda q: self.recommend(q)['recommend_treatment']
        )
        df['expected_niv'] = df['income_quintile'].apply(
            lambda q: self.recommend(q)['expected_niv']
        )
        # Force numeric type (in case of object)
        df['expected_niv'] = pd.to_numeric(df['expected_niv'], errors='coerce').fillna(0)
        return df

    def summary_report(self, df_users):
        """
        Generate a business summary: number of treated users, total expected NIV, etc.
        """
        df_policy = self.apply_policy(df_users)
        n_treat = df_policy['recommended_treatment'].sum()
        total_niv = df_policy['expected_niv'].sum()
        avg_niv = df_policy['expected_niv'].mean()

        # Breakdown by quintile
        breakdown = df_policy.groupby('income_quintile').agg(
            count=('income_quintile', 'size'),
            treat_count=('recommended_treatment', 'sum'),
            avg_niv=('expected_niv', 'mean'),
            total_niv=('expected_niv', 'sum')
        ).reset_index()

        report = f"""
        ========== POLICY DECISION REPORT ==========
        Intervention cost per user: ${self.cost}
        Expected default loss increase: ${self.default_risk_delta * self.loss:.2f} per treated user

        Recommended policy:
        - Total users: {len(df_users)}
        - Users to treat: {n_treat} ({100*n_treat/len(df_users):.1f}%)
        - Total expected NIV: ${total_niv:,.2f}
        - Average NIV per treated user: ${avg_niv:.2f}

        Breakdown by income quintile:
        {breakdown.to_string(index=False)}
        =============================================
        """
        return report

# Example usage if run standalone
if __name__ == "__main__":
    cate_map = {'low': 40, 'medium_low': 80, 'medium': 120, 'medium_high': 100, 'high': 60}
    engine = DecisionEngine(cate_map, intervention_cost=50, avg_loan_loss=500, default_risk_increase=0.02)
    users = pd.DataFrame({
        'income_quintile': ['low', 'medium', 'high', 'medium_low', 'medium_high']
    })
    report = engine.summary_report(users)
    print(report)
"""
Cox Proportional Hazards model for default timing.
Models the hazard of default as a function of treatment and covariates.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DefaultSurvivalModel:
    def __init__(self, df_panel: pd.DataFrame):
        self.df = df_panel.copy()
        self.model = None
        self.survival_df = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Build one-row-per-user right-censored survival DataFrame from panel."""
        survival_df = self.df.groupby('user_id').agg(
            event_observed=('default', 'max'),
            income=('income', 'first'),
            credit_score=('credit_score', 'first'),
            treatment_assigned=('treatment_assigned', 'first')
        ).reset_index()

        # Duration = month of first default for events; max observed month for censored
        event_months = self.df[self.df['default'] == 1].groupby('user_id')['month'].min()
        survival_df['duration'] = survival_df['user_id'].map(event_months)
        survival_df.loc[survival_df['event_observed'] == 0, 'duration'] = self.df['month'].max()

        self.survival_df = survival_df[
            ['user_id', 'income', 'credit_score', 'treatment_assigned', 'duration', 'event_observed']
        ].dropna()
        self.survival_df = self.survival_df[self.survival_df['duration'] > 0]

        n_events = self.survival_df['event_observed'].sum()
        print(f"Survival data prepared: {len(self.survival_df)} users, {n_events} default events")
        
    def fit(self):
        if self.survival_df is None or len(self.survival_df) == 0:
            print("No survival data available to fit model.")
            return None
        n_events = self.survival_df['event_observed'].sum()
        print(f"Number of default events: {n_events}")
        if n_events == 0:
            print("No default events in the dataset. Cannot fit survival model.")
            self.model = None
            return None
        if n_events < 20:
            print(f"Warning: Only {n_events} events (< 20). Cox model may be unstable.")
        
        self.model = CoxPHFitter(penalizer=0.1)
        try:
            self.model.fit(
                self.survival_df,
                duration_col='duration',
                event_col='event_observed',
                formula='treatment_assigned + income + credit_score'
            )
            print("Cox model fitted successfully.")
        except Exception as e:
            print(f"Full model failed: {e}")
            print("Falling back to treatment-only model...")
            try:
                self.model = CoxPHFitter(penalizer=0.1)
                self.model.fit(
                    self.survival_df,
                    duration_col='duration',
                    event_col='event_observed',
                    formula='treatment_assigned'
                )
                print("Treatment-only model fitted successfully.")
            except Exception as e2:
                print(f"Treatment-only model also failed: {e2}")
                self.model = None
        return self.model
    
    def hazard_ratio(self):
        if self.model is None:
            return None
        
        summary = self.model.summary
        hr = summary.loc['treatment_assigned', 'exp(coef)']
        
        # Prefer exp(coef) CI columns (already exponentiated); fall back to log-scale and exp manually
        lower_cols = [col for col in summary.columns if 'lower' in col.lower() and 'exp' in col.lower()]
        upper_cols = [col for col in summary.columns if 'upper' in col.lower() and 'exp' in col.lower()]

        if lower_cols and upper_cols:
            ci_lower = summary.loc['treatment_assigned', lower_cols[0]]
            ci_upper = summary.loc['treatment_assigned', upper_cols[0]]
        else:
            # Log-scale CI columns — exponentiate to get HR bounds
            log_lower_cols = [col for col in summary.columns if 'lower' in col.lower()]
            log_upper_cols = [col for col in summary.columns if 'upper' in col.lower()]
            if log_lower_cols and log_upper_cols:
                ci_lower = np.exp(summary.loc['treatment_assigned', log_lower_cols[0]])
                ci_upper = np.exp(summary.loc['treatment_assigned', log_upper_cols[0]])
            else:
                ci = self.model.confidence_intervals_.loc['treatment_assigned']
                lower_key = [k for k in ci.index if 'lower' in str(k).lower()][0]
                upper_key = [k for k in ci.index if 'upper' in str(k).lower()][0]
                ci_lower = np.exp(ci.loc[lower_key])
                ci_upper = np.exp(ci.loc[upper_key])
            
        direction = "increases" if hr > 1 else "decreases"
        percentage = abs(hr - 1) * 100
        
        return {
            'hazard_ratio': hr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': f"Treatment {direction} default hazard by {percentage:.1f}%"
        }
    
    def interpret(self):
        if self.model is None:
            return "No model fitted – please check if there are enough default events."
        hr_result = self.hazard_ratio()
        if hr_result is None:
            return "Could not compute hazard ratio."
        hr = hr_result['hazard_ratio']
        ci_l = hr_result['ci_lower']
        ci_u = hr_result['ci_upper']
        
        direction_phrase = "higher" if hr > 1 else "lower"
        magnitude = abs(hr - 1) * 100
        
        return f"""
        ========== DEFAULT SURVIVAL MODEL ==========
        
        Cox Proportional Hazards Model:
        - Hazard Ratio for Treatment: {hr:.2f}
        - 95% CI: [{ci_l:.2f}, {ci_u:.2f}]
        
        {hr_result['interpretation']}
        
        This means treated users default at an approximately {magnitude:.1f}% {direction_phrase}
        rate than untreated users, controlling for income and credit score.
        """
    
    def plot_survival(self, save_path=None):
        if self.model is None:
            print("No model to plot.")
            return None
    
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # 1. Grab the raw baseline cumulative hazard matrix from lifelines
        # This gives us a solid, uncorrupted pandas Series/DataFrame of the timeline
        baseline_cum_hazard = self.model.baseline_cumulative_hazard_
        
        # 2. Extract coefficients and calculate the mean baseline characteristics
        params = self.model.params_
        mean_income = self.survival_df['income'].mean()
        mean_credit = self.survival_df['credit_score'].mean()
        
        # Calculate the log-partial hazard components for covariates excluding treatment
        base_log_partial_hazard = (params.get('income', 0) * mean_income) + (params.get('credit_score', 0) * mean_credit)
        
        # Calculate log-partial hazards for both scenarios
        lph_treat0 = base_log_partial_hazard + (params.get('treatment_assigned', 0) * 0)
        lph_treat1 = base_log_partial_hazard + (params.get('treatment_assigned', 0) * 1)
        
        # 3. Transform cumulative hazards into survival curves: S(t) = exp(-H_0(t) * exp(beta*X))
        # This converts the baseline data into the correct arrays natively
        timeline = baseline_cum_hazard.index.values
        
        # Ensure we start cleanly at month 0 with 100% survival probability
        if 0 not in timeline:
            timeline = np.insert(timeline, 0, 0)
            s0_vals = np.insert(np.exp(-baseline_cum_hazard.iloc[:, 0].values * np.exp(lph_treat0)), 0, 1.0)
            s1_vals = np.insert(np.exp(-baseline_cum_hazard.iloc[:, 0].values * np.exp(lph_treat1)), 0, 1.0)
        else:
            s0_vals = np.exp(-baseline_cum_hazard.iloc[:, 0].values * np.exp(lph_treat0))
            s1_vals = np.exp(-baseline_cum_hazard.iloc[:, 0].values * np.exp(lph_treat1))
    
        # 4. Plot using explicit step configurations
        ax.step(timeline, s0_vals, label='Control (no treatment)', linewidth=2, where='post')
        ax.step(timeline, s1_vals, label='Treatment', linewidth=2, where='post')
    
        ax.set_xlabel('Months')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Survival Curves: Treated vs Untreated')
    
        # Dynamic, safe zooming parameters
        y_min = min(s0_vals.min(), s1_vals.min()) - 0.05
        ax.set_ylim(max(0.0, y_min), 1.02)
        ax.set_xlim(0, timeline.max() + 0.5)
    
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        return fig

if __name__ == "__main__":
    from src.data_generator import CreditDataGenerator
    gen = CreditDataGenerator()
    _, panel = gen.generate()
    surv = DefaultSurvivalModel(panel)
    surv.fit()
    if surv.model is not None:
        print(surv.interpret())
        surv.plot_survival()
    else:
        print("Could not fit survival model.")
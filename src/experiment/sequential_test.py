"""
Sequential testing using SPRT (Sequential Probability Ratio Test).
Allows early stopping without inflating Type I error.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class SequentialTest:
    def __init__(self, alpha=0.05, beta=0.20, mde=0.05, control_mean=100):
        """
        Parameters:
        alpha: Desired Type I error rate
        beta: Desired Type II error rate
        mde: Minimum detectable effect (proportion)
        control_mean: Expected mean of control group
        """
        self.alpha = alpha
        self.beta = beta
        self.mde = mde
        self.control_mean = control_mean
        self.effect = mde * control_mean
        
        # SPRT boundaries
        # A = (1-beta)/alpha, B = beta/(1-alpha)
        self.A = (1 - beta) / alpha
        self.B = beta / (1 - alpha)
        
    def log_likelihood_ratio(self, data_treat, data_control):
        """Compute log likelihood ratio for current data"""
        # Assume normal distribution with known variance (simplified)
        # For revenue, we can use approximate normality
        n_t = len(data_treat)
        n_c = len(data_control)
        if n_t == 0 or n_c == 0:
            return 0
        
        mean_t = np.mean(data_treat)
        mean_c = np.mean(data_control)
        # Pooled variance
        var_t = np.var(data_treat, ddof=1) if n_t > 1 else 1
        var_c = np.var(data_control, ddof=1) if n_c > 1 else 1
        pooled_var = ((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2) if (n_t + n_c) > 2 else 1
        
        # Log-likelihood under alternative (mean difference = effect)
        log_lik_alt = -0.5 * (np.sum((data_treat - (mean_c + self.effect))**2) / pooled_var +
                              np.sum((data_control - mean_c)**2) / pooled_var)
        # Log-likelihood under null (mean difference = 0)
        log_lik_null = -0.5 * (np.sum((data_treat - mean_c)**2) / pooled_var +
                               np.sum((data_control - mean_c)**2) / pooled_var)
        return log_lik_alt - log_lik_null
    
    def update(self, new_treat_obs, new_control_obs, current_treat=None, current_control=None):
        """
        Update sequential test with new observations.
        Returns decision: 'continue', 'stop_reject_null', or 'stop_accept_null'
        """
        if current_treat is None:
            current_treat = []
        if current_control is None:
            current_control = []
        current_treat.extend(new_treat_obs)
        current_control.extend(new_control_obs)
        
        lr = np.exp(self.log_likelihood_ratio(current_treat, current_control))
        
        if lr >= self.A:
            decision = 'stop_reject_null'  # Treatment is better
        elif lr <= self.B:
            decision = 'stop_accept_null'  # No significant difference
        else:
            decision = 'continue'
        
        return {
            'decision': decision,
            'lr': lr,
            'A': self.A,
            'B': self.B,
            'n_treat': len(current_treat),
            'n_control': len(current_control)
        }
    
    def simulate_trajectory(self, true_effect, n_max=5000, step=10):
        """
        Simulate experiment trajectory to visualize SPRT.
        true_effect: actual effect (could be 0 or mde*control_mean)
        """
        np.random.seed(42)
        control_data = np.random.normal(self.control_mean, 20, n_max)
        treat_data = np.random.normal(self.control_mean + true_effect, 20, n_max)
        
        lr_history = []
        decisions = []
        treat_accum = []
        control_accum = []
        
        for i in range(step, n_max+1, step):
            treat_subset = treat_data[:i]
            control_subset = control_data[:i]
            lr = np.exp(self.log_likelihood_ratio(treat_subset, control_subset))
            lr_history.append(lr)
            treat_accum.append(i)
            if lr >= self.A:
                decisions.append('reject')
                break
            elif lr <= self.B:
                decisions.append('accept')
                break
            else:
                decisions.append('continue')
        
        return treat_accum, lr_history, decisions
    
    def plot_sprt(self, true_effect=0, n_max=2000, step=10, save_path=None):
        """Plot SPRT boundaries and simulated trajectory"""
        n_obs, lr, decisions = self.simulate_trajectory(true_effect, n_max, step)
        plt.figure(figsize=(12, 6))
        plt.plot(n_obs, lr, 'b-', label='Likelihood Ratio')
        plt.axhline(y=self.A, color='r', linestyle='--', label=f'Reject H0 (A={self.A:.2f})')
        plt.axhline(y=self.B, color='g', linestyle='--', label=f'Accept H0 (B={self.B:.2f})')
        plt.yscale('log')
        plt.xlabel('Sample Size per Group')
        plt.ylabel('Likelihood Ratio (log scale)')
        plt.title(f'SPRT Monitoring (True Effect = ${true_effect:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        return plt.gcf()

if __name__ == "__main__":
    st = SequentialTest(alpha=0.05, beta=0.20, mde=0.05, control_mean=100)
    print(f"SPRT boundaries: A={st.A:.2f}, B={st.B:.2f}")
    print("\nSimulating under null (true effect = 0):")
    st.plot_sprt(true_effect=0)
    print("\nSimulating under alternative (true effect = 5):")
    st.plot_sprt(true_effect=5)
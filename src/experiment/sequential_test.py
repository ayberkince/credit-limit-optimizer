"""
Sequential testing using SPRT (Sequential Probability Ratio Test).
Allows early stopping without inflating Type I error.
Assumes known variance (pre-specified from historical data).
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

class SequentialTest:
    def __init__(self, alpha=0.05, beta=0.20, mde=0.05, control_mean=100, control_std=None):
        """
        Parameters:
        alpha: Desired Type I error rate
        beta: Desired Type II error rate
        mde: Minimum detectable effect (proportion)
        control_mean: Expected mean of control group
        control_std: Known standard deviation of outcome (required for simulation)
        """
        self.alpha = alpha
        self.beta = beta
        self.mde = mde
        self.control_mean = control_mean
        # If control_std not provided, estimate from typical coefficient of variation (20%)
        self.control_std = control_std if control_std is not None else control_mean * 0.2
        self.effect = mde * control_mean

        self.A = (1 - beta) / alpha
        self.B = beta / (1 - alpha)

    def log_likelihood_ratio(self, data_treat, data_control):
        """Compute log likelihood ratio for current data (assumes known variance)"""
        n_t = len(data_treat)
        n_c = len(data_control)
        if n_t == 0 or n_c == 0:
            return 0.0

        mean_t = np.mean(data_treat)
        mean_c = np.mean(data_control)
        var = self.control_std ** 2
        if var == 0:
            return 0.0

        # Avoid overflow by using log-sum-exp trick (though not needed here)
        log_lik_alt = -0.5 * (np.sum((data_treat - (mean_c + self.effect))**2) / var +
                              np.sum((data_control - mean_c)**2) / var)
        log_lik_null = -0.5 * (np.sum((data_treat - mean_c)**2) / var +
                               np.sum((data_control - mean_c)**2) / var)
        return log_lik_alt - log_lik_null

    def update(self, new_treat_obs, new_control_obs, current_treat=None, current_control=None):
        if current_treat is None:
            current_treat = []
        if current_control is None:
            current_control = []
        current_treat.extend(new_treat_obs)
        current_control.extend(new_control_obs)

        lr = np.exp(self.log_likelihood_ratio(current_treat, current_control))
        # Handle numerical issues
        if np.isnan(lr) or np.isinf(lr):
            if self.log_likelihood_ratio(current_treat, current_control) > 0:
                lr = self.A + 1  # force rejection
            else:
                lr = self.B - 1  # force acceptance

        if lr >= self.A:
            decision = 'stop_reject_null'
        elif lr <= self.B:
            decision = 'stop_accept_null'
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

    def simulate_trajectory(self, true_effect, n_max=2000, step=10):
        """Simulate experiment trajectory (uses known control_std)"""
        np.random.seed(42)
        control_data = np.random.normal(self.control_mean, self.control_std, n_max)
        treat_data = np.random.normal(self.control_mean + true_effect, self.control_std, n_max)

        lr_history = []
        treat_accum = []
        for i in range(step, n_max+1, step):
            treat_subset = treat_data[:i]
            control_subset = control_data[:i]
            # Use log-likelihood and exponentiate carefully
            log_lr = self.log_likelihood_ratio(treat_subset, control_subset)
            if np.isnan(log_lr):
                break
            # Clip to avoid overflow
            log_lr = np.clip(log_lr, -100, 100)
            lr = np.exp(log_lr)
            lr_history.append(lr)
            treat_accum.append(i)
            if lr >= self.A or lr <= self.B:
                break
        return treat_accum, lr_history

    def plot_sprt(self, true_effect=0, n_max=2000, step=10, save_path=None):
        """Plot SPRT boundaries and simulated trajectory"""
        n_obs, lr = self.simulate_trajectory(true_effect, n_max, step)
        if len(n_obs) == 0:
            print(f"Warning: No data for true_effect={true_effect}")
            return None
        plt.figure(figsize=(12, 6))
        plt.plot(n_obs, lr, 'b-', linewidth=2, label='Likelihood Ratio')
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
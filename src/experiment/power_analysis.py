import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class PowerAnalysis:
    def __init__(self, alpha=0.05, beta=0.20, mde=None, control_mean=None, control_std=None):
        """
        Parameters:
        alpha: significance level
        beta: Type II error (1 - power)
        mde: Minimum detectable effect IN THE SAME UNITS AS THE OUTCOME (e.g., dollars)
        control_mean: mean outcome in control group (unused but kept for compatibility)
        control_std: standard deviation of outcome
        """
        self.alpha = alpha
        self.beta = beta
        self.mde = mde  # now in dollars
        self.control_std = control_std

    def required_sample_size(self):
        """Compute sample size per group for two-sample t-test"""
        if self.mde is None or self.control_std is None:
            raise ValueError("mde and control_std must be provided")
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(1 - self.beta)
        d = self.mde / self.control_std  # Cohen's d
        n_per_group = int(np.ceil(2 * (z_alpha + z_beta)**2 / d**2))
        return max(n_per_group, 10)  # at least 10 per group

    def power_curve(self, sample_sizes=None):
        if sample_sizes is None:
            sample_sizes = np.arange(10, 5000, 50)
        d = self.mde / self.control_std
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        powers = []
        for n in sample_sizes:
            ncp = d * np.sqrt(n/2)
            crit = stats.t.ppf(1 - self.alpha/2, df=2*n - 2)
            power = 1 - stats.nct.cdf(crit, df=2*n - 2, nc=ncp)
            powers.append(power)
        return sample_sizes, powers

    def plot_power_curve(self, save_path=None):
        sample_sizes, powers = self.power_curve()
        plt.figure(figsize=(10,6))
        plt.plot(sample_sizes, powers, 'b-', linewidth=2)
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% power threshold')
        plt.xlabel('Sample Size per Group')
        plt.ylabel('Statistical Power')
        plt.title(f'Power Curve (MDE = ${self.mde:.2f}, α = {self.alpha})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        return plt.gcf()
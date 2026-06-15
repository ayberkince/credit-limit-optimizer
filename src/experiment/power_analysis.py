"""
Power analysis for experiment design.
Calculates required sample size and visualizes power curves.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class PowerAnalysis:
    def __init__(self, alpha=0.05, beta=0.20, mde=0.05, control_mean=100, control_std=50):
        """
        Parameters:
        alpha: Significance level (Type I error)
        beta: Type II error (1 - power)
        mde: Minimum detectable effect as a proportion (e.g., 0.05 for 5% lift)
        control_mean: Expected mean revenue in control group
        control_std: Expected standard deviation of revenue
        """
        self.alpha = alpha
        self.beta = beta
        self.mde = mde
        self.control_mean = control_mean
        self.control_std = control_std
        self.effect_size = mde * control_mean
        
    def required_sample_size(self):
        """Compute sample size per group for two-sample t-test"""
        # Standard formula for two-sample t-test with equal sizes
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(1 - self.beta)
        # Cohen's d
        d = self.effect_size / self.control_std
        n_per_group = int(np.ceil(2 * (z_alpha + z_beta)**2 / d**2))
        return n_per_group
    
    def power_curve(self, sample_sizes=None):
        """Generate power curve for a range of sample sizes"""
        if sample_sizes is None:
            sample_sizes = np.arange(100, 5000, 100)
        d = self.effect_size / self.control_std
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        powers = []
        for n in sample_sizes:
            # Non-centrality parameter for t-test
            ncp = d * np.sqrt(n/2)  # since n per group, total = 2n
            # Critical value
            crit = stats.t.ppf(1 - self.alpha/2, df=2*n - 2)
            # Power = 1 - probability of Type II error
            power = 1 - stats.nct.cdf(crit, df=2*n - 2, nc=ncp)
            powers.append(power)
        return sample_sizes, powers
    
    def plot_power_curve(self, figsize=(10,6)):
        """Plot power vs sample size"""
        sample_sizes, powers = self.power_curve()
        plt.figure(figsize=figsize)
        plt.plot(sample_sizes, powers, 'b-', linewidth=2)
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% power threshold')
        plt.xlabel('Sample Size per Group')
        plt.ylabel('Statistical Power')
        plt.title(f'Power Curve (MDE = {self.mde*100}%, α = {self.alpha})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('power_curve.png', dpi=100)
        plt.show()
        return plt.gcf()

if __name__ == "__main__":
    # Example with typical credit limit experiment
    pa = PowerAnalysis(alpha=0.05, beta=0.20, mde=0.05, control_mean=100, control_std=50)
    n = pa.required_sample_size()
    print(f"Required sample size per group: {n}")
    print(f"Total users needed: {2*n}")
    pa.plot_power_curve()
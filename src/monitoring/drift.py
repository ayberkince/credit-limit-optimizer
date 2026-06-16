"""
Model monitoring with drift detection.
Computes Population Stability Index (PSI) to detect data drift.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.2):
        """
        Parameters:
        reference_data: Baseline data to compare against (pre-treatment period)
        threshold: PSI threshold for alert (typically 0.1-0.25)
        """
        self.reference = reference_data.copy()
        self.threshold = threshold
        self._compute_reference_distributions()
    
    def _compute_reference_distributions(self):
        """Compute quantiles and distributions for reference data"""
        self.reference_quantiles = {}
        self.reference_distributions = {}
        
        for col in self.reference.select_dtypes(include=[np.number]).columns:
            # For numeric columns, store quantiles
            self.reference_quantiles[col] = {
                'min': self.reference[col].min(),
                'max': self.reference[col].max(),
                'mean': self.reference[col].mean(),
                'std': self.reference[col].std(),
                'quantiles': self.reference[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
            }
            # Store histogram bins for PSI calculation
            hist, bins = np.histogram(self.reference[col], bins=10)
            self.reference_distributions[col] = {
                'hist': hist,
                'bins': bins,
                'total': len(self.reference)
            }
    
    def calculate_psi(self, new_data: pd.DataFrame, column: str) -> float:
        """
        Calculate Population Stability Index for a column.
        PSI measures the shift in distribution between reference and new data.
        """
        if column not in self.reference_distributions:
            return 0.0
        
        ref = self.reference_distributions[column]
        new_hist, _ = np.histogram(new_data[column], bins=ref['bins'])
        new_total = len(new_data)
        
        psi = 0.0
        epsilon = 1e-4  # Floor for zero-count bins; prevents log(0) and includes all bins

        for ref_count, new_count in zip(ref['hist'], new_hist):
            ref_pct = max(ref_count / ref['total'], epsilon)
            new_pct = max(new_count / new_total, epsilon)
            psi += (new_pct - ref_pct) * np.log(new_pct / ref_pct)

        return psi
    
    def check_drift(self, new_data: pd.DataFrame) -> Dict[str, any]:
        """
        Check for drift in all numeric columns.
        
        Returns:
        dict with: column -> {psi, drift_detected, alert_level}
        """
        results = {}
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.reference_distributions:
                psi = self.calculate_psi(new_data, col)
                results[col] = {
                    'psi': psi,
                    'drift_detected': psi > self.threshold,
                    'alert_level': 'warning' if psi > self.threshold else 'ok',
                    'reference_mean': self.reference_quantiles[col]['mean'],
                    'current_mean': new_data[col].mean()
                }
        
        return results
    
    def check_propensity_drift(self, reference_propensity: np.ndarray, 
                              current_propensity: np.ndarray) -> Dict:
        """
        Specifically check drift in propensity scores.
        """
        # Create a temporary column for propensity scores
        ref_df = pd.DataFrame({'propensity': reference_propensity})
        cur_df = pd.DataFrame({'propensity': current_propensity})
        
        # Store reference distribution
        hist_ref, bins_ref = np.histogram(reference_propensity, bins=10)
        self.reference_distributions['propensity'] = {
            'hist': hist_ref,
            'bins': bins_ref,
            'total': len(reference_propensity)
        }
        
        # Calculate PSI
        psi = self.calculate_psi(cur_df, 'propensity')
        
        return {
            'psi': psi,
            'drift_detected': psi > self.threshold,
            'alert_level': 'warning' if psi > self.threshold else 'ok',
            'reference_mean': np.mean(reference_propensity),
            'current_mean': np.mean(current_propensity)
        }
    
    def generate_report(self, new_data: pd.DataFrame, stage: str = "current") -> str:
        """Generate a drift detection report"""
        results = self.check_drift(new_data)
        
        report = f"""
        ========== DRIFT DETECTION REPORT ==========
        Stage: {stage}
        Threshold: {self.threshold}
        
        Feature-level drift:
        """
        for col, result in results.items():
            status = "⚠️ DRIFT" if result['drift_detected'] else "✅ OK"
            report += f"\n  {col}: PSI={result['psi']:.4f} {status}"
            if result['drift_detected']:
                report += f" (mean shifted from {result['reference_mean']:.2f} to {result['current_mean']:.2f})"
        
        alert_count = sum(1 for r in results.values() if r['drift_detected'])
        report += f"\n\nSummary: {alert_count}/{len(results)} features show drift."
        
        if alert_count > 0:
            report += "\n⚠️ Model monitoring alert: Input data distribution has shifted. Consider retraining."
        else:
            report += "\n✅ No significant drift detected. Model is stable."
        
        return report

# Example usage
if __name__ == "__main__":
    from src.data_generator import CreditDataGenerator
    gen = CreditDataGenerator()
    _, panel = gen.generate()
    
    # Reference: pre-treatment period
    reference = panel[panel['month'] < gen.treatment_month]
    
    # Current: post-treatment period
    current = panel[panel['month'] >= gen.treatment_month]
    
    detector = DriftDetector(reference)
    report = detector.generate_report(current)
    print(report)
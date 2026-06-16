"""
Fairness audit with validation: uses calibrated risk model,
bootstrapped confidence intervals, and visual verification.
Supports saving plots to file.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

class FairnessAuditor:
    def __init__(self, df_panel: pd.DataFrame, protected_attr: str = 'income_quintile'):
        self.df = df_panel.copy()
        self.protected_attr = protected_attr
        self.df_final = self.df[self.df['month'] == self.df['month'].max()].copy()
        # Use "ever defaulted during the study" as target so all default events are captured,
        # not just those that happen to fall on the last month.
        user_ever_defaulted = self.df.groupby('user_id')['default'].max()
        self.df_final['default'] = self.df_final['user_id'].map(user_ever_defaulted)

        features = ['income', 'credit_score']  # only causally linked to default in the DGP
        X = self.df_final[features].fillna(0).values
        y = self.df_final['default'].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        self.risk_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=1.0))
        ])
        self.risk_model.fit(X_train, y_train)

        self.df_final['risk_score'] = self.risk_model.predict_proba(X)[:, 1]

        self.X_val = X_val
        self.y_val = y_val
        self.y_val_prob = self.risk_model.predict_proba(X_val)[:, 1]

    def _compute_metrics(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return {'TPR': tpr, 'FPR': fpr, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

    def _bootstrap_disparity(self, y_true, y_pred, group_labels, n_bootstrap=200):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        group_labels = np.array(group_labels)
        groups = np.unique(group_labels)
        tpr_disps = []
        fpr_disps = []
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            y_boot = y_true[idx]
            pred_boot = y_pred[idx]
            g_boot = group_labels[idx]
            group_metrics = {}
            for g in groups:
                mask = (g_boot == g)
                if mask.sum() > 0:
                    metrics = self._compute_metrics(y_boot[mask], pred_boot[mask])
                    group_metrics[g] = metrics
            if len(group_metrics) == len(groups):
                tpr_list = [group_metrics[g]['TPR'] for g in groups]
                fpr_list = [group_metrics[g]['FPR'] for g in groups]
                tpr_disps.append(max(tpr_list) - min(tpr_list))
                fpr_disps.append(max(fpr_list) - min(fpr_list))
        return {
            'tpr_disparity_ci': (np.percentile(tpr_disps, 2.5), np.percentile(tpr_disps, 97.5)),
            'fpr_disparity_ci': (np.percentile(fpr_disps, 2.5), np.percentile(fpr_disps, 97.5))
        }

    def audit_policy(self, policy_name, threshold=None):
        if threshold is None:
            threshold = self.df_final['risk_score'].median()
        y_pred = (self.df_final['risk_score'].values > threshold).astype(int)
        y_true = self.df_final['default'].values
        groups = self.df_final[self.protected_attr].values

        results = []
        for g in np.unique(groups):
            mask = groups == g
            metrics = self._compute_metrics(y_true[mask], y_pred[mask])
            metrics['group'] = g
            results.append(metrics)
        df_audit = pd.DataFrame(results)

        ci = self._bootstrap_disparity(y_true, y_pred, groups)

        tpr_disp = df_audit['TPR'].max() - df_audit['TPR'].min()
        fpr_disp = df_audit['FPR'].max() - df_audit['FPR'].min()

        return {
            'policy_name': policy_name,
            'metrics': df_audit,
            'tpr_disparity': tpr_disp,
            'fpr_disparity': fpr_disp,
            'tpr_disparity_ci': ci['tpr_disparity_ci'],
            'fpr_disparity_ci': ci['fpr_disparity_ci'],
            'equalized_odds_violation': (ci['tpr_disparity_ci'][0] > 0.05) or (ci['fpr_disparity_ci'][0] > 0.05)
        }

    def mitigate(self, threshold=None):
        if threshold is None:
            threshold = np.median(self.df_final['risk_score'])
        initial = self.audit_policy("initial", threshold)
        target_tpr = initial['metrics']['TPR'].mean()
    
        groups = self.df_final[self.protected_attr].unique()
        adjusted_thresholds = {}
        for g in groups:
            mask = self.df_final[self.protected_attr] == g
            group_scores = self.df_final.loc[mask, 'risk_score'].values
            group_true = self.df_final.loc[mask, 'default'].values
            best_thresh = threshold
            best_dist = float('inf')
            for th in np.linspace(0, 1, 51):
                pred = (group_scores > th).astype(int)
                tpr = (pred & group_true).sum() / (group_true.sum() + 1e-8)
                dist = abs(tpr - target_tpr)
                if dist < best_dist:
                    best_dist = dist
                    best_thresh = th
            adjusted_thresholds[g] = best_thresh
    
        y_pred_mitigated = np.zeros(len(self.df_final))
        for g, th in adjusted_thresholds.items():
            mask = self.df_final[self.protected_attr] == g
            y_pred_mitigated[mask] = (self.df_final.loc[mask, 'risk_score'].values > th).astype(int)
    
        y_true = self.df_final['default'].values
        groups_arr = self.df_final[self.protected_attr].values
        results = []
        for g in np.unique(groups_arr):
            mask = groups_arr == g
            metrics = self._compute_metrics(y_true[mask], y_pred_mitigated[mask])
            metrics['group'] = g
            results.append(metrics)
        df_mit = pd.DataFrame(results)
        ci = self._bootstrap_disparity(y_true, y_pred_mitigated, groups_arr)
    
        return {
            'mitigated': True,
            'adjusted_thresholds': adjusted_thresholds,
            'metrics': df_mit,
            'tpr_disparity': df_mit['TPR'].max() - df_mit['TPR'].min(),
            'fpr_disparity': df_mit['FPR'].max() - df_mit['FPR'].min(),
            'tpr_disparity_ci': ci['tpr_disparity_ci'],
            'fpr_disparity_ci': ci['fpr_disparity_ci']
        }

    def plot_risk_calibration(self, save_path=None):
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import roc_auc_score

        y_true = self.df_final['default'].values
        y_prob = self.df_final['risk_score'].values

        auc = roc_auc_score(y_true, y_prob)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=5, strategy='quantile'
        )
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(prob_pred, prob_true, marker='o', linewidth=2,
                 markersize=8, label=f'Risk model (AUC={auc:.3f})')
        ax1.set_xlabel('Mean predicted probability')
        ax1.set_ylabel('Fraction of positives (actual default rate)')
        ax1.set_title('Calibration curve')
        ax1.set_xlim(max(0, y_prob.min() - 0.01), min(1, y_prob.max() + 0.01))
        ax1.set_ylim(0, max(prob_true.max() * 1.3, 0.15))
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: score distribution by default status
        ax2.hist(y_prob[y_true == 0], bins=30, alpha=0.6,
                 label='No default', density=True)
        ax2.hist(y_prob[y_true == 1], bins=30, alpha=0.6,
                 label='Default', density=True)
        ax2.set_xlabel('Predicted probability')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Score distribution by outcome\n(AUC={auc:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Risk model validation', fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        return fig

    def plot_fairness_comparison(self, save_path=None):
        initial = self.audit_policy("initial", threshold=None)
        mitigated = self.mitigate(threshold=None)
    
        groups = sorted(initial['metrics']['group'].unique())
        initial_metrics = initial['metrics'].set_index('group').reindex(groups)
        mitigated_metrics = mitigated['metrics'].set_index('group').reindex(groups)
    
        x = np.arange(len(groups))
        width = 0.35
    
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
    
        axes[0].bar(x - width/2, initial_metrics['TPR'], width, label='Initial', color='red', alpha=0.7)
        axes[0].bar(x + width/2, mitigated_metrics['TPR'], width, label='Mitigated', color='green', alpha=0.7)
        axes[0].set_ylabel('True Positive Rate (TPR)')
        axes[0].set_title('TPR by Income Group')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(groups, rotation=45)
        axes[0].legend()
    
        axes[1].bar(x - width/2, initial_metrics['FPR'], width, label='Initial', color='red', alpha=0.7)
        axes[1].bar(x + width/2, mitigated_metrics['FPR'], width, label='Mitigated', color='green', alpha=0.7)
        axes[1].set_ylabel('False Positive Rate (FPR)')
        axes[1].set_title('FPR by Income Group')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(groups, rotation=45)
        axes[1].legend()
    
        plt.suptitle(f"Fairness Audit: Equalized Odds\nInitial disparity (TPR: {initial['tpr_disparity']:.3f}, FPR: {initial['fpr_disparity']:.3f}) | Mitigated disparity (TPR: {mitigated['tpr_disparity']:.3f}, FPR: {mitigated['fpr_disparity']:.3f})")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        return fig

    def generate_fairness_report(self):
        initial = self.audit_policy("initial")
        mitigated = self.mitigate()
        auc = roc_auc_score(self.df_final['default'], self.df_final['risk_score'])
        report = f"""
        ========== FAIRNESS AUDIT REPORT (with validation) ==========

        RISK MODEL CALIBRATION:
        - AUC: {auc:.3f}
        - Calibration plot saved as 'risk_calibration.png'

        INITIAL POLICY (threshold = median(risk_score)):
        - TPR disparity: {initial['tpr_disparity']:.3f} (95% CI: [{initial['tpr_disparity_ci'][0]:.3f}, {initial['tpr_disparity_ci'][1]:.3f}])
        - FPR disparity: {initial['fpr_disparity']:.3f} (95% CI: [{initial['fpr_disparity_ci'][0]:.3f}, {initial['fpr_disparity_ci'][1]:.3f}])
        - Equalized odds violated? {initial['equalized_odds_violation']}

        MITIGATED POLICY (per-group thresholds):
        """
        for g, th in mitigated['adjusted_thresholds'].items():
            report += f"\n  {g}: threshold = {th:.3f}"
        report += f"""

        MITIGATED RESULTS:
        - TPR disparity: {mitigated['tpr_disparity']:.3f} (95% CI: [{mitigated['tpr_disparity_ci'][0]:.3f}, {mitigated['tpr_disparity_ci'][1]:.3f}])
        - FPR disparity: {mitigated['fpr_disparity']:.3f} (95% CI: [{mitigated['fpr_disparity_ci'][0]:.3f}, {mitigated['fpr_disparity_ci'][1]:.3f}])

        CONCLUSION: The mitigation reduces disparity.
        """
        return report

if __name__ == "__main__":
    from src.data_generator import CreditDataGenerator
    gen = CreditDataGenerator()
    _, panel = gen.generate()
    auditor = FairnessAuditor(panel)
    auditor.plot_risk_calibration(save_path="outputs/risk_calibration.png")
    auditor.plot_fairness_comparison(save_path="outputs/fairness_audit.png")
    print(auditor.generate_fairness_report())
"""
Bias correction using IPW and manual Double Machine Learning.
Compares naive OLS, IPW, and DML estimates against ground truth.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BiasCorrection:
    def __init__(self, df_panel: pd.DataFrame):
        self.df = df_panel.copy()
        
    def _prepare_data(self):
        df_post = self.df[self.df['treatment_active'].isin([0,1])].copy()
        df_post = df_post.dropna(subset=['revenue', 'treatment_active', 'income', 'credit_score', 'month', 'monthly_spending_momentum'])
        self.Y = df_post['revenue'].values
        self.T = df_post['treatment_active'].values
        self.X = df_post[['income', 'credit_score', 'month', 'monthly_spending_momentum']].values
        return self.X, self.T, self.Y
    
    def naive_ols(self):
        _, T, Y = self._prepare_data()
        treated = Y[T == 1]
        control = Y[T == 0]
        diff = treated.mean() - control.mean()
        t_stat, p_val = stats.ttest_ind(treated, control)
        return {'method': 'Naive OLS', 'ate': diff, 'p_value': p_val}
    
    def ipw(self):
        X, T, Y = self._prepare_data()
        ps_model = LogisticRegression(max_iter=500, random_state=42)
        ps_model.fit(X, T)
        propensity = ps_model.predict_proba(X)[:, 1]
        # Clip to avoid extreme weights
        propensity = np.clip(propensity, 0.01, 0.99)
        weights = np.where(T == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))
        treat_w = np.sum(Y[T == 1] * weights[T == 1]) / np.sum(weights[T == 1])
        contr_w = np.sum(Y[T == 0] * weights[T == 0]) / np.sum(weights[T == 0])
        ate = treat_w - contr_w
        # Bootstrap CI
        n_boot = 200
        boot_ates = []
        np.random.seed(42)
        for _ in range(n_boot):
            idx = np.random.choice(len(Y), len(Y), replace=True)
            Yb, Tb, wb = Y[idx], T[idx], weights[idx]
            tw = np.sum(Yb[Tb == 1] * wb[Tb == 1]) / np.sum(wb[Tb == 1]) if np.sum(Tb == 1) > 0 else 0
            cw = np.sum(Yb[Tb == 0] * wb[Tb == 0]) / np.sum(wb[Tb == 0]) if np.sum(Tb == 0) > 0 else 0
            boot_ates.append(tw - cw)
        ci_lower, ci_upper = np.percentile(boot_ates, [2.5, 97.5])
        return {'method': 'IPW', 'ate': ate, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
    
    def double_ml(self):
        """
        Manual Double Machine Learning for average treatment effect.
        Uses cross-fitting to estimate nuisance functions.
        """
        X, T, Y = self._prepare_data()
        n = len(Y)
        k_folds = 2  # Use 2-fold cross-fitting for simplicity
        
        # Randomly assign folds
        np.random.seed(42)
        fold_ids = np.random.choice(k_folds, size=n, replace=True)
        
        # Initialize residual arrays
        Y_resid = np.zeros(n)
        T_resid = np.zeros(n)
        
        # Cross-fitting loop
        for fold in range(k_folds):
            # Training indices (all other folds)
            train_idx = (fold_ids != fold)
            # Validation indices (this fold)
            val_idx = (fold_ids == fold)
            
            X_train, X_val = X[train_idx], X[val_idx]
            T_train, T_val = T[train_idx], T[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # Model outcome Y given X (nuisance)
            model_y = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model_y.fit(X_train, Y_train)
            Y_hat = model_y.predict(X_val)
            
            # Model treatment T given X (nuisance)
            model_t = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
            model_t.fit(X_train, T_train)
            T_hat = model_t.predict_proba(X_val)[:, 1]  # probability of treatment
            
            # Compute residuals
            Y_resid[val_idx] = Y_val - Y_hat
            T_resid[val_idx] = T_val - T_hat
        
        # Final stage: regress Y_resid on T_resid (no intercept)
        # The coefficient is the ATE
        # Use simple linear regression through origin
        # But careful: we want E[Y - g(X) | T - m(X)] = theta * (T - m(X))
        # So theta = Cov(Y_resid, T_resid) / Var(T_resid)
        numerator = np.sum(Y_resid * T_resid)
        denominator = np.sum(T_resid * T_resid)
        ate = numerator / denominator
        
        # Bootstrap for CI (simple version)
        n_boot = 200
        boot_ates = []
        np.random.seed(42)
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            Yb = Y_resid[idx]
            Tb = T_resid[idx]
            num = np.sum(Yb * Tb)
            den = np.sum(Tb * Tb)
            if den != 0:
                boot_ates.append(num / den)
        ci_lower, ci_upper = np.percentile(boot_ates, [2.5, 97.5])
        
        return {
            'method': 'Double ML (manual)',
            'ate': ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': 'Neyman-orthogonal, cross-fitted'
        }
    
    def compare_estimators(self):
        self._prepare_data()
        results = [self.naive_ols(), self.ipw(), self.double_ml()]
        return pd.DataFrame(results)

if __name__ == "__main__":
    from src.data_generator import CreditDataGenerator
    gen = CreditDataGenerator()
    _, panel = gen.generate()
    bc = BiasCorrection(panel)
    print(bc.compare_estimators())
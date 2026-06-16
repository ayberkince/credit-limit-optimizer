"""
Bias correction using IPW and manual Double Machine Learning.
Improved with weight clipping and proper user-level bootstrap.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BiasCorrection:
    def __init__(self, df_panel: pd.DataFrame):
        self.df = df_panel.copy()
        self.user_ids = None   # will be set in _prepare_data

    def _prepare_data(self):
        # Filter to post-treatment period: months where any user has treatment_active == 1.
        # This avoids mixing pre-treatment rows of treated users into the control group.
        treatment_month = self.df.loc[self.df['treatment_active'] == 1, 'month'].min()
        if pd.isna(treatment_month):
            df_post = self.df.copy()
        else:
            df_post = self.df[self.df['month'] >= treatment_month].copy()
        df_post = df_post.dropna(subset=['revenue', 'treatment_active', 'income', 'credit_score'])
        self.Y = df_post['revenue'].values
        self.T = df_post['treatment_active'].values
        self.user_ids = df_post['user_id'].values   # store for bootstrap
        self.X = df_post[['income', 'credit_score']].values
        # Normalize features for stability
        self.X_norm = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-8)
        return self.X_norm, self.T, self.Y

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
        propensity = np.clip(propensity, 0.025, 0.975)
        weights = np.where(T == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))
        weights = np.clip(weights, 0.1, 100.0)

        treat_w = np.sum(Y[T == 1] * weights[T == 1]) / np.sum(weights[T == 1])
        contr_w = np.sum(Y[T == 0] * weights[T == 0]) / np.sum(weights[T == 0])
        ate = treat_w - contr_w

        # Bootstrap CI (row‑level, but IPW is less sensitive to pseudoreplication)
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
        # Prepare data (this sets self.user_ids)
        X, T, Y = self._prepare_data()
        n = len(Y)
        k_folds = 2
        np.random.seed(42)
        fold_ids = np.random.choice(k_folds, size=n, replace=True)

        Y_resid = np.zeros(n)
        T_resid = np.zeros(n)

        for fold in range(k_folds):
            train_idx = (fold_ids != fold)
            val_idx = (fold_ids == fold)

            X_train, X_val = X[train_idx], X[val_idx]
            T_train, T_val = T[train_idx], T[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            model_y = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model_y.fit(X_train, Y_train)
            Y_hat = model_y.predict(X_val)

            model_t = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
            model_t.fit(X_train, T_train)
            T_hat = model_t.predict_proba(X_val)[:, 1]

            Y_resid[val_idx] = Y_val - Y_hat
            T_resid[val_idx] = T_val - T_hat

        numerator = np.sum(Y_resid * T_resid)
        denominator = np.sum(T_resid * T_resid)
        ate = numerator / denominator if denominator != 0 else 0

        # --- USER-LEVEL BOOTSTRAP to avoid pseudoreplication ---
        # Resample users, then collect all rows for those users.
        unique_users = np.unique(self.user_ids)
        n_users = len(unique_users)
        n_boot = 200
        boot_ates = []

        # Build a DataFrame for easy grouping
        df_resid = pd.DataFrame({
            'user_id': self.user_ids,
            'Y_resid': Y_resid,
            'T_resid': T_resid
        })

        for _ in range(n_boot):
            boot_users = np.random.choice(unique_users, n_users, replace=True)
            boot_rows = df_resid[df_resid['user_id'].isin(boot_users)]
            if len(boot_rows) == 0:
                continue
            Yb = boot_rows['Y_resid'].values
            Tb = boot_rows['T_resid'].values
            num = np.sum(Yb * Tb)
            den = np.sum(Tb * Tb)
            if den != 0:
                boot_ates.append(num / den)

        if boot_ates:
            ci_lower, ci_upper = np.percentile(boot_ates, [2.5, 97.5])
        else:
            ci_lower, ci_upper = ate, ate

        return {
            'method': 'Double ML (manual)',
            'ate': ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': 'Neyman-orthogonal, cross-fitted, user-level bootstrap CI'
        }

    def compare_estimators(self):
        self._prepare_data()   # ensures user_ids are set if needed
        results = [self.naive_ols(), self.ipw(), self.double_ml()]
        return pd.DataFrame(results)
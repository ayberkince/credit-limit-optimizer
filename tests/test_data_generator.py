import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
from src.data_generator import CreditDataGenerator

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = CreditDataGenerator("config.yaml")

    def test_generation(self):
        users, panel = self.gen.generate()
        self.assertEqual(len(users), self.gen.n_users)
        self.assertEqual(len(panel), self.gen.n_users * self.gen.n_months)
        self.assertIn('treatment_active', panel.columns)
        self.assertIn('default', panel.columns)

    def test_treatment_bias(self):
        users, panel = self.gen.generate()
        # treatment should be correlated with income (confounding)
        treat_mean_income = panel[panel['treatment_assigned']==1]['income'].mean()
        control_mean_income = panel[panel['treatment_assigned']==0]['income'].mean()
        self.assertGreater(treat_mean_income, control_mean_income)

    def test_default_logic(self):
        users, panel = self.gen.generate()
        final_defaults = panel[panel['month'] == self.gen.n_months-1]['default']
        self.assertTrue(final_defaults.isin([0,1]).all())
        # default rate should be >0
        self.assertGreater(final_defaults.mean(), 0)

if __name__ == "__main__":
    unittest.main()
import pandas as pd
import numpy as np
import pickle
from scipy.stats import binom
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

# Load data
summary = pd.read_csv('summary.csv')

# Prepare the data for regression
X = summary[['avg_poss_x','avg_poss_y']].values

y_def = summary['Def'].values


# Train the regression model for Def
reg_def = RandomForestRegressor().fit(X, y_def)

with open('reg_def.pkl', 'wb') as f:
    pickle.dump(reg_def, f)


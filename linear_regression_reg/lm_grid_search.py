import os, sys, inspect
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -----Linear Regression Models-----
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor, Ridge
from sklearn.kernel_ridge import KernelRidge

# -----Import regressor_solver------
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import regressor_solver


input_file = "../dataset_/movie_metadata_cleaned_categ_num_only.csv"
dta = pd.read_csv(input_file)

dta_clean = dta
# remove the null values, that is fill NaN with there - FIXME: Rihards, naive implementation
dta_clean = dta_clean.fillna(value=0, axis=1)
dta_clean = dta_clean.dropna()
dta_clean = dta_clean.drop('Unnamed: 0', axis=1)


X_a = dta_clean.drop('worldwide_gross', axis=1)
y_a = dta_clean['worldwide_gross']

df_1 = dta_clean[dta_clean["worldwide_gross"] < 10000000]
X_1 = df_1.drop('worldwide_gross', axis=1)
y_1 = df_1['worldwide_gross']

df_2 = dta_clean[dta_clean["worldwide_gross"] >= 10000000]
df_2 = df_2[df_2["worldwide_gross"] < 300000000]
X_2 = df_2.drop('worldwide_gross', axis=1)
y_2 = df_2['worldwide_gross']

df_3 = dta_clean[dta_clean["worldwide_gross"] >= 300000000]
X_3 = df_3.drop('worldwide_gross', axis=1)
y_3 = df_3['worldwide_gross']

# shuffle the whole dataset
X_a, X_d, y_a, y_d = train_test_split(X_a, y_a, test_size=0, random_state=0)
X_1, X_d, y_1, y_d = train_test_split(X_1, y_1, test_size=0, random_state=0)
X_2, X_d, y_2, y_d = train_test_split(X_2, y_2, test_size=0, random_state=0)
X_3, X_d, y_3, y_d = train_test_split(X_3, y_3, test_size=0, random_state=0)


#########################
####### Models ##########
#########################
models = [LinearRegression(),
          PassiveAggressiveRegressor(C=0.001, n_iter=20),
          Ridge(alpha=0.01),
          KernelRidge(kernel='rbf', gamma=0.1)]
models_cfg = dict()

models_cfg[LinearRegression.__name__] = dict(
    model__fit_intercept=[True]
)

models_cfg[PassiveAggressiveRegressor.__name__] = dict(
    model__C= [1e-2, 1e-3, 1e-4]
)

models_cfg[Ridge.__name__] = dict(
    model__alpha= [0.1, 0.01, 0.001]
)

models_cfg[KernelRidge.__name__] = dict(
    model__alpha= [1e0, 0.1, 1e-2, 1e-3],
    model__gamma= np.logspace(-2, 2, 5).ravel()
)

# tuples_of_data = [(X_a,y_a, "all_samples"), (X_1,y_1, "samples_class1") , (X_2,y_2, "samples_class2"), (X_3,y_3, "samples_class3")]
tuples_of_data = [(X_a, y_a, "all_samples")]


#########################
### Start ###############
#########################
orig_stdout = sys.stdout  # save orig datetime and save orign stdout
time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
for ind, tupl in enumerate(tuples_of_data):
    # restart the current itterator for each run
    global itter_current
    itter_current = 0
    x_crr, y_crr, dsc = tupl
    trg = "regressRes_" + time + "_" + dsc + ".log"
    new_file = open(trg, "w")
    sys.stdout = new_file
    # set the itterator run to start from
    global itter_start
    itter_start = 0
    regressor_solver.run_for_many(x_crr, y_crr, dsc, models, models_cfg)
    new_file.close()
# reassign the org stdout for some reason
sys.stdout = orig_stdout

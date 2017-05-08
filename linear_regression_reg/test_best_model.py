import sys
from datetime import datetime
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

# -----Linear Regression Models-----
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor, Ridge
from sklearn.kernel_ridge import KernelRidge


# -----Util-------------------------
def log(X):
    df_t = pd.DataFrame(X[:, :10])
    X_t = df_t.replace(0, 1 / math.e)
    return np.concatenate((X, np.log(X_t)), axis=1)


logarithmicTransformer = FunctionTransformer(log)

# -----Main-------------------------
input_file = "../dataset_/no_imdb_names-count_cat-tf_184f.csv"
dta = pd.read_csv(input_file)

# remove the null values, that is fill NaN with there - FIXME: Rihards, naive implementation
dta_clean = dta.dropna()
#dta_clean = dta_clean.fillna(value=0, axis=1)
#dta_clean = dta_clean.dropna()
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
# X_a, X_d, y_a, y_d = train_test_split(X_a, y_a, test_size=0, random_state=0)
# X_1, X_d, y_1, y_d = train_test_split(X_1, y_1, test_size=0, random_state=0)
# X_2, X_d, y_2, y_d = train_test_split(X_2, y_2, test_size=0, random_state=0)
# X_3, X_d, y_3, y_d = train_test_split(X_3, y_3, test_size=0, random_state=0)

# # Tests
# y_1.hist()
# # Transform some feature to the log
# [print(x) for x in X_a.columns]
# X_a.production_budget = np.log(X_a.production_budget)
# y_a = y_a.replace(0, 1/math.e)
# y_a = np.log(y_a)

#########################
####### Models ##########
#########################
models = [LinearRegression(),
          PassiveAggressiveRegressor(C=0.001, n_iter=20),
          Ridge(alpha=0.01),
          KernelRidge(kernel='rbf', gamma=0.1)]
models_cfg = dict()


model = KernelRidge(alpha=0.01, gamma=0.01).fit(X_a, y_a)
print(model)
#score = model.score(X_a, y_a)
#print(score)

scores = cross_val_score(model, X_a, y_a, cv=4)
print(scores)



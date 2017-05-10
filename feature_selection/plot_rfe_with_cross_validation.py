"""
===================================================
Recursive feature elimination with cross-validation
===================================================

A recursive feature elimination example with automatic tuning of the
number of features selected with cross-validation.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Read data
path_full = "../dataset_/no_imdb_names-count_cat-tf_184f.csv"
#path_train = "../dataset_/no_imdb_names-count_cat-tf_184f_train.csv"
#path_test = "../dataset_/no_imdb_names-count_cat-tf_184f_test.csv"

dta_full = pd.read_csv(path_full)
# dta_full = dta_full.fillna(value=0, axis=1)
dta_full = dta_full.dropna()
dta_full = dta_full.drop('Unnamed: 0', axis=1)
dta_full_input = dta_full.drop('worldwide_gross', 1)
dta_full_target = dta_full['worldwide_gross']

X = dta_full_input.ix[:, 1:30]
y = dta_full_target


# DISCRETIZE THE TARGET VARIABLE
def label_gross_9(gross):
    if (gross < 500000):
        return 1
    elif ((gross >= 500000) & (gross < 5000000)):
        return 2
    elif ((gross >= 5000000) & (gross < 20000000)):
        return 3
    elif ((gross >= 20000000) & (gross < 50000000)):
        return 4
    elif ((gross >= 50000000) & (gross < 70000000)):
        return 5
    elif ((gross >= 70000000) & (gross < 125000000)):
        return 6
    elif ((gross >= 125000000) & (gross < 250000000)):
        return 7
    elif ((gross >= 250000000) & (gross < 550000000)):
        return 8
    elif (gross >= 550000000):
        return 9


y = y.apply(lambda gross: label_gross_9(gross))

# Build a classification task using 3 informative features

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
tree = ExtraTreesClassifier(n_estimators=250, random_state=0)

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=tree, step=1, cv=StratifiedKFold(4),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

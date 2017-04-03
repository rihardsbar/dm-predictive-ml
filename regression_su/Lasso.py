import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
# Input data files are available in the "../input/" directory.
from subprocess import check_output
input_folder = "../../moviedata"
# print(check_output(["ls", input_folder]).decode("utf8"))

import os
import pandas as pd
from pandas import DataFrame,Series
from sklearn import tree
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso as lo
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

# f = pd.read_csv(input_folder+"/movie_metadata.csv")
f = pd.read_csv(input_folder+"/movie_metadata_cleaned_categ_num_only.csv")

#data=DataFrame(f)
# data.head(10)

# np.sum(data.isnull())
data_2 = f.dropna()
# print( np.sum(data_2.isnull()) )
# print( data_2.shape )
# print( type(data_2) )

# Select all numerical features
# index_filter_num_all=data_2.dtypes[data.dtypes!='object'].index

y = data_2['worldwide_gross']
X = data_2.drop('worldwide_gross', axis=1)

# Split data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#shuffle the whole dataset
X_r, X_d, y_r, y_d = train_test_split(X, y, test_size=0, random_state=0)
max_score_tra = 0
para_tra = None

max_score_cv = 0
para_cv = None

for precompute in [True, False]:
    for fit_intercept in [True, False]:
        for normalize in [True, False]:
            for copy_X in [True, False]:
                for selection in ['cyclic', 'random']:
                    for warm_start in [True, False]:
                        for positive in [True, False]:
                    
                            clf = lo(precompute=precompute, fit_intercept = fit_intercept, normalize = normalize, 
                                copy_X = copy_X, selection = selection, warm_start = warm_start, positive = positive, random_state = 0)
                            clf1 = lo(precompute=precompute, fit_intercept = fit_intercept, normalize = normalize, 
                                copy_X = copy_X, selection = selection, warm_start = warm_start, positive = positive, random_state = 0)
                            clf.fit(X_train, y_train)
                            score_tra = clf.score(X_test, y_test)
                            score_cv = cross_val_score(clf1, X_r, y_r, cv=5)
                            if score_tra > max_score_tra:
                                max_score_tra = score_tra
                                para_tra =  clf.get_params

                            if score_cv.mean() > max_score_cv:
                            	max_score_cv = score_cv.mean()
                                para_cv =  clf1.get_params
        #                     y_pred = clf.predict(X_test)
        #                     print clf.get_params
# print clf.scores_
print max_score_tra
print para_tra

print max_score_cv
print para_cv


# scaler = preprocessing.StandardScaler().fit(X_r)
# X_train_transformed = scaler.transform(X_r)
# clf1 = br()
# clf2 = br()
# score = cross_val_score(clf1, X_r, y_r, cv=5)
# print score
# print score.mean()

# score2 = cross_val_score(clf2, X, y, cv=5)
# print score2
# print score2.mean()



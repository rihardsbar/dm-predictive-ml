#ignore warnings for clear printing
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns # More snazzy plotting library
import itertools

file_path = "./working_dataset.csv"
dta = pd.read_csv(file_path)
#clean up data non numeric rows
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in dta.iteritems():
    if type(colvalue[1]) == str:
        #if colname not in str_list:
            str_list.append(colname)
# Get to the numeric columns by inversion
num_list = dta.columns.difference(str_list)
#USe only the numeriv values
dta_clean = dta[num_list]
#remove the null values, that is fill NaN with there - FIXME: Rihards, naive implementation
#dta_clean = dta_clean.fillna(value=0, axis=1)
dta_clean = dta_clean.dropna()
dta_clean = dta_clean.reindex_axis(sorted(dta_clean.columns), axis=1)
#clean up data from zero rows
for colname, colvalue in dta_clean.iteritems():
    if colname != 'facenumber_in_poster':
        dta_clean = dta_clean[dta_clean[colname] != 0]
dta_clean.count()
#clasify the data for the logistic regression
def label_gross (gross):
    if (gross < 1000000) : return 1
    elif ((gross >= 1000000) & (gross < 10000000)) : return 2
    elif ((gross >= 10000000) & (gross < 50000000)) : return 3
    elif ((gross >= 50000000) & (gross < 200000000)) : return 4
    elif (gross >= 200000000) : return 5

y_target = dta_clean.gross.apply (lambda gross: label_gross (gross))
y_target = np.ravel(y_target)
x_data = dta_clean.drop('gross', axis=1)
#x_data = x_data.drop('aspect_ratio', axis=1)
#x_data = x_data.drop('duration', axis=1)
#x_data = x_data.drop('director_facebook_likes', axis=1)
dta_clean['gross_class'] = dta_clean.gross.apply (lambda gross: label_gross (gross))

#standarlisze
for colname, colvalue in x_data.iteritems():
                standard_scaler = preprocessing.StandardScaler().fit(x_data[colname])
                x_data[colname] = standard_scaler.transform(x_data[colname])


#calculate number of different componets in polynomal transfor
#poly_n_output_features = []
#for deg in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
#    poly = PolynomialFeatures(deg).fit(x_data)
#    poly_n_output_features.append(poly.n_output_features_)
#print poly_n_output_features
poly_n_output_features = [16, 136, 816, 3876, 15504, 54264, 170544, 490314, 1307504, 3268760, 7726160, 17383860, 37442160, 77558760, 155117520, 300540195]

def run_grid_search_pca(x,y,name, degrees, n_components, solvers, class_weight, Cs):
    #create pipline and use GridSearch to find the betst params
    polynomial = PolynomialFeatures().fit(x_data)
    pca = decomposition.PCA().fit(x_data)
    logistic = linear_model.LogisticRegression()
    pipe = Pipeline(steps=[('polynomial', polynomial),('pca', pca), ('logistic', logistic)])
    #create estimator
    estimator = GridSearchCV(pipe,
                             dict(polynomial__degree = degrees,
                                  pca__n_components = n_components,
                                  logistic__C = Cs,
                                  logistic__solver = solvers,
                                  logistic__class_weight = class_weight)
                             ,verbose=2, cv=5, n_jobs=4)
    #run the esmimator
    estimator.fit(x, y)
    print "GREP_ME***Results of ["  + name + "] estimator run are"
    print estimator.cv_results_
    print "GREP_ME***Best params of ["  + name + "] estimator run are"
    print estimator.best_params_
    print "GREP_ME***Best score of ["  + name + "] estimator run are"
    print estimator.best_score_

def run_grid_search_fa(x,y,name, degrees, n_components, solvers, class_weight, Cs):
    #create pipline and use GridSearch to find the betst params
    polynomial = PolynomialFeatures().fit(x_data)
    fa = decomposition.FactorAnalysis().fit(x_data)
    logistic = linear_model.LogisticRegression()
    pipe = Pipeline(steps=[('polynomial', polynomial),('fa', fa), ('logistic', logistic)])
    #create estimator
    estimator = GridSearchCV(pipe,
                             dict(polynomial__degree = degrees,
                                  fa__n_components = n_components,
                                  logistic__C = Cs,
                                  logistic__solver = solvers,
                                  logistic__class_weight = class_weight)
                             ,verbose=2, cv=5, n_jobs=4)
    #run the esmimator
    estimator.fit(x, y)
    print "GREP_ME***Results of ["  + name + "] estimator run are"
    print estimator.cv_results_
    print "GREP_ME***Best params of ["  + name + "] estimator run are"
    print estimator.best_params_
    print "GREP_ME***Best score of ["  + name + "] estimator run are"
    print estimator.best_score_

def run():
    ####################################################
    # Process a simple first degree pipeline with PCA  #
    ###################################################
    degrees=[1]
    n_components = [4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
    solvers =  ['newton-cg', 'lbfgs', 'liblinear', 'sag']
    class_weight = [None, "balanced"]
    Cs = np.logspace(-4, 4, 3)
    run_grid_search_pca(x_data, y_target, "First degree, PCA - no mle", degrees, n_components, solvers, class_weight, Cs)

    ###############################################
    # Process a 5 degree pipeline with  PCA mle  #
    ###############################################

    #cannot go more than three as there are less samples than features eventually
    degrees=[1,2,3]
    n_components = ['mle']
    solvers =  ['newton-cg', 'lbfgs', 'liblinear', 'sag']
    class_weight = [None, "balanced"]
    Cs = np.logspace(-4, 4, 3)
    run_grid_search_pca(x_data, y_target, "3 degrees, PCA with mle", degrees, n_components, solvers, class_weight, Cs)

    ##########################################
    # Process a 3 degree pipeline with FA   #
    ##########################################
    for deg in [1,2,3]:
        degrees=[deg]
        n_components = [4, 5, 6, 7, 8, 9, 10, 11, 12, poly_n_output_features[deg-1] - 1]
        solvers =  ['newton-cg', 'lbfgs', 'liblinear', 'sag']
        class_weight = [None, "balanced"]
        Cs = np.logspace(-4, 4, 3)
        print n_components
        string = str(deg) + " degree, FA"
        run_grid_search_fa(x_data, y_target, string, degrees, n_components, solvers, class_weight, Cs)

#ignore warnigs
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    run()

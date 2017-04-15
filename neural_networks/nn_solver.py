#import all helpers
import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from patsy import dmatrices
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.feature_selection import GenericUnivariateSelect, RFE
from sklearn import metrics
from sklearn import linear_model, decomposition, datasets
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression, mutual_info_regression
import itertools
from itertools import  product
import pprint

#import regressors
#----Ensemble and Generalise Lin. Models---------------
from sklearn.ensemble import       ExtraTreesRegressor
from sklearn.linear_model import   LinearRegression


#----Neural Networks--------------- 1
from sklearn.neural_network import MLPRegressor
#-----Support Vector Machines------ 3
from sklearn.svm import            SVR
from sklearn.svm import            LinearSVR
from sklearn.svm import            NuSVR
#----Ensemble---------------------- 1
from sklearn.ensemble import       AdaBoostRegressor
#----Isotonic ---------------------1
from sklearn.isotonic import         IsotonicRegression
#---Gausian------------------------1
from sklearn.gaussian_process import GaussianProcessRegressor



#file_path =  "../dataset/movie_metadata_cleaned_tfidf_num_only_min.csv"
file_path =  "../dataset/movie_metadata_cleaned_categ_num_only.csv"
#file_path = "../dataset/movie_metadata_cleaned_no_vector_num_only.csv"

dta = pd.read_csv(file_path)
dta_clean = dta
#remove the null values, that is fill NaN with there - FIXME: Rihards, naive implementation
dta_clean = dta_clean.fillna(value=0, axis=1)
dta_clean = dta_clean.dropna()
dta_clean = dta_clean.drop('Unnamed: 0', axis=1)

#models = [AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor,BayesianRidge,ElasticNet,HuberRegressor,Lars,Lasso,LassoLars,LinearRegression,PassiveAggressiveRegressor,Ridge,SGDRegressor,OrthogonalMatchingPursuit,RANSACRegressor,TheilSenRegressor,KNeighborsRegressor,RadiusNeighborsRegressor,MLPRegressor,SVR,LinearSVR,NuSVR,DecisionTreeRegressor,ExtraTreeRegressor]

##define helpers 
def get_powers_list(n_samples, n_features, n):
    base_arr = [{"pw":2},{"pw":3},{"pw":4}]
    max_pw = math.ceil(3320/n_features)
    if max_pw > 7: max_pw = 7
    step = math.floor((max_pw-4) / n)
    if step < 1 : step = 1
    extra_arr = [{"pw":power} for power in range(4 + step, max_pw, step)]
    if  n_samples/n_features < 2:
        res = [{"pw":1}]
    elif max_pw - 1 == 2:
        res = [{"pw":2}]
    elif max_pw - 1 == 3:
        res = [{"pw":2}, {"pw":3}]
    elif max_pw - 1 == 4:
        res = [{"pw":2},{"pw":3},{"pw":4}]
    else :
        res = base_arr + extra_arr
    return res

def get_components_list(n_features, lst):
    lst = lst + [{"pw": 0.1},{"pw": 0.4},{"pw": 0.5},{"pw": 0.8}]
    lst = sorted(list(map(lambda x: math.floor(x["pw"]*n_features), lst)) + [1, 3, 5], reverse=True)
    lst[0] = lst[0]-1
    lst_n = [n for n in lst if n < 3321]
    if len(lst_n) < len(lst):
        lst_n = [3320] + lst_n 
    return lst_n

##define new transformers
def dummy(X):  
    return X

def poly(X, pw):
    res = X
    for power in range(2,pw + 1):
        res = np.concatenate((res, np.power(X, power)), axis=1)
    return res

def log(X):
    df_t = pd.DataFrame(X)
    X_t = df_t.replace(0, 1/math.e) 
    return np.concatenate((X, np.log(X_t)), axis=1)


DummyTransformer = FunctionTransformer(dummy)
LogarithmicTransformer = FunctionTransformer(log)
PolynomialTransformer = FunctionTransformer(poly)

###define new config###########

#########################
####Data Preprocessor ###
#########################
#preprocessors = [DummyTransformer, LogarithmicTransformer, PolynomialTransformer]
preprocessors = [DummyTransformer]
preprocessors_cfg = {}
preprocessors_cfg[DummyTransformer.func.__name__] = {}
preprocessors_cfg[LogarithmicTransformer.func.__name__] = {}
preprocessors_cfg[PolynomialTransformer.func.__name__] = dict(
        preprocessor__kw_args = []
        )
#########################
####  Data Transformer ##
#########################
#transfomers = [DummyTransformer, Normalizer(), StandardScaler()]
transfomers = [DummyTransformer]
transfomers_cfg = {}
transfomers_cfg[DummyTransformer.func.__name__] = {}
transfomers_cfg[Normalizer.__name__] = dict(
        transfomer__norm = ['l1', 'l2', 'max']
        )
transfomers_cfg[StandardScaler.__name__] = {}
###########################
####Dim Reducer, Feat Sel.#
###########################
#reducers = [DummyTransformer, PCA(), GenericUnivariateSelect(), RFE(ExtraTreesRegressor())]
reducers = [DummyTransformer]
reducers_cfg = {}
reducers_cfg[DummyTransformer.func.__name__] = {}
reducers_cfg[PCA.__name__] = dict(
        reducer__n_components = [],
        reducer__whiten = [True, False],
        reducer__svd_solver = ['auto']
        )
reducers_cfg[GenericUnivariateSelect.__name__] = dict(
        reducer__score_func = [f_regression],
        reducer__mode = ['k_best'],
        reducer__param = []
        )
reducers_cfg[RFE.__name__] = dict(
        reducer__n_features_to_select = [],
        reducer__step = [0.1]
        )
#########################
####### Models ##########
#########################
#models = [MLPRegressor(), SVR(), LinearSVR(), NuSVR(), AdaBoostRegressor(), IsotonicRegression(), GaussianProcessRegressor()]
models = [NuSVR()]
models_cfg = {}
models_cfg[MLPRegressor.__name__] = dict(
    model__hidden_layer_sizes = [(50,), (200,), (500,)],
    model__activation = ['identity', 'logistic', 'tanh', 'relu'],
    model__solver = ['adam', 'lbfgs'],
    #model__learning_rate = ['constant', 'invscaling', 'adaptive'],
    model__max_iter = [200, 500],  
    model__shuffle = [True, False]  
)
models_cfg[SVR.__name__] = dict(
    model__C = [0.8, 1.0],
    model__kernel = ['rbf', 'sigmoid'],
    model__shrinking = [True, False]
)

models_cfg[LinearSVR.__name__] = dict(
    model__C = [0.8, 1.0],
    model__loss = ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    model__epsilon = [0, 0.1],
    model__fit_intercept = [True, False],
    model__max_iter = [1000, 2000]
)

models_cfg[NuSVR.__name__] = dict(
    model__C = [0.8, 1.0],
    model__nu = [0.5, 0.8, 1.0],
    model__kernel = ['rbf', 'sigmoid'],
    model__shrinking = [True, False]
)

def run_grid_search(x,y,preprocessor, transfomer, reducer, model, results, errors, errors_ind):
    
    #create pipline and use GridSearch to find the best params for given pipeline
    name = type(model).__name__
    preprocessor_name = type(preprocessor).__name__ if (type(preprocessor).__name__ != "FunctionTransformer") else preprocessor.func.__name__
    transfomer_name =  type(transfomer).__name__ if (type( transfomer).__name__ != "FunctionTransformer") else  transfomer.func.__name__
    reducer_name = type(reducer).__name__ if (type(reducer).__name__ != "FunctionTransformer") else reducer.func.__name__
    
    #Define and save pipe cfg
    pipeline_cfg = "| preprocessor:" + preprocessor_name +  " | transfomer: " + transfomer_name + " | reducer: " + reducer_name
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('transfomer', transfomer), ('reducer', reducer),('model', model)])
    
    #create a dict with param grid
    param_grid = dict(models_cfg[name], **dict(reducers_cfg[reducer_name], **dict(transfomers_cfg[transfomer_name], **preprocessors_cfg[preprocessor_name])))
    #create estimator
    cv = 4
    print('####################################################################################')
    print()
    print('####################################################################################')
    print ("***Starting ["  + name + "] estimator run, pipeline: "+ pipeline_cfg+" ")
    print("##param_grid##")
    print(param_grid)
    estimator = GridSearchCV(pipe,param_grid,verbose=2, cv=cv, n_jobs=1)
    #run the esmimator, except eceptions, sape errors
    try:
            estimator.fit(x, y)
            print ("GREP_ME***Results of ["  + name + "] estimatorrun are")
            print (estimator.cv_results_)
            print ("GREP_ME***Best params of ["  + name + "] estimator,pipeline:"+ pipeline_cfg+"  run are")
            print (estimator.best_params_)
            print ("GREP_ME***Best score of ["  + name + "] estimator, pipeline:"+ pipeline_cfg+" run are")
            print (estimator.best_score_)
            if (name not in results) or (estimator.best_score_ > results[name]["score"]):
                results[name] = {"score": estimator.best_score_, "pipe":pipeline_cfg, "best_cfg": estimator.best_params_}
    except (ValueError, MemoryError) as err:
            print ("GREP_ME***Error caught for  ["  + name + "] , pipeline: ["+ pipeline_cfg+"] ")
            errors_ind.append({"cfg": "Model["+ name +"] pipe: " + pipeline_cfg})
            errors.append({"Model["+ name +"] pipe: " + pipeline_cfg: {"error": err}})
            pass
            
def run_solver(x,y,preprocessors, transfomers, reducers, models, results, errors, errors_ind):
    # mix it, so that the sample order is randomized
    x, _X_dummy, y, _y_dummy = train_test_split(x, y, test_size=0)
    n_samples, n_features = x.shape
    for preprocessor, transfomer, reducer, model in product(preprocessors, transfomers, reducers, models):
        ##run gridesearch with new amout of features, depending of preprocessor and hence pass the right amount of maximum components to the reducers
        if preprocessor.func.__name__ == LogarithmicTransformer.func.__name__ :
            n_components = get_components_list(n_features, [{"pw":2}, {"pw":1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            run_grid_search(x,y,preprocessor, transfomer, reducer, model, results, errors, errors_ind)
        elif preprocessor.func.__name__ == PolynomialTransformer.func.__name__:
            kw_arg_powers = get_powers_list(n_samples, n_features, 3)
            pw_lst = []
            for pw in kw_arg_powers:
                pw_lst = pw_lst + [pw]
                preprocessors_cfg[PolynomialTransformer.func.__name__]["preprocessor__kw_args"] = [pw]
                n_components = get_components_list(n_features, pw_lst)
                reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
                reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
                reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
                run_grid_search(x,y,preprocessor, transfomer, reducer, model, results, errors, errors_ind)
        else:
            n_components = get_components_list(n_features, [{"pw":1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            run_grid_search(x,y,preprocessor, transfomer, reducer, model, results, errors, errors_ind)

##function for trigrering gridserach and priting results
def run_for_many(x,y, cl_n):
    results = {}
    errors = []
    errors_ind = []
    print ("#########################################")
    print ("###Starting all estimators for cl: "+ str(cl_n))
    print ("#########################################")
    run_solver(x,y, preprocessors, transfomers, reducers, models, results, errors, errors_ind)
    print ("#########################################")
    print ("###Finished all estimators for cl: "+ str(cl_n))
    print ("#########################################")

    print ("#########################################")
    print ("######Printing all errors for cl: "+ str(cl_n))
    print ("#########################################")
    print(errors)
    print ("#########################################")
    print ("######Printing errors summary for cl: "+ str(cl_n))
    print ("#########################################")
    print(errors_ind)
    print ("#########################################")
    print ("#######Printing results for cl: "+ str(cl_n))
    print ("#########################################")
    print(results)
    print("priting simply sorted numbers, grep them to find the best cfg or cl: "+ str(cl_n))
    scores = [results[model]["score"] for model in results]
    print(sorted(scores))

            
##to run for multiple classes of data, add the tuples of x and y  to the tuples array of data and decsription for the purposes logging. For now it is set to run for all the samples there are. For instance tuples_of_data = [(X,y, "all samples"), (X_1,y_1, "samples class1") , (X_2,y_2", "samples class2")]
#for each tupple extracted from the array a new log file is going to be generated, so that each run is in a different log file.
X = dta_clean.drop('worldwide_gross', axis=1)
y = dta_clean['worldwide_gross']

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

#tuples_of_data = [(X,y, "all_samples"), (X_1,y_1, "samples_class1") , (X_2,y_2, "samples_class2"), (X_3,y_3, "samples_class3")]

tuples_of_data = [(X,y, "all_samples")]

#save orig datetime and save orign stdout
orig_stdout = sys.stdout
time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
for ind, tupl in enumerate(tuples_of_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_crr, y_crr, dsc = tupl
        trg = "regressRes_" + time + "_" + dsc + ".log"
        new_file = open(trg,"w")
        sys.stdout = new_file
        run_for_many(x_crr, y_crr, dsc)
        new_file.close()
#reassign the org stdout for some reason
sys.stdout = orig_stdout
import os
import sys
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.feature_selection import GenericUnivariateSelect, RFE
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression, mutual_info_regression
from itertools import product
from multiprocessing import Process, Value, Array
import pickle
import shutil
from sklearn.model_selection import train_test_split

##impot classifieris
#-----Ensemble---------------------
from sklearn.ensemble import       AdaBoostClassifier
from sklearn.ensemble import       BaggingClassifier
from sklearn.ensemble import       ExtraTreesClassifier
from sklearn.ensemble import       GradientBoostingClassifier
from sklearn.ensemble import       RandomForestClassifier

#----Generalized Linear models-----
from sklearn.linear_model import   PassiveAggressiveClassifier
from sklearn.linear_model import   LogisticRegression
from sklearn.linear_model import   RidgeClassifier
from sklearn.linear_model import   SGDClassifier

#-----Naive Bayes ---all class-----
from sklearn.naive_bayes import    GaussianNB
from sklearn.naive_bayes import    MultinomialNB
from sklearn.naive_bayes import    BernoulliNB

#---Nearest Neighbors--------------
from sklearn.neighbors import      KNeighborsClassifier
from sklearn.neighbors import      RadiusNeighborsClassifier
from sklearn.neighbors import      NearestCentroid

#----Neural Networks---------------
from sklearn.neural_network import MLPClassifier

#-----Support Vector Machines------
from sklearn.svm import            SVC
from sklearn.svm import            LinearSVC
from sklearn.svm import            NuSVC

#-----Decission Trees--------------
from sklearn.tree import           DecisionTreeClassifier
from sklearn.tree import           ExtraTreeClassifier

# -----Ensemble---------------------
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# ----Generalized Linear models-----
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor

# ---Nearest Neighbors----
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

# ----Neural Networks---------------
from sklearn.neural_network import MLPRegressor

# -----Support Vector Machines------
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR

# -----Decission Trees--------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

# ----extras
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge

from sklearn.base import BaseEstimator, TransformerMixin

class ClassifierTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, label_func):
        self.x_trans = 0
        self.y_cl = 0
        self.label_func = label_func
        self.model =  GradientBoostingClassifier(n_estimators = 1000,
                      learning_rate = 0.01,
                      max_depth =  10,
                      min_samples_leaf = 5,
                      max_features = 0.01,
                      max_leaf_nodes =  None)

    def fit(self, X, y):
        print("fitting classifier")
        print("x len is:", end = " ")
        print(len(X), end = " ")
        print("y len is:", end = " ")
        print(len(y))
        self.y_cl = y.map(self.label_func)
        self.model.fit(X,self.y_cl)
        print("finished")
        return self

    def transform(self, X):
        print("transforming classifier")
        print("x len is:", end = " ")
        print(len(X))
        self.x_trans = pd.concat([pd.DataFrame(self.model.predict(X)), pd.DataFrame(X)], axis=1)
        print("finished")
        return self.x_trans


file_path =  "../dataset/no_imdb_names-count_cat-tf_184f_train.csv"


dta = pd.read_csv(file_path)
dta_clean = dta
#remove the null values, that is fill NaN with there - FIXME: Rihards, naive implementation
dta_clean = dta_clean.fillna(value=0, axis=1)
dta_clean = dta_clean.dropna()
dta_clean = dta_clean.drop('Unnamed: 0', axis=1)

##define helpers
def get_powers_list(n_samples, n_features, n):
    return [{"pw":1},{"pw":2},{"pw":3},{"pw":4}]

def get_components_list(n_features, lst, log_poly = False):
    max_pw = max(lst, key=lambda x: x["pw"])["pw"]
    current_feat = 10*max_pw + n_features - 10
    if log_poly: current_feat = 10*max_pw + n_features
    #lst = [{"pw": 0.1},{"pw": 0.45},{"pw": 0.5},{"pw": 0.8}, {"pw": 0.2},{"pw": 0.65},{"pw": 0.99}]
    #lst = [{"pw": 0.2}, {"pw": 0.28}, {"pw": 0.36}, {"pw": 0.44},{"pw": 0.52}, {"pw": 0.6}]
    lst = [{"pw": 0.3}, {"pw": 0.6}, {"pw": 1}]
    lst = sorted(list(map(lambda x: math.floor(x["pw"]*current_feat), lst)), reverse=True)
    return lst


##define new transformers
def dummy(X):
    return X

def poly(X, pw):
    vector = X[:,10:]
    res    = X[:,:10]
    X      = X[:,:10]
    for power in range(2,pw + 1):
        res = np.concatenate((res, np.power(X, power)), axis=1)
    return np.concatenate((res, vector), axis=1)

def log(X):
    df_t = pd.DataFrame(X[:,:10])
    X_t = df_t.replace(0, 1/math.e)
    return np.concatenate((X, np.log(X_t)), axis=1)

def log_poly(X, pw):
    #do log
    df_t = pd.DataFrame(X[:,:10])
    X_t = df_t.replace(0, 1/math.e)
    log_res = np.log(X_t)
    
    #do poly
    vector = X[:,10:]
    res    = X[:,:10]
    X      = X[:,:10]
    for power in range(2,pw + 1):
        res = np.concatenate((res, np.power(X, power)), axis=1)
    res_poly_log = np.concatenate((res, log_res), axis=1)
    
    #return conat results
    return np.concatenate((res_poly_log, vector), axis=1)

def classify(X, classifier):
            x_pre = pipe_dict['precomp_transform']
            X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(x_pre, y, test_size=0.25)
            gardient_class = GradientBoostingClassifier(n_estimators = 1000,
                      learning_rate = 0.01,
                      max_depth =  10,
                      min_samples_leaf = 5,
                      max_features = 0.01,
                      max_leaf_nodes =  None)
            gardient_class.fit(X_cl_train, y_cl_train)

            x_pre = pd.concat([pd.DataFrame(gardient_class.predict(x_pre)), pd.DataFrame(x_pre)], axis=1)
            y_gross = dta_clean['worldwide_gross']
            x_fin, _X_dummy, y_fin, _y_dummy = train_test_split(x_pre, y_gross, test_size=0)

DummyTransformer = FunctionTransformer(dummy)
LogarithmicTransformer = FunctionTransformer(log)
PolynomialTransformer = FunctionTransformer(poly)
LogPolynomialTransformer = FunctionTransformer(log_poly)
ClassifierlTransformer = FunctionTransformer(classify)


###define a global itteration var###########
itter_start   = 0
itter_current = 0
###define new config###########

################################
### Default Data Preprocessor ###
#################################
#preprocessors = [LogPolynomialTransformer, LogarithmicTransformer, PolynomialTransformer]
preprocessors = [LogPolynomialTransformer]
#preprocessors = [DummyTransformer]
preprocessors_cfg = dict()
preprocessors_cfg[DummyTransformer.func.__name__] = {}
preprocessors_cfg[LogarithmicTransformer.func.__name__] = {}
preprocessors_cfg[PolynomialTransformer.func.__name__] = dict(
    preprocessor__kw_args=[]
)
preprocessors_cfg[LogPolynomialTransformer.func.__name__] = dict(
    preprocessor__kw_args=[]
)
################################
#### Default Data Transformer ##
################################
#transfomers = [DummyTransformer, StandardScaler()]
transfomers = [StandardScaler()]
#transfomers = [DummyTransformer]
transfomers_cfg = dict()
transfomers_cfg[DummyTransformer.func.__name__] = {}
transfomers_cfg[Normalizer.__name__] = dict(
    transfomer__norm=['l1', 'l2', 'max']
)
transfomers_cfg[StandardScaler.__name__] = {}

####################################
### Default Dim Reducer, Feat Sel. #
####################################
#reducers = [DummyTransformer, PCA(), GenericUnivariateSelect(), RFE(ExtraTreesRegressor())]
reducers= [RFE(ExtraTreesRegressor())]
#reducers = [DummyTransformer]
reducers_cfg = dict()
reducers_cfg[DummyTransformer.func.__name__] = {}
reducers_cfg[PCA.__name__] = dict(
    reducer__n_components=[],
    reducer__svd_solver=['auto']
)
reducers_cfg[GenericUnivariateSelect.__name__] = dict(
    reducer__score_func=[f_regression],
    reducer__mode=['k_best'],
    reducer__param=[]
)
reducers_cfg[RFE.__name__] = dict(
    reducer__n_features_to_select=[],
    reducer__step=[0.1]
)

#########################
####### Models ##########
#########################
#models = [LinearSVC(),MLPClassifier(),GradientBoostingClassifier(),RandomForestClassifier(),LogisticRegression()]
#models = [AdaBoostClassifier(),BaggingClassifier(),ExtraTreesClassifier(),GradientBoostingClassifier(),RandomForestClassifier(),PassiveAggressiveClassifier(),LogisticRegression(),RidgeClassifier(),SGDClassifier(),GaussianNB(),MultinomialNB(),KNeighborsClassifier(),RadiusNeighborsClassifier(),NearestCentroid(),MLPClassifier(),SVC(),LinearSVC(),NuSVC(),DecisionTreeClassifier(),ExtraTreeClassifier()]
#models_reg = [AdaBoostRegressor(),BaggingRegressor(),ExtraTreesRegressor(),GradientBoostingRegressor(),RandomForestRegressor(),ElasticNet(),HuberRegressor(),Lasso(),LassoLars(),LinearRegression(),PassiveAggressiveRegressor(),Ridge(),SGDRegressor(),OrthogonalMatchingPursuit(),RANSACRegressor(),KNeighborsRegressor(),RadiusNeighborsRegressor(),MLPRegressor(),SVR(),LinearSVR(),NuSVR(),DecisionTreeRegressor(),ExtraTreeRegressor()]
models_reg = [BaggingRegressor(),ExtraTreesRegressor(),GradientBoostingRegressor()]
    
models_class = [GradientBoostingClassifier()]
models_class_cfg = {}
models_cfg = {}

#full params - dont work
'''
models_cfg[BaggingClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__bootstrap = [True, False],
    model__bootstrap_features = [True, False],
)
models_cfg[ExtraTreesClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__criterion = ['gini', 'entropy'],
    model__max_features  = ["auto", None, "sqrt"],
    model__bootstrap = [True, False],
    model__oob_scorS
    model__min_samples_split = [2, 0.1, 0.5],
    model__min_samples_leaf = [1,  0.1, 0.5],
    model__min_weight_fraction_leaf = [0.0 ,0.1, 0.5],
    model__max_leaf_nodes =  [None, 10, 50],
    model__min_impurity_split = [1e-7, 1e-6, 1e-5]
)
models_cfg[GradientBoostingClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__max_features  = ["auto", None, "sqrt"],
    model__max_depth = [3, 5, 10],
    model__max_leaf_nodes =  [None, 10, 50],
    model__min_impurity_split = [1e-7, 1e-6, 1e-5],
    model__subsample = [0.1, 0.5, 1.0],
    model__min_samples_split = [2, 0.1, 0.5],
    model__min_samples_leaf = [1,  0.1, 0.5],
    model__min_weight_fraction_leaf = [0.0 ,0.1, 0.5],
)
models_cfg[RandomForestClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__criterion = ['gini', 'entropy'],
    model__max_features  = ["auto", None, "sqrt"],
    model__max_depth = [None ,3, 5, 10],
    model__min_samples_split = [2, 0.1, 0.5],
    model__min_samples_leaf = [1,  0.1, 0.5],
    model__min_weight_fraction_leaf = [0.0 ,0.1, 0.5],
    model__max_leaf_nodes =  [None, 10, 50],
    model__bootstrap = [True, False],
    model__oob_score = [True, False]
)
models_cfg[LogisticRegression.__name__] = dict(
    model__solver =  ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
    model__class_weight = [None, "balanced"],
    model__C = np.logspace(-4, 4, 3),
    model__tol = [1e-4, 1e-3, 1e-2],
    model__multi_class =  ['ovr', 'multinomial'],
    model__max_iter = [50, 100, 300],
    model__penalty =['l1', 'l2']
)
'''

#feasible params for running 5 models with pipeline 
'''
models_cfg[MLPClassifier.__name__] = dict(
    model__hidden_layer_sizes = [100],
    model__activation = ['identity', 'logistic', 'tanh', 'relu'],
    model__solver = ['lbfgs', 'sgd', 'adam'],
    model__max_iter = [400],
    model__learning_rate_init = [ 0.8, 0.01,  0.1]

)
models_cfg[LinearSVC.__name__] = dict(
    model__C = np.logspace(-4, 4, 3),
    model__loss = ['hinge', 'squared_hinge']
)
models_cfg[GradientBoostingClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__max_features  = ["auto", None, "sqrt"],
    model__max_depth = [3, 5, 10],
    model__max_leaf_nodes =  [None, 10, 50]
)
models_cfg[RandomForestClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__criterion = ['gini', 'entropy'],
    model__max_features  = ["auto", None, "sqrt"],
    model__max_depth = [None ,3, 5, 10]
)
models_cfg[LogisticRegression.__name__] = dict(
    model__solver =  ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
    model__C = np.logspace(-4, 4, 3),
    model__max_iter = [50, 100, 300]
)
'''

##gradient tuning
models_class_cfg[GradientBoostingClassifier.__name__] = dict(
              model__n_estimators = [1000],
              model__learning_rate = [0.01],
              model__max_depth =  [10],
              model__min_samples_leaf = [5],
              model__max_features = [0.01],
              model__max_leaf_nodes =  [None]
)


#params for running with no pipeline
'''
models_cfg[AdaBoostClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__learning_rate = [0.1, 0.5,1.0],
    model__algorithm = ['SAMME', 'SAMME.R']
    )
models_cfg[BaggingClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__bootstrap = [True, False],
    model__bootstrap_features = [True, False]
)
models_cfg[ExtraTreesClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__criterion = ['gini', 'entropy'],
    model__max_features  = ["auto", None, "sqrt"]
)
models_cfg[GradientBoostingClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__max_features  = ["auto", None, "sqrt"],
    model__max_depth = [3, 5, 10]
)
models_cfg[RandomForestClassifier.__name__] = dict(
    model__n_estimators = [10, 50, 100, 130],
    model__criterion = ['gini', 'entropy'],
    model__max_features  = ["auto", None, "sqrt"]
)
models_cfg[PassiveAggressiveClassifier.__name__] = dict(
    model__fit_intercept = [True],
    model__n_iter = [5, 10 , 20],
    model__shuffle = [True, False],
    model__loss = ['hinge', 'squared_hinge']
)
models_cfg[LogisticRegression.__name__] = dict(
    model__solver =  ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
    model__C = np.logspace(-4, 4, 3),
    model__max_iter = [50, 100, 300]
)
models_cfg[RidgeClassifier.__name__] = dict(
    model__tol = [1e-4, 1e-3, 1e-2],
    model__solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
    model__fit_intercept = [True]
)
models_cfg[SGDClassifier.__name__] = dict(
    model__loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    model__l1_ratio =[0.15, 0.4 , 0.8]
)
models_cfg[GaussianNB.__name__] = {}
models_cfg[MultinomialNB.__name__] = dict(
    model__alpha = np.reciprocal(np.logspace(-4, 4, 3))
)
models_cfg[KNeighborsClassifier.__name__] = dict(
    model__n_neighbors = [5, 10 , 20],
    model__algorithm = ['ball_tree', 'kd_tree', 'auto'],
    model__leaf_size = [15, 30, 50]
)
models_cfg[RadiusNeighborsClassifier.__name__] = dict(
    model__radius = [0.1, 0.5, 1.0],
    model__algorithm = ['ball_tree', 'kd_tree',  'auto'],
    model__p = [1, 2, 3],
    model__outlier_label = [6]
)
models_cfg[NearestCentroid.__name__] = dict(
    model__shrink_threshold = [None, 0.1, 0.4, 0.8]
)
models_cfg[MLPClassifier.__name__] = dict(
    model__hidden_layer_sizes = [100],
    model__activation = ['identity', 'logistic', 'tanh', 'relu'],
    model__solver = ['lbfgs', 'sgd', 'adam'],
    model__max_iter = [500]
    #model__learning_rate_init = [ 0.8, 0.01,  0.1]

)
models_cfg[SVC.__name__] = dict(
    model__kernel = ['rbf', 'sigmoid'],
    model__degree = [2,3,5],
    model__coef0 = [0.0, 0.5, 1.0]
)
models_cfg[LinearSVC.__name__] = dict(
    model__C = np.logspace(-4, 4, 3),
    model__loss = ['hinge', 'squared_hinge']
)
models_cfg[NuSVC.__name__] = dict(
    model__nu = [0.1, 0.2],
    model__kernel = ['rbf', 'sigmoid'],
    model__degree = [2,3,5]
 )
models_cfg[DecisionTreeClassifier.__name__] = dict(
    model__criterion = ['gini', 'entropy'],
    model__max_features  = ["auto", None, "sqrt"],
    model__min_samples_split = [2, 0.1, 0.5]
)
models_cfg[ExtraTreeClassifier.__name__] =  dict(
    model__criterion = ['gini', 'entropy'],
    model__max_features  = ["auto", None, "sqrt"],
    model__min_samples_split = [2, 0.1, 0.5]
)
'''

'''
models_cfg[AdaBoostClassifier.__name__] = {}
models_cfg[BaggingClassifier.__name__] = {}
models_cfg[ExtraTreesClassifier.__name__] = {}
models_cfg[GradientBoostingClassifier.__name__] = {}
models_cfg[RandomForestClassifier.__name__] = {}
models_cfg[PassiveAggressiveClassifier.__name__] = {}
models_cfg[LogisticRegression.__name__] = {}
models_cfg[RidgeClassifier.__name__] = {}
models_cfg[SGDClassifier.__name__] = {}
models_cfg[GaussianNB.__name__] = {}
models_cfg[MultinomialNB.__name__] = {}
models_cfg[KNeighborsClassifier.__name__] = {}
models_cfg[RadiusNeighborsClassifier.__name__] = {}
models_cfg[NearestCentroid.__name__] = {}
models_cfg[MLPClassifier.__name__] = {}
models_cfg[SVC.__name__] = {}
models_cfg[LinearSVC.__name__] = {}
models_cfg[NuSVC.__name__] = {}
models_cfg[DecisionTreeClassifier.__name__] = {}
models_cfg[ExtraTreeClassifier.__name__] = {}
'''

models_cfg[AdaBoostRegressor.__name__] = {}
models_cfg[BaggingRegressor.__name__] = {}
models_cfg[ExtraTreesRegressor.__name__] = {}
models_cfg[GradientBoostingRegressor.__name__] = {}
models_cfg[RandomForestRegressor.__name__] = {}
models_cfg[BayesianRidge.__name__] = {}
models_cfg[ElasticNet.__name__] = {}
models_cfg[HuberRegressor.__name__] = {}
models_cfg[Lars.__name__] = {}
models_cfg[Lasso.__name__] = {}
models_cfg[LassoLars.__name__] = {}
models_cfg[LinearRegression.__name__] = {}
models_cfg[PassiveAggressiveRegressor.__name__] = {}
models_cfg[Ridge.__name__] = {}
models_cfg[SGDRegressor.__name__] = {}
models_cfg[OrthogonalMatchingPursuit.__name__] = {}
models_cfg[RANSACRegressor.__name__] = {}
models_cfg[TheilSenRegressor.__name__] = {}
models_cfg[KNeighborsRegressor.__name__] = {}
models_cfg[RadiusNeighborsRegressor.__name__] = {}
models_cfg[MLPRegressor.__name__] = {}
models_cfg[SVR.__name__] = {}
models_cfg[LinearSVR.__name__] = {}
models_cfg[NuSVR.__name__] = {}
models_cfg[DecisionTreeRegressor.__name__] = {}
models_cfg[ExtraTreeRegressor.__name__] = {}


def launch_pipe_instance(x,y, pipe, cfg_dict, pipeline_cfg, precomp_pipe, errors, errors_ind, ind):
    print ("Starting precomp pipline for "+ str(cfg_dict))
    #run the pipe, except eceptions, save errors
    try:
            #precomp_pipe.put_nowait({"pipeline_cfg": pipeline_cfg, "cfg_dict": cfg_dict,"precomp_transform": pipe.set_params(**cfg_dict).fit_transform(x,y)})
            dump_dict = {"pipeline_cfg": pipeline_cfg, "cfg_dict": cfg_dict,"precomp_transform": pipe.set_params(**cfg_dict).fit_transform(x,y)}
            tmp_trg = "./tmp/" + str(itter_current) + "_" + str(ind)
            with open(tmp_trg, 'wb') as handle:
                  pickle.dump(dump_dict, handle)
            print ("Finished precomp pipline for "+ str(cfg_dict))
            

    except (ValueError, MemoryError) as err:
            print ("GREP_ME***Error caught for  precomp pipeline: ["+ pipeline_cfg+"] ")
            errors_ind.append({"cfg": pipeline_cfg})
            print(err)
            pass

def get_pipe_result(x, y, preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind):
    global itter_current
    itter_current += 1
    #create pipline for the preprocessing
    preprocessor_name = type(preprocessor).__name__ if (type(preprocessor).__name__ != "FunctionTransformer") else preprocessor.func.__name__
    transfomer_name =  type(transfomer).__name__ if (type( transfomer).__name__ != "FunctionTransformer") else  transfomer.func.__name__
    reducer_name = type(reducer).__name__ if (type(reducer).__name__ != "FunctionTransformer") else reducer.func.__name__
    
    print('####################################################################################')
    print('################# Runing the itteration %d  of pipeline precomp      ###############' %(itter_current))
    print('####################################################################################')
    
    #Define and save pipe cfg
    pipeline_cfg = "| preprocessor:" + preprocessor_name +  " | transfomer: " + transfomer_name + " | reducer: " + reducer_name
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('transfomer', transfomer), ('reducer', reducer)])
    print(pipeline_cfg)
    
    #itterate over each cfg variation and precompute the result
    param_grid = dict(dict(preprocessors_cfg[preprocessor_name], **dict(transfomers_cfg[transfomer_name], **reducers_cfg[reducer_name])))
    print(param_grid)
    processes = []
    local_ind = 0
    for _terms  in list(product(*[preprocessors_cfg[preprocessor_name][it] for it in preprocessors_cfg[preprocessor_name]])):
        cfg_dict = dict((term, _terms[ind]) for ind, term in enumerate(tuple(it for it in preprocessors_cfg[preprocessor_name])))
    
        for _terms  in list(product(*[transfomers_cfg[transfomer_name][it] for it in transfomers_cfg[transfomer_name]])):
            cfg_dict.update(dict((term, _terms[ind]) for ind, term in enumerate(tuple(it for it in transfomers_cfg[transfomer_name]))))
                
            for _terms  in list(product(*[reducers_cfg[reducer_name][it] for it in reducers_cfg[reducer_name]])):
                cfg_dict.update(dict((term, _terms[ind]) for ind, term in enumerate(tuple(it for it in reducers_cfg[reducer_name]))))
                #launch in a parraler manner a pipe dict
                #launch_pipe_instance(x,y, pipe, cfg_dict, pipeline_cfg, precomp_pipe, errors, errors_ind, local_ind)
                local_ind += 1
                final_cfg = cfg_dict
                p = Process(target=launch_pipe_instance, args=(x,y, pipe, cfg_dict, pipeline_cfg, precomp_pipe, errors, errors_ind, local_ind))
                p.start()
                processes.append(p)

    for p in processes: p.join()
                

def run_grid_search(x,y, model_class, model_reg, cfg_dict, pipeline_cfg, results, errors, errors_ind, label_fn):
    global itter_current
    itter_current += 1
    #check if itteration start is set to something different than 0 and then check if current itteration has been reached
    if itter_start != 0 and itter_current < itter_start: return
    #create pipline and use GridSearch to find the best params for given pipeline
    name = type(model_reg).__name__
    name_class = type(model_reg).__name__
    
    #Define and save pipe cfg
    #pipe = Pipeline(steps=[('Classifier', ClassifierTransformer(label_fn)) ,('model', model_reg)])
    pipe = Pipeline(steps=[('model', model_reg)])

    #create a dict with param grid
    param_grid = models_cfg[name]
    #create estimator
    cv = 5
    print('####################################################################################')
    print('################# Runing the itteration %d  of the GridSearchCV ####################' %(itter_current))
    print('####################################################################################')
    print ("***Starting ["  + name + "] estimator run, pipeline: "+ pipeline_cfg+" ")
    print("##param_grid##")
    print(param_grid)
    estimator = GridSearchCV(pipe,param_grid,verbose=2, cv=cv, n_jobs=-1)
    #run the esmimator, except eceptions, sape errors
    try:
            estimator.fit(x, y)
            print ("GREP_ME***Results of ["  + name + "] estimatorrun are")
            print (estimator.cv_results_)            
            print ("GREP_ME***Best params of ["  + name + "] estimator,pipeline:"+ pipeline_cfg+"  run are")
            best_param = dict(estimator.best_params_, **cfg_dict)
            print (best_param)
            print ("GREP_ME***Best score of ["  + name + "] estimator, pipeline:"+ pipeline_cfg+" run are")
            print (estimator.best_score_)
            if (name not in results) or (estimator.best_score_ > results[name]["score"]):
                results[name] = {"score": estimator.best_score_, "pipe":pipeline_cfg, "best_cfg": best_param}
    except (ValueError, MemoryError) as err:
            print ("GREP_ME***Error caught for  ["  + name + "] , pipeline: ["+ pipeline_cfg+"] ")
            print(err)
            pass

def run_solver(x,y,preprocessors, transfomers, reducers, models_class, models_reg, results, errors, errors_ind, precomp_pipe, label_fn):
    # mix it, so that the sample order is randomized
    #x, _X_dummy, y, _y_dummy = train_test_split(x, y, test_size=0)
    n_samples, n_features = x.shape   
       
    #make a dir for preprocessor temp files
    try:
        os.mkdir("./tmp")
    except FileExistsError:
        #rm temp dir and make new one
        shutil.rmtree("./tmp")
        os.mkdir("./tmp")
    
    
    #precompute the preprocessing results so that can be resued by grisearch and dont need to be precomputed.    
    for preprocessor, transfomer, reducer in product(preprocessors, transfomers, reducers):
       ##run gridesearch with new amout of features, depending of preprocessor and hence pass the right amount of maximum components to the reducers 
        if preprocessor.func.__name__ == LogarithmicTransformer.func.__name__ :
            n_components = get_components_list(n_features, [{"pw":2}, {"pw":1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            get_pipe_result(x, y,preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind)
        elif preprocessor.func.__name__ == PolynomialTransformer.func.__name__ or preprocessor.func.__name__ == LogPolynomialTransformer.func.__name__:
            LogPol = False
            if preprocessor.func.__name__ == LogPolynomialTransformer.func.__name__: LogPol = True
            kw_arg_powers = get_powers_list(n_samples, n_features, 3)
            pw_lst = []
            for pw in kw_arg_powers:
                pw_lst = pw_lst + [pw]
                preprocessors_cfg[preprocessor.func.__name__]["preprocessor__kw_args"] = [pw]
                n_components = get_components_list(n_features, pw_lst, LogPol)
                print()
                print("LogPol " + str(LogPol))
                print("n_components")
                print(n_components)
                print("pw_lst")
                print(pw_lst)
                reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
                reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
                reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
                get_pipe_result(x, y,preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind)
        else:
            n_components = get_components_list(n_features, [{"pw":1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            get_pipe_result(x, y,preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind)
      
   
    #for each physically saved pickle run grid search for each model
    for filename in os.listdir("./tmp"):
        pipe_dict = pickle.loads(open("./tmp/" + filename, 'rb').read())
        for model_class in models_class: 
            #clasifiy train and predict data with 75% - can be misleading, but ok for finding quick estimate of it percorms
            print("Preclassifying results")
            x_pre = pipe_dict['precomp_transform']
            X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(x_pre, y, test_size=0.20)
            gardient_class = GradientBoostingClassifier(n_estimators = 1000,
                      learning_rate = 0.01,
                      max_depth =  10,
                      min_samples_leaf = 5,
                      max_features = 0.01,
                      max_leaf_nodes =  None)
            gardient_class.fit(X_cl_train, y_cl_train)
            print("Classify score is :" + str(gardient_class.score(x_pre,y)))
            x_pre = pd.concat([pd.DataFrame(gardient_class.predict(x_pre)), pd.DataFrame(x_pre)], axis=1)
            y_gross = dta_clean['worldwide_gross']
            x_fin, _X_dummy, y_fin, _y_dummy = train_test_split(x_pre, y_gross, test_size=0)  
            for model_reg in models_reg: 
                 run_grid_search(x_fin, y_fin, model_class, model_reg,  pipe_dict['cfg_dict'],  pipe_dict['pipeline_cfg'], results, errors, errors_ind, label_fn)

                #run_grid_search(pipe_dict['precomp_transform'], dta_clean['worldwide_gross'], model_class, model_reg,  pipe_dict['cfg_dict'],  pipe_dict['pipeline_cfg'], results, errors, errors_ind, label_fn)

##run calssifiers for two 4 cases - 2 classes, 3 clasees, 4 classes, 5 clasess

def label_gross_2 (gross):
    if (gross < 200000000) : return 1
    elif (gross >= 200000000) : return 2

def label_gross_3 (gross):
    if (gross < 10000000) : return 1
    elif ((gross >= 10000000) & (gross < 300000000)) : return 2
    elif (gross >= 300000000) : return 3

def label_gross_4 (gross):
    if (gross < 5000000) : return 1
    elif ((gross >= 5000000) & (gross < 50000000)) : return 2
    elif ((gross >= 50000000) & (gross < 350000000)) : return 3
    elif (gross >= 350000000) : return 4

def label_gross_5 (gross):
    if (gross < 1000000) : return 1
    elif ((gross >= 1000000) & (gross < 25000000)) : return 2
    elif ((gross >= 25000000) & (gross < 100000000)) : return 3
    elif ((gross >= 100000000) & (gross < 400000000)) : return 4
    elif (gross >= 400000000) : return 5
    
def label_gross_6 (gross):
    if (gross < 1000000) : return 1
    elif ((gross >= 1000000) & (gross < 25000000)) : return 2
    elif ((gross >= 25000000) & (gross < 50000000)) : return 3
    elif ((gross >= 50000000) & (gross < 150000000)) : return 4
    elif ((gross >= 150000000) & (gross < 450000000)) : return 5
    elif (gross >= 450000000) : return 6

def label_gross_7 (gross):
    if (gross < 500000) : return 1
    elif ((gross >= 500000) & (gross < 5000000)) : return 2
    elif ((gross >= 5000000) & (gross < 50000000)) : return 3
    elif ((gross >= 50000000) & (gross < 150000000)) : return 4
    elif ((gross >= 150000000) & (gross < 200000000)) : return 5
    elif ((gross >= 200000000) & (gross < 500000000)) : return 6
    elif (gross >= 500000000) : return 7
    
def label_gross_8 (gross):
    if (gross < 500000) : return 1
    elif ((gross >= 500000) & (gross < 5000000)) : return 2
    elif ((gross >= 5000000) & (gross < 20000000)) : return 3
    elif ((gross >= 20000000) & (gross < 50000000)) : return 4
    elif ((gross >= 50000000) & (gross < 100000000)) : return 5
    elif ((gross >= 100000000) & (gross < 250000000)) : return 6
    elif ((gross >= 250000000) & (gross < 550000000)) : return 7
    elif (gross >= 550000000) : return 8

def run_for_many(cl_n,label_fn):
    results = {}
    precomp_pipe = []
    errors = []
    errors_ind = []
    X = dta_clean.drop('worldwide_gross', axis=1)
    y = dta_clean.worldwide_gross.apply (lambda gross: label_fn (gross))
    #y = dta_clean['worldwide_gross']
    print ("#########################################")
    print ("###Starting all estimators for cl: "+ str(cl_n))
    print ("#########################################")
    run_solver(X,y, preprocessors, transfomers, reducers, models_class, models_reg, results, errors, errors_ind, precomp_pipe, label_fn)
    print ("#########################################")
    print ("###Finished all estimators for cl: "+ str(cl_n))
    print ("#########################################")

    print ("#########################################")
    print ("#######Printing results for cl: "+ str(cl_n))
    print ("#########################################")
    print(results)
    print("priting simply sorted numbers, grep them to find the best cfg or cl: "+ str(cl_n))
    scores = [results[model]["score"] for model in results]
    print(sorted(scores))


#ignore warnigs

desc = "no_imdb_gradient_boost_class_with_regression"


labels = [label_gross_8, label_gross_7, label_gross_6, label_gross_5, label_gross_4, label_gross_3, label_gross_2]
#labels = [label_gross_3]
#save orig datetime and save orign stdout
orig_stdout = sys.stdout
time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
for ind, cb in enumerate(labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #restart the current itterator for each run
        global itter_current
        itter_current = 0
        trg = "classifyRes_" + time + "_" + desc + "_" + cb.__name__ + ".log"
        new_file = open(trg,"w")
        sys.stdout = new_file
        #set the itterator run to start from
        global itter_start
        itter_start = 0        
        run_for_many(cb.__name__,cb)
        #return stdout for some reason
sys.stdout = orig_stdout


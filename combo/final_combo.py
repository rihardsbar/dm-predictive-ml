import os
import sys
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.feature_selection import GenericUnivariateSelect, RFE
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression, mutual_info_regression
from itertools import product
from multiprocessing import Process, Value, Array
from threading import Thread
import pickle
import shutil
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from joblib import Parallel, delayed

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

path_full = "../dataset/no_imdb_names-count_cat-tf_184f.csv"
path_train = "../dataset/no_imdb_names-count_cat-tf_184f_train.csv"
path_test = "../dataset/no_imdb_names-count_cat-tf_184f_test.csv"

dta_full = pd.read_csv(path_full)
dta_full = dta_full.fillna(value=0, axis=1)
dta_full = dta_full.dropna()
dta_full = dta_full.drop('Unnamed: 0', axis=1)

dta_train = pd.read_csv(path_train)
dta_train = dta_train.fillna(value=0, axis=1)
dta_train = dta_train.dropna()
dta_train = dta_train.drop('Unnamed: 0', axis=1)

dta_test = pd.read_csv(path_test)
dta_test = dta_test.fillna(value=0, axis=1)
dta_test = dta_test.dropna()
dta_test = dta_test.drop('Unnamed: 0', axis=1)

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

def label_gross_9 (gross):
    if (gross < 500000) : return 1
    elif ((gross >= 500000) & (gross < 5000000)) : return 2
    elif ((gross >= 5000000) & (gross < 20000000)) : return 3
    elif ((gross >= 20000000) & (gross < 50000000)) : return 4
    elif ((gross >= 50000000) & (gross < 70000000)) : return 5
    elif ((gross >= 70000000) & (gross < 125000000)) : return 6
    elif ((gross >= 125000000) & (gross < 250000000)) : return 7
    elif ((gross >= 250000000) & (gross < 550000000)) : return 8
    elif (gross >= 550000000) : return 9
    
def label_gross_10 (gross):
    if    (gross  < 500000) : return 1
    elif ((gross >= 500000)    & (gross < 5000000)) : return 2
    elif ((gross >= 5000000)   & (gross < 20000000)) : return 3
    elif ((gross >= 20000000)  & (gross < 50000000)) : return 4
    elif ((gross >= 50000000)  & (gross < 70000000)) : return 5
    elif ((gross >= 70000000)  & (gross < 125000000)) : return 6
    elif ((gross >= 125000000) & (gross < 250000000)) : return 7
    elif ((gross >= 250000000) & (gross < 400000000)) : return 8
    elif ((gross >= 400000000) & (gross < 600000000)) : return 9
    elif  (gross >= 600000000) : return 10

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

DummyTransformer = FunctionTransformer(dummy)
LogarithmicTransformer = FunctionTransformer(log)
PolynomialTransformer = FunctionTransformer(poly)
LogPolynomialTransformer = FunctionTransformer(log_poly)



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
#models_reg = [BaggingRegressor(),ExtraTreesRegressor(),GradientBoostingRegressor()]

models_class     = [GradientBoostingClassifier()]
models_reg       = [GradientBoostingRegressor()]
models_class_cfg = {}
models_reg_cfg = {}


##gradient tuning

models_class_cfg[GradientBoostingClassifier.__name__] = dict(
              model__n_estimators = [200, 500, 1000],
              model__learning_rate = [0.01, 0.1],
              model__max_depth =  [5, 10],
              model__min_samples_leaf = [5, 50],
              model__max_features = [0.01, 0.1],
              model__max_leaf_nodes =  [None, 5]
)

models_reg_cfg[GradientBoostingRegressor.__name__] = dict(
              model__n_estimators = [200, 500, 1000],
              model__learning_rate = [0.01, 0.1],
              model__max_depth =  [5, 10],
              model__min_samples_leaf = [5, 50],
              model__max_features = [0.01, 0.1],
              model__max_leaf_nodes =  [None, 5]
)
'''

models_class_cfg[GradientBoostingClassifier.__name__] = dict(
              model__n_estimators = [200, 1000],
              model__max_leaf_nodes =  [5]
)

models_reg_cfg[GradientBoostingRegressor.__name__] = dict(
              model__learning_rate = [0.01],
              model__max_depth =  [5, 10]
)
'''

def launch_pipe_instance(x,y, pipe, cfg_dict, pipeline_cfg, errors_ind, model_dir, local_ind):
    print ("Starting precomp pipline for "+ str(cfg_dict) + " for " + model_dir)
    #run the pipe, except eceptions, save errors
    try:
            #precomp_pipe.put_nowait({"pipeline_cfg": pipeline_cfg, "cfg_dict": cfg_dict,"precomp_transform": pipe.set_params(**cfg_dict).fit_transform(x,y)})
            x_train, x_test = x
            y_train, y_test = y
            pipe.set_params(**cfg_dict).fit(x_train,y_train)            
            dump_dict = {"pipeline_cfg": pipeline_cfg, "cfg_dict": cfg_dict, "x_train": pipe.transform(x_train), "y_train":y_train, "x_test": pipe.transform(x_test), "y_test": y_test}
            tmp_trg = "./" + model_dir + "/" + str(itter_current) + "_" + str(ind)
            with open(tmp_trg, 'wb') as handle:
                  pickle.dump(dump_dict, handle)
            print ("Finished precomp pipline for "+ str(cfg_dict))
            

    except (ValueError, MemoryError) as err:
            print ("GREP_ME***Error caught for  precomp pipeline: ["+ pipeline_cfg+"] ")
            errors_ind.append({"cfg": pipeline_cfg})
            print(err)
            pass

def get_pipe_result(x, y,preprocessor, transfomer, reducer, errors_ind, model_dir):
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
                p = Process(target=launch_pipe_instance, args=(x,y, pipe, cfg_dict, pipeline_cfg, errors_ind, model_dir, local_ind))
                p.start()
                processes.append(p)

    for p in processes: p.join()

def run_class_precomp_instanace(x_train_cl, y_train_cl, x_test_cl, y_test_cl, model_class, mod_dict_class, cfg_dict_class, pipeline_cfg_class, ind):
    name_class = type(model_class).__name__
    print("")
    print ("Starting precomp class for "+ str(name_class))
        
    try:
        print ("Precomp class ["  + name_class + "] estimator,pipeline:"+ pipeline_cfg_class+"  params:")
        print(cfg_dict_class)
        print ("Precomp class ["  + name_class + "] estimator, model params:")
        print(mod_dict_class)
        pipe_class = Pipeline(steps=[('model', model_class)])
        pipe_class.set_params(**mod_dict_class).fit(x_train_cl, y_train_cl)
        train_score = pipe_class.score(x_train_cl,y_train_cl)
        test_score = pipe_class.score(x_test_cl,y_test_cl)
        print("Classify train score is: ")
        print(train_score)
        print("Classify test  score is: ")
        print(test_score)
        dump_dict = {"name_class":name_class, "mod_dict_class": mod_dict_class,"pipeline_cfg_class": pipeline_cfg_class, "cfg_dict_class": cfg_dict_class, "x_train_cl_res": pipe_class.predict(x_train_cl), "x_test_cl_res": pipe_class.predict(x_test_cl), "class_train_score": train_score, "class_test_score": test_score}
        tmp_trg = "./tmp_class_res/" + str(itter_current) + "_" + str(ind)
        with open(tmp_trg, 'wb') as handle:
            pickle.dump(dump_dict, handle)
            print ("Finished precomp class for "+ str(name_class))
            
    except (ValueError, MemoryError) as err:
            print ("GREP_ME***Error caught for  precomp class: ("+ name_class+") ")
            print(err)
            pass
    sys.stdout.flush()
    
def pre_class_precomp(x_train_cl, y_train_cl, x_test_cl, y_test_cl, model_class, cfg_dict_class, pipeline_cfg_class):       
    global itter_current
    itter_current += 1
    #create pipline for the preprocessing
    name_class = type(model_class).__name__
    
    print('####################################################################################')
    print('################# Runing the itteration %d  of class precomp     ###############' %(itter_current))
    print('####################################################################################')

    #pool = Pool(processes=9,maxtasksperchild=1)
    #itterate over each cfg variation and compute the result
    param_grid = dict(models_class_cfg[name_class])
    print(param_grid)
    processes_args = []
    local_ind = 0
    for _terms  in list(product(*[models_class_cfg[name_class][it] for it in models_class_cfg[name_class]])):
        local_ind = local_ind + 1
        mod_dict_class = dict((term, _terms[ind]) for ind, term in enumerate(tuple(it for it in models_class_cfg[name_class])))     
        processes_args.append((x_train_cl, y_train_cl, x_test_cl, y_test_cl, model_class, mod_dict_class, cfg_dict_class, pipeline_cfg_class,local_ind))
        
    Parallel(n_jobs=-1, verbose=1)(delayed(run_class_precomp_instanace)(*p_args) for p_args in processes_args)
    
    
def run_class_precomp(x_train_cl, y_train_cl, x_test_cl, y_test_cl, models_class, cfg_dict_class, pipeline_cfg_class):
        for model_class in models_class:
            pre_class_precomp(x_train_cl, y_train_cl, x_test_cl, y_test_cl, model_class, cfg_dict_class, pipeline_cfg_class)
            

def run_model_search_instance(x_train_reg, y_train_reg, x_test_reg, y_test_reg, model_reg, mod_dict_reg, cfg_dict_reg, pipeline_cfg_reg, res_dict_class, results):
    name_class = res_dict_class['name_class']
    name_reg = type(model_reg).__name__
    print("")
    print("Running model search")
        
    try:

        print("GREP_ME***printing model search instance results")

        print ("Classify ["  + name_class + "] estimator,pipeline:"+ res_dict_class['pipeline_cfg_class']+"  params:")
        print(res_dict_class['cfg_dict_class'])
        print ("Classify ["  + name_class + "] estimator, model params:")
        print(res_dict_class['mod_dict_class'])
        print("Classify train score is: ")
        print(res_dict_class['class_train_score'])
        print("Classify test  score is: ")
        print(res_dict_class['class_test_score'])

        #populate x_train and x_test with class data
        x_train_reg_cl = pd.concat([pd.DataFrame(res_dict_class['x_train_cl_res']), pd.DataFrame(x_train_reg)], axis=1)
        x_test_reg_cl  = pd.concat([pd.DataFrame(res_dict_class['x_test_cl_res']), pd.DataFrame(x_test_reg)], axis=1)
        
   
        #train the regressor no classes
        print ("Regressor no cl ["  + name_reg + "] estimator,pipeline:"+ pipeline_cfg_reg+"  params:")
        print(cfg_dict_reg)
        print ("Regressor no cl ["  + name_reg + "] estimator, model params:")
        print(mod_dict_reg)
        '''
        pipe_reg = Pipeline(steps=[('model', model_reg)])
        pipe_reg.set_params(**mod_dict_reg).fit(x_train_reg, y_train_reg)
        pipe_reg_cv = Pipeline(steps=[('model', model_reg)]).set_params(**mod_dict_reg)
        print("Regressor no cl train score is: ")
        print(pipe_reg.score(x_train_reg,y_train_reg))
        print("Regressor no cl valid score is: ")
        print(cross_val_score(pipe_reg_cv, x_train_reg,y_train_reg).mean())
        print("Regressor no cl test  score is: ")
        print(pipe_reg.score(x_test_reg,y_test_reg))
        '''

        #train the regressor with classes
        print ("Regressor with cl ["  + name_reg + "] estimator,pipeline given before")
        pipe_reg_cl = Pipeline(steps=[('model', model_reg)])
        pipe_reg_cl.set_params(**mod_dict_reg).fit(x_train_reg_cl, y_train_reg)
        pipe_reg_cl_cv = Pipeline(steps=[('model', model_reg)]).set_params(**mod_dict_reg)
        print("Regressor with cl train score is: ")
        print(pipe_reg_cl.score(x_train_reg_cl,y_train_reg))
        print("Regressor with cl valid score is: ")
        print(cross_val_score(pipe_reg_cl_cv, x_train_reg_cl,y_train_reg).mean())
        print("Regressor with cl test score is: ")
        print(pipe_reg_cl.score(x_test_reg_cl,y_test_reg))
    except (ValueError, MemoryError) as err:
        print ("GREP_ME***Error caught for  ["  + name_class + "]")
        print(mod_dict_class)
        print ("GREP_ME***Error caught for  ["  + name_reg + "]")
        print(mod_dict_reg)
        print(err)
        pass    
    
    sys.stdout.flush()

def pre_run_model_search(x_train_reg, y_train_reg, x_test_reg, y_test_reg, model_reg, cfg_dict_reg, pipeline_cfg_reg, res_dict_class, results):
    global itter_current
    itter_current += 1
    #create pipline for the preprocessing
    name_reg = type(model_reg).__name__
    
    print('####################################################################################')
    print('################# Runing the itteration %d  of model search      ###############' %(itter_current))
    print('####################################################################################')

    #itterate over each cfg variation and compute the result
    param_grid = dict(models_reg_cfg[name_reg])
    print(param_grid)
    processes = []
    processes_args = []
    local_ind = 0       
    for _terms  in list(product(*[models_reg_cfg[name_reg][it] for it in models_reg_cfg[name_reg]])):
            local_ind += 1
            mod_dict_reg = dict((term, _terms[ind]) for ind, term in enumerate(tuple(it for it in models_reg_cfg[name_reg])))
            processes_args.append((x_train_reg, y_train_reg, x_test_reg, y_test_reg, model_reg, mod_dict_reg, cfg_dict_reg, pipeline_cfg_reg, res_dict_class, results))

    Parallel(n_jobs=-1, verbose=1)(delayed(run_model_search_instance)(*p_args) for p_args in processes_args)           


def run_model_search(x_train_reg, y_train_reg, x_test_reg, y_test_reg, models_reg, cfg_dict_reg, pipeline_cfg_reg, res_dict_class, results):
        for model_reg in models_reg:
            pre_run_model_search(x_train_reg, y_train_reg, x_test_reg, y_test_reg, model_reg, cfg_dict_reg, pipeline_cfg_reg, res_dict_class, results)
            
def precompute(x,y, preprocessors, transfomers,reducers, errors_ind, model_dir):
        
    n_samples, n_features = x[0].shape   
    #precompute the preprocessing results so that can be resued by grisearch and dont need to be precomputed.    
    for preprocessor, transfomer, reducer in product(preprocessors, transfomers, reducers):
       ##run gridesearch with new amout of features, depending of preprocessor and hence pass the right amount of maximum components to the reducers 
        if preprocessor.func.__name__ == LogarithmicTransformer.func.__name__ :
            n_components = get_components_list(n_features, [{"pw":2}, {"pw":1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            get_pipe_result(x, y,preprocessor, transfomer, reducer, errors_ind, model_dir)
        elif preprocessor.func.__name__ == PolynomialTransformer.func.__name__ or preprocessor.func.__name__ == LogPolynomialTransformer.func.__name__:
            LogPol = False
            if preprocessor.func.__name__ == LogPolynomialTransformer.func.__name__: LogPol = True
            kw_arg_powers = get_powers_list(n_samples, n_features, 3)
            pw_lst = []
            for pw in kw_arg_powers:
                pw_lst = pw_lst + [pw]
                preprocessors_cfg[preprocessor.func.__name__]["preprocessor__kw_args"] = [pw]
                n_components = get_components_list(n_features, pw_lst, LogPol)
                reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
                reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
                reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
                get_pipe_result(x, y,preprocessor, transfomer, reducer, errors_ind, model_dir)
        else:
            n_components = get_components_list(n_features, [{"pw":1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            get_pipe_result(x, y,preprocessor, transfomer, reducer, errors_ind, model_dir)
        
def run_solver(data_train,data_test, preprocessors, transfomers, reducers, models_class, models_reg, results, errors_ind, label_fn, new_file):
       
    #make a dir for preprocessor temp files
    try:
        os.mkdir("./tmp_reg")
    except FileExistsError:
        #rm temp dir and make new one
        shutil.rmtree("./tmp_reg")
        os.mkdir("./tmp_reg")
    
        #make a dir for preprocessor temp files
    try:
        os.mkdir("./tmp_class")
    except FileExistsError:
        #rm temp dir and make new one
        shutil.rmtree("./tmp_class")
        os.mkdir("./tmp_class")
        
    try:
        os.mkdir("./tmp_class_res")
    except FileExistsError:
        #rm temp dir and make new one
        shutil.rmtree("./tmp_class_res")
        os.mkdir("./tmp_class_res")
    
    x_train = data_train.drop('worldwide_gross', axis=1)
    y_train_cl = data_train.worldwide_gross.apply (lambda gross: label_fn (gross))
    y_train_reg = data_train['worldwide_gross']

    x_test = data_test.drop('worldwide_gross', axis=1)
    y_test_cl = data_test.worldwide_gross.apply (lambda gross: label_fn (gross))
    y_test_reg = data_test['worldwide_gross']

    
    #precomp pipeline for calssification
    precompute((x_train, x_test),(y_train_cl, y_test_cl), preprocessors, transfomers,reducers, errors_ind, 'tmp_class')
    
    ##precomp pipeline for regression
    precompute((x_train, x_test),(y_train_reg, y_test_reg), preprocessors, transfomers,reducers, errors_ind, 'tmp_reg')
  
    print("i have finshed precomputing pipeline")

    #precomp results for calssification
    for filename_class in os.listdir("./tmp_class"):
        pipe_dict_class = pickle.loads(open("./tmp_class/" + filename_class, 'rb').read())
        run_class_precomp(        pipe_dict_class['x_train'],
                                  pipe_dict_class['y_train'],
                                  pipe_dict_class['x_test'],
                                  pipe_dict_class['y_test'],
                                               models_class,
                                  pipe_dict_class['cfg_dict'],
                                  pipe_dict_class['pipeline_cfg']
                            )
    print("i have finshed precomputing classification")

    #for each physically saved pickle run grid search for each model
    for filename_class in os.listdir("./tmp_class_res"):
        res_dict_class = pickle.loads(open("./tmp_class_res/" + filename_class, 'rb').read())
        for filename_reg in os.listdir("./tmp_reg"):
            pipe_dict_reg = pickle.loads(open("./tmp_reg/" + filename_reg, 'rb').read())
            run_model_search(pipe_dict_reg['x_train'],  pipe_dict_reg['y_train'], pipe_dict_reg['x_test'], pipe_dict_reg['y_test'], models_reg,  pipe_dict_reg['cfg_dict'], pipe_dict_reg['pipeline_cfg'], res_dict_class, results)                                     

def run_for_many(cl_n,label_fn, new_file):
    results = {}
    errors_ind = []
    data_train = dta_train
    data_test = dta_test
    #y = dta_clean['worldwide_gross']
    print ("#########################################")
    print ("###Starting all estimators for cl: "+ str(cl_n))
    print ("#########################################")
    run_solver(data_train,data_test, preprocessors, transfomers, reducers, models_class, models_reg, results, errors_ind, label_fn, new_file)
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

desc = "no_imdb_gradient_boost_class_with_regression_final"


#labels = [label_gross_10, label_gross_9, label_gross_8, label_gross_7, label_gross_6, label_gross_5, label_gross_4, label_gross_3, label_gross_2]
#labels = [label_gross_3]


##Ronald
#labels = [label_gross_3, label_gross_2]

##Kelvin
#labels = [label_gross_5, label_gross_4]

##Su
#labels = [label_gross_7, label_gross_6]

##Ahmet
#labels = [label_gross_9, label_gross_8]

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
        run_for_many(cb.__name__,cb, trg)
        #return stdout for some reason
sys.stdout = orig_stdout


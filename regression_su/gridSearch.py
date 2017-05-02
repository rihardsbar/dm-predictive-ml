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
# import matplotlib
import numpy as np
# import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge as br
from sklearn.linear_model import ElasticNet as en
from sklearn.linear_model import Lars as ls
from sklearn.linear_model import Lasso as lo
from sklearn.linear_model import LassoLars as ll
from sklearn.linear_model import HuberRegressor as hr


from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.feature_selection import GenericUnivariateSelect, RFE
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression

from itertools import  product
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline, make_pipeline

import math
# import statsmodels.api as sm
# from patsy import dmatrices
from sklearn import metrics

# import seaborn as sns # More snazzy plotting library
import itertools
from itertools import  product
import pprint
from sklearn.ensemble import ExtraTreesRegressor


# f = pd.read_csv(input_folder+"/movie_metadata.csv")
# f = pd.read_csv(input_folder+"/movie_metadata_cleaned_categ_num_only.csv")
# dta_clean = f.dropna()

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
# import matplotlib
import numpy as np
# import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge as br
from sklearn.linear_model import ElasticNet as en
from sklearn.linear_model import Lars as ls
from sklearn.linear_model import Lasso as lo
from sklearn.linear_model import LassoLars as ll
from sklearn.linear_model import HuberRegressor as hr


from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.feature_selection import GenericUnivariateSelect, RFE
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression

from itertools import  product
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline, make_pipeline

import math
# import statsmodels.api as sm
# from patsy import dmatrices
from sklearn import metrics

# import seaborn as sns # More snazzy plotting library
import itertools
from itertools import  product
import pprint
from sklearn.ensemble import ExtraTreesRegressor

from multiprocessing import Process, Value, Array
from asyncio import Queue
from threading import Thread
import pickle
import shutil

f = pd.read_csv(input_folder+"/no_imdb_names-count_cat-tf_184f.csv")
# f = pd.read_csv(input_folder+"/movie_metadata_cleaned_categ_num_only.csv")
dta_clean = f.dropna()
dta_clean = dta_clean.drop('Unnamed: 0', axis=1)

X_a = dta_clean.drop('worldwide_gross', axis=1)
y_a = dta_clean['worldwide_gross']

df_1_1 = dta_clean[dta_clean["worldwide_gross"] < 200000000]
X_1_1 = df_1_1.drop('worldwide_gross', axis=1)
y_1_1 = df_1_1['worldwide_gross']

df_1_2 = dta_clean[dta_clean["worldwide_gross"] >= 200000000]
X_1_2 = df_1_2.drop('worldwide_gross', axis=1)
y_1_2 = df_1_2['worldwide_gross']

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

#shuffle the whole dataset
X_a, X_d, y_a, y_d = train_test_split(X_a, y_a, test_size=0, random_state=0)
#shuffle the whole dataset
X_1, X_d, y_1, y_d = train_test_split(X_1, y_1, test_size=0, random_state=0)
#shuffle the whole dataset
X_2, X_d, y_2, y_d = train_test_split(X_2, y_2, test_size=0, random_state=0)
#shuffle the whole dataset
X_3, X_d, y_3, y_d = train_test_split(X_3, y_3, test_size=0, random_state=0)


##define helpers
def get_powers_list(n_samples, n_features, n):
    return [{"pw":1},{"pw":2},{"pw":3},{"pw":4}]

def get_components_list(n_features, lst):
    max_pw = max(lst, key=lambda x: x["pw"])["pw"]
    current_feat = 10*max_pw + n_features - 10
    lst = [{"pw": 0.1},{"pw": 0.45},{"pw": 0.5},{"pw": 0.8}, {"pw": 0.2},{"pw": 0.65},{"pw": 0.99}]
    lst = sorted(list(map(lambda x: math.floor(x["pw"]*current_feat), lst)) + [1, 3, 5], reverse=True)
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

###define a global itteration var###########
itter_start   = 0
itter_current = 0

DummyTransformer = FunctionTransformer(dummy)
LogarithmicTransformer = FunctionTransformer(log)
PolynomialTransformer = FunctionTransformer(poly)

###define new config###########

#########################
####Data Preprocessor ###
#########################
preprocessors = [DummyTransformer, LogarithmicTransformer, PolynomialTransformer]
preprocessors_cfg = {}
preprocessors_cfg[DummyTransformer.func.__name__] = {}
preprocessors_cfg[LogarithmicTransformer.func.__name__] = {}
preprocessors_cfg[PolynomialTransformer.func.__name__] = dict(
        preprocessor__kw_args = []
        )

#########################
####  Data Transformer ##
#########################
transfomers = [DummyTransformer, Normalizer(), StandardScaler()]
transfomers_cfg = {}
transfomers_cfg[DummyTransformer.func.__name__] = {}
transfomers_cfg[Normalizer.__name__] = dict(
        transfomer__norm = ['l1', 'l2', 'max']
        )
transfomers_cfg[StandardScaler.__name__] = {}

###########################
####Dim Reducer, Feat Sel.#
###########################
reducers = [DummyTransformer, PCA(), GenericUnivariateSelect(), RFE(ExtraTreesRegressor())]
reducers_cfg = {}
reducers_cfg[DummyTransformer.func.__name__] = {}
reducers_cfg[PCA.__name__] = dict(
        reducer__n_components = [],
        # reducer__whiten = [True, False],
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

# models = [br(), en(), ls(), lo(), ll()]
models = [br(), en(), ll()]

models_cfg = {}

def init(para = None):
    # print (para);
    if para == 2:

        #########################
        ####Data Preprocessor ###
        #########################

        preprocessors = [DummyTransformer]
        preprocessors_cfg = {}
        preprocessors_cfg[DummyTransformer.func.__name__] = {}


        #########################
        ####  Data Transformer ##
        #########################
        transfomers = [DummyTransformer]
        transfomers_cfg = {}
        transfomers_cfg[DummyTransformer.func.__name__] = {}

        ###########################
        ####Dim Reducer, Feat Sel.#
        ###########################
        reducers = [DummyTransformer]
        reducers_cfg = {}
        reducers_cfg[DummyTransformer.func.__name__] = {}

        models_cfg[br.__name__] = dict(
            # model__n_iter = [10, 50, 100, 150, 200, 250, 300, 350],
            model__compute_score = [True, False],
            # model__fit_intercept = [True, False],
            model__normalize = [True, False],
            # model__copy_X = [True, False],
            # model__verbose = [True, False]
            )

        models_cfg[en.__name__] = dict(
            # model__max_iter = [10, 50, 100, 150, 200, 250, 300, 350],
            model__precompute = [True, False, 'auto'],
            # model__fit_intercept = [True, False],
            # model__normalize = [True, False],
            # model__copy_X = [True, False],
            # model__warm_start = [True, False],
            model__positive = [True, False],
            model_selection = ['cyclic', 'random']
            )

        # models_cfg[ls.__name__] = dict(
        #     model__precompute = [True, False, 'auto'],
        #     # model__fit_intercept = [True, False],
        #     # model__normalize = [True, False],
        #     # model__copy_X = [True, False],
        #     model__fit_path = [True, False],
        #     model__positive = [True, False],
        #     # model_verbose = [True, False]
        #     )

        # models_cfg[lo.__name__] = dict(
        #     # model__max_iter = [10, 50, 100, 150, 200, 250, 300, 350],
        #     model__precompute = [True, False, 'auto'],
        #     # model__fit_intercept = [True, False],
        #     # model__normalize = [True, False],
        #     # model__copy_X = [True, False],
        #     model__warm_start = [True, False],
        #     model__positive = [True, False],
        #     model_selection = ['cyclic', 'random']
        #     )

        models_cfg[ll.__name__] = dict(
            # model__max_iter = [10, 50, 100, 150, 200, 250, 300, 350],
            model__precompute = [True, False, 'auto'],
            # model__fit_intercept = [True, False],
            # model__normalize = [True, False],
            # model__copy_X = [True, False],
            model__fit_path = [True, False],
            model__positive = [True, False],
            # model_verbose = [True, False]
            )
        # models_cfg[hr.__name__] = dict(
        #   )
    else:
        models_cfg[br.__name__] = dict(
            )

        models_cfg[en.__name__] = dict(
            )

        # models_cfg[ls.__name__] = dict(
        #     )

        # models_cfg[lo.__name__] = dict(
        #     )

        models_cfg[ll.__name__] = dict(
            )
        # models_cfg[hr.__name__] = dict(
        #   )


def run_grid_search(x,y, model, cfg_dict, pipeline_cfg, results, errors, errors_ind):
    global itter_current
    itter_current += 1
    #check if itteration start is set to something different than 0 and then check if current itteration has been reached
    if itter_start != 0 and itter_current < itter_start: return
    #create pipline and use GridSearch to find the best params for given pipeline
    name = type(model).__name__

    #Define and save pipe cfg
    pipe = Pipeline(steps=[('model', model)])

    #create a dict with param grid
    param_grid = models_cfg[name]
    #create estimator
    cv = 4
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

def launch_pipe_instance(x, y, pipe, cfg_dict, pipeline_cfg, precomp_pipe, errors, errors_ind, ind):
    print("Starting precomp pipline for " + str(cfg_dict))
    # run the pipe, except exceptions, save errors
    try:
        # precomp_pipe.put_nowait({"pipeline_cfg": pipeline_cfg, "cfg_dict": cfg_dict,"precomp_transform": pipe.set_params(**cfg_dict).fit_transform(x,y)})
        dump_dict = {"pipeline_cfg": pipeline_cfg, "cfg_dict": cfg_dict,
                     "precomp_transform": pipe.set_params(**cfg_dict).fit_transform(x, y)}
        tmp_trg = "./tmp/" + str(itter_current) + "_" + str(ind)
        with open(tmp_trg, 'wb') as handle:
            pickle.dump(dump_dict, handle)
        print("Finished precomp pipline for " + str(cfg_dict))

    except (ValueError, MemoryError) as err:
        print("GREP_ME***Error caught for  precomp pipeline: [" + pipeline_cfg + "] ")
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

# def run_solver(x,y,preprocessors, transfomers, reducers, models, results, errors, errors_ind):
def run_solver(x,y,preprocessors, transfomers, reducers, models, results, errors, errors_ind, precomp_pipe):
    # mix it, so that the sample order is randomized
    x, _X_dummy, y, _y_dummy = train_test_split(x, y, test_size=0)
    n_samples, n_features = x.shape

    #make a dir for preprocessor temp files
    try:
        os.mkdir("./tmp")
    except FileExistsError:
        #rm temp dir and make new one
        shutil.rmtree("./tmp")
        os.mkdir("./tmp")

    # for preprocessor, transfomer, reducer, model in product(preprocessors, transfomers, reducers, models):
    for preprocessor, transfomer, reducer in product(preprocessors, transfomers, reducers):
        ##run gridesearch with new amout of features, depending of preprocessor and hence pass the right amount of maximum components to the reducers
        if preprocessor.func.__name__ == LogarithmicTransformer.func.__name__ :
            n_components = get_components_list(n_features, [{"pw":2}, {"pw":1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            # run_grid_search(x,y,preprocessor, transfomer, reducer, model, results, errors, errors_ind)
            get_pipe_result(x, y,preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind)
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
                # run_grid_search(x,y,preprocessor, transfomer, reducer, model, results, errors, errors_ind)
                get_pipe_result(x, y,preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind)
        else:
            n_components = get_components_list(n_features, [{"pw":1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            # run_grid_search(x,y,preprocessor, transfomer, reducer, model, results, errors, errors_ind)
            get_pipe_result(x, y,preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind)

    #for each physically saved pickle run grid search for each model
    for filename in os.listdir("./tmp"):
        pipe_dict = pickle.loads(open("./tmp/" + filename, 'rb').read())
        for model in models: 
            run_grid_search( pipe_dict['precomp_transform'],y, model,  pipe_dict['cfg_dict'],  pipe_dict['pipeline_cfg'], results, errors, errors_ind)

# def run_for_many(cl_n,label_fn):
def run_for_many(X, y, sam):
    results = {}
    errors = []
    errors_ind = []
    precomp_pipe = []

    print ("#########################################")
    # print ("###Starting all estimators for cl: "+ str(cl_n))
    print ("###Starting all estimators for cl: " + sam)
    print ("#########################################")
    # run_solver(X,y, preprocessors, transfomers, reducers, models, results, errors, errors_ind)
    run_solver(X,y, preprocessors, transfomers, reducers, models, results, errors, errors_ind, precomp_pipe)
    print ("#########################################")
    # print ("###Finished all estimators for cl: "+ str(cl_n))
    print ("###Finished all estimators for cl: " + sam)
    print ("#########################################")

    print ("#########################################")
    # print ("######Printing all errors for cl: "+ str(cl_n))
    print ("######Printing all errors for cl: " + sam)
    print ("#########################################")
    print(errors)
    print ("#########################################")
    # print ("######Printing errors summary for cl: "+ str(cl_n))
    print ("######Printing errors summary for cl: " + sam)
    print ("#########################################")
    print(errors_ind)
    print ("#########################################")
    # print ("#######Printing results for cl: "+ str(cl_n))
    print ("#######Printing results for cl: " + sam)
    print ("#########################################")
    print(results)
    print("priting simply sorted numbers, grep them to find the best cfg or cl: " + sam)
    scores = [results[model]["score"] for model in results]
    print(sorted(scores))


#ignore warnigs


tuples_of_data = [(X_a,y_a, "all_samples"), 
(X_1_1,y_1_1, "samples1_class1"), (X_1_2,y_1_2, "samples1_class2"),
(X_1,y_1, "samples2_class1") , (X_2,y_2, "samples2_class2"), (X_3,y_3, "samples2_class3")]

# tuples_of_data = [(X_1_1,y_1_1, "samples1_class1"), (X_1_2,y_1_2, "samples1_class2"),
# (X_1,y_1, "samples2_class1") , (X_2,y_2, "samples2_class2"), (X_3,y_3, "samples2_class3")]

# labels = [label_gross_3, label_gross_2, label_gross_4, label_gross_5]
#save orig datetime and save orign stdout
orig_stdout = sys.stdout

# for ind, cb in enumerate(labels):
for x in range(1,3):

    for item in tuples_of_data:
        
        with warnings.catch_warnings():
            time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            warnings.simplefilter("ignore")
            # trg = "classifyRes_" + time + "_" + cb.__name__ + ".log"
            trg = "classifyRes_" + time + "_" + item[2] + '_' + str(x) + ".log"
            new_file = open(trg,"w")
            sys.stdout = new_file
            init(x)
            # print (trg + str(x))
            # break
            run_for_many(item[0], item[1], item[2])
            #return stdout for some reason
sys.stdout = orig_stdout




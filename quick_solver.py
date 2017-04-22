import os
import sys
import time
import warnings
import datetime
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

warnings.filterwarnings('ignore')

### Global variables
###define a global itteration var###########
itter_start = 0
itter_current = 0


## Define new transformers
def dummy(X):
    return X


def poly(X, pw):
    res = X
    for power in range(2, pw + 1):
        res = np.concatenate((res, np.power(X, power)), axis=1)
    return res


def log(X):
    df_t = pd.DataFrame(X)
    X_t = df_t.replace(0, 1 / math.e)
    return np.concatenate((X, np.log(X_t)), axis=1)


DummyTransformer = FunctionTransformer(dummy)
LogarithmicTransformer = FunctionTransformer(log)
PolynomialTransformer = FunctionTransformer(poly)

#################################
### Default Data Preprocessor ###
#################################
#preprocessors = [DummyTransformer, LogarithmicTransformer, PolynomialTransformer]
preprocessors = [DummyTransformer]
preprocessors_cfg = dict()
preprocessors_cfg[DummyTransformer.func.__name__] = {}
preprocessors_cfg[LogarithmicTransformer.func.__name__] = {}
preprocessors_cfg[PolynomialTransformer.func.__name__] = dict(
    preprocessor__kw_args=[]
)

################################
#### Default Data Transformer ##
################################
#transfomers = [DummyTransformer, StandardScaler()]
transfomers = [DummyTransformer]
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
reducers = [DummyTransformer]
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

##################################################
### Functions
##################################################
def get_powers_list(n_samples, n_features, n):
    base_arr = [{"pw": 2}, {"pw": 3}, {"pw": 4}]
    max_pw = math.ceil(3320 / n_features)
    if max_pw > 7: max_pw = 7
    step = math.floor((max_pw - 4) / n)
    if step < 1: step = 1
    extra_arr = [{"pw": power} for power in range(4 + step, max_pw, step)]
    if n_samples / n_features < 2:
        res = [{"pw": 1}]
    elif max_pw - 1 == 2:
        res = [{"pw": 2}]
    elif max_pw - 1 == 3:
        res = [{"pw": 2}, {"pw": 3}]
    elif max_pw - 1 == 4:
        res = [{"pw": 2}, {"pw": 3}, {"pw": 4}]
    else:
        res = base_arr + extra_arr
    return res


def get_components_list(n_features, lst):
    lst = lst + [{"pw": 0.1}, {"pw": 0.4}, {"pw": 0.5}, {"pw": 0.8}]
    lst = sorted(list(map(lambda x: math.floor(x["pw"] * n_features), lst)) + [1, 3, 5], reverse=True)
    lst[0] = lst[0] - 1
    lst_n = [n for n in lst if n < 3321]
    if len(lst_n) < len(lst):
        lst_n = [3320] + lst_n
    return lst_n


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
        errors_ind.append({"cfg": pipeline_cfg})
        errors.append({"Precomp pipe: " + pipeline_cfg: {"error": err}})
        pass


def get_pipe_result(x, y, preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind):
    global itter_current
    itter_current += 1
    # create pipline for the preprocessing
    preprocessor_name = type(preprocessor).__name__ if (
    type(preprocessor).__name__ != "FunctionTransformer") else preprocessor.func.__name__
    transfomer_name = type(transfomer).__name__ if (
    type(transfomer).__name__ != "FunctionTransformer") else  transfomer.func.__name__
    reducer_name = type(reducer).__name__ if (
    type(reducer).__name__ != "FunctionTransformer") else reducer.func.__name__

    print('####################################################################################')
    print('################# Runing the itteration %d  of pipeline precomp      ###############' % (itter_current))
    print('####################################################################################')

    # Define and save pipe cfg
    pipeline_cfg = "| preprocessor:" + preprocessor_name + " | transfomer: " + transfomer_name + " | reducer: " + reducer_name
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('transfomer', transfomer), ('reducer', reducer)])
    print(pipeline_cfg)

    # itterate over each cfg variation and precompute the result
    param_grid = dict(dict(preprocessors_cfg[preprocessor_name],
                           **dict(transfomers_cfg[transfomer_name], **reducers_cfg[reducer_name])))
    print(param_grid)
    processes = []
    local_ind = 0
    for _terms in list(
            product(*[preprocessors_cfg[preprocessor_name][it] for it in preprocessors_cfg[preprocessor_name]])):
        cfg_dict = dict(
            (term, _terms[ind]) for ind, term in enumerate(tuple(it for it in preprocessors_cfg[preprocessor_name])))

        for _terms in list(product(*[transfomers_cfg[transfomer_name][it] for it in transfomers_cfg[transfomer_name]])):
            cfg_dict.update(dict(
                (term, _terms[ind]) for ind, term in enumerate(tuple(it for it in transfomers_cfg[transfomer_name]))))

            for _terms in list(product(*[reducers_cfg[reducer_name][it] for it in reducers_cfg[reducer_name]])):
                cfg_dict.update(dict(
                    (term, _terms[ind]) for ind, term in enumerate(tuple(it for it in reducers_cfg[reducer_name]))))
                # launch in a parraler manner a pipe dict
                # launch_pipe_instance(x,y, pipe, cfg_dict, pipeline_cfg, precomp_pipe, errors, errors_ind, local_ind)
                local_ind += 1
                final_cfg = cfg_dict
                p = Process(target=launch_pipe_instance,
                            args=(x, y, pipe, cfg_dict, pipeline_cfg, precomp_pipe, errors, errors_ind, local_ind))
                p.start()
                processes.append(p)

    for p in processes:
        p.join()


def run_grid_search(x, y, model, model_cfg, cfg_dict, pipeline_cfg, results, errors, errors_ind):
    global itter_current
    itter_current += 1
    # check if itteration start is set to something different than 0 and then check if current itteration has been reached
    if itter_start != 0 and itter_current < itter_start: return
    # create pipline and use GridSearch to find the best params for given pipeline
    name = type(model).__name__

    # Define and save pipe cfg
    pipe = Pipeline(steps=[('model', model)])

    # create a dict with param grid
    param_grid = model_cfg[name]
    # create estimator
    cv = 4
    print('####################################################################################')
    print('################# Running the iteration %d  of the GridSearchCV ####################' % (itter_current))
    print('####################################################################################')
    print("***Starting [" + name + "] estimator run, pipeline: " + pipeline_cfg + " ")
    print("##param_grid##")
    print(param_grid)
    estimator = GridSearchCV(pipe, param_grid, verbose=2, cv=cv, n_jobs=-1)
    # run the estimator, except exceptions, sape errors
    try:
        estimator.fit(x, y)
        print("GREP_ME***Results of [" + name + "] estimatorrun are")
        print(estimator.cv_results_)
        print("GREP_ME***Best params of [" + name + "] estimator,pipeline:" + pipeline_cfg + "  run are")
        best_param = dict(estimator.best_params_, **cfg_dict)
        print(best_param)
        print("GREP_ME***Best score of [" + name + "] estimator, pipeline:" + pipeline_cfg + " run are")
        print(estimator.best_score_)
        if (name not in results) or (estimator.best_score_ > results[name]["score"]):
            results[name] = {"score": estimator.best_score_, "pipe": pipeline_cfg, "best_cfg": best_param}
    except (ValueError, MemoryError) as err:
        print("GREP_ME***Error caught for  [" + name + "] , pipeline: [" + pipeline_cfg + "] ")
        print(err)
        pass

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
    

# def run_solver(x,y,preprocessors, transfomers, reducers, models, results, errors, errors_ind, precomp_pipe):
def run_solver(x, y, models, models_cfg, results, errors, errors_ind, precomp_pipe):
    n_samples, n_features = x.shape
    # simply schuffle the data so that cross validation gets random samples
    x, _X_dummy, y, _y_dummy = train_test_split(x, y, test_size=0)
    # Make a dir for preprocessor temp files
    try:
        os.mkdir("./tmp")
    except FileExistsError:
        # rm temp dir and make new one
        shutil.rmtree("./tmp")
        os.mkdir("./tmp")

    # Precompute the preprocessing results so that can be reused by gridsearch and dont need to be precomputed.
    ts = time.time()
    for preprocessor, transfomer, reducer in product(preprocessors, transfomers, reducers):
        # Run gridesearch with new amout of features, depending of preprocessor and hence pass the right amount of maximum components to the reducers
        if preprocessor.func.__name__ == LogarithmicTransformer.func.__name__:
            n_components = get_components_list(n_features, [{"pw": 2}, {"pw": 1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            get_pipe_result(x, y, preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind)
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
                get_pipe_result(x, y, preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind)
        else:
            n_components = get_components_list(n_features, [{"pw": 1}])
            reducers_cfg[PCA.__name__]["reducer__n_components"] = n_components
            reducers_cfg[GenericUnivariateSelect.__name__]["reducer__param"] = n_components
            reducers_cfg[RFE.__name__]["reducer__n_features_to_select"] = n_components
            get_pipe_result(x, y, preprocessor, transfomer, reducer, precomp_pipe, errors, errors_ind)

    print("Pre-computation of pre-processing models completed in {}".format(datetime.timedelta(seconds=time.time()-ts)))

    # for each physically saved pickle run grid search for each model
    for filename in os.listdir("./tmp"):
        pipe_dict = pickle.loads(open("./tmp/" + filename, 'rb').read())
        # for model in models:
        for model in models:
            run_grid_search(pipe_dict['precomp_transform'], y, model, models_cfg, pipe_dict['cfg_dict'], pipe_dict['pipeline_cfg'],
                            results, errors, errors_ind)

## Function for trigrering gridserach and priting results
def run_for_many(x, y, cl_n, models, models_cfg):
    results = {}
    precomp_pipe = []
    errors = []
    errors_ind = []
    print("#########################################")
    print("###Starting all estimators for cl: " + str(cl_n))
    print("#########################################")
    run_solver(x, y, models, models_cfg, results, errors, errors_ind, precomp_pipe)
    print("#########################################")
    print("###Finished all estimators for cl: " + str(cl_n))
    print("#########################################")

    print("#########################################")
    print("#######Printing results for cl: " + str(cl_n))
    print("#########################################")
    print(results)
    print("priting simply sorted numbers, grep them to find the best cfg or cl: " + str(cl_n))
    scores = [results[model]["score"] for model in results]
    print(sorted(scores))


# The code in this function is meant to be a starting point for evaluating other models
def simple_experiment(file_path):

    # Read data
    dta = pd.read_csv(file_path)
    dta_clean = dta
    # remove the null values, that is fill NaN with there - FIXME: Rihards, naive implementation
    dta_clean = dta_clean.fillna(value=0, axis=1)
    dta_clean = dta_clean.dropna()
    dta_clean = dta_clean.drop('Unnamed: 0', axis=1)
    
    
    #########################
    ####### Models ##########
    #########################
    models_class = [AdaBoostClassifier(),BaggingClassifier(),ExtraTreesClassifier(),GradientBoostingClassifier(),RandomForestClassifier(),PassiveAggressiveClassifier(),LogisticRegression(),RidgeClassifier(),SGDClassifier(),GaussianNB(),MultinomialNB(),KNeighborsClassifier(),RadiusNeighborsClassifier(),NearestCentroid(),MLPClassifier(),SVC(),LinearSVC(),NuSVC(),DecisionTreeClassifier(),ExtraTreeClassifier()]
    
    models_reg = [AdaBoostRegressor(),BaggingRegressor(),ExtraTreesRegressor(),GradientBoostingRegressor(),RandomForestRegressor(),ElasticNet(),HuberRegressor(),Lars(),Lasso(),LassoLars(),LinearRegression(),PassiveAggressiveRegressor(),Ridge(),SGDRegressor(),OrthogonalMatchingPursuit(),RANSACRegressor(),KNeighborsRegressor(),RadiusNeighborsRegressor(),MLPRegressor(),SVR(),LinearSVR(),NuSVR(),DecisionTreeRegressor(),ExtraTreeRegressor()]
    
    models_cfg = {}
    
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

    ##to run for multiple classes of data, add the tuples of x and y  to the tuples array of data and decsription for the purposes logging. For now it is set to run for all the samples there are. For instance tuples_of_data = [(X,y, "all samples"), (X_1,y_1, "samples class1") , (X_2,y_2", "samples class2")]
    # for each tupple extracted from the array a new log file is going to be generated, so that each run is in a different log file.
    
    X_all = dta_clean.drop('worldwide_gross', axis=1)
    y_all = dta_clean['worldwide_gross']
    desc = "quickReg" + file_path.replace('.','').replace('/','').replace('dataset','').replace('csv','')
    tuples_of_data = [(X_all, y_all, desc)]
    #########################
    ### Start Regress########
    #########################
    orig_stdout = sys.stdout #  save orig datetime and save orign stdout
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    for ind, tupl in enumerate(tuples_of_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
            run_for_many(x_crr, y_crr, dsc, models_reg, models_cfg)
            new_file.close()
            
    
    desc = "quickClass" + file_path.replace('.','').replace('/','').replace('dataset','').replace('csv','')
    labels = [label_gross_3, label_gross_2, label_gross_4, label_gross_5]
    #save orig datetime and save orign stdout
    orig_stdout = sys.stdout
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
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
            x_crr = dta_clean.drop('worldwide_gross', axis=1)
            y_crr = dta_clean.worldwide_gross.apply (lambda gross: cb (gross))
            dsc = desc + "_" + cb.__name__
            run_for_many(x_crr, y_crr, dsc, models_class, models_cfg)
            new_file.close()
    
    # reassign the org stdout for some reason
    sys.stdout = orig_stdout


if __name__ == "__main__":
    files = [
        "./dataset/movie_metadata_cleaned_tfidf_num_only_min.csv",
        "./dataset/movie_metadata_cleaned_categ_num_only.csv",
        "./dataset/movie_metadata_cleaned_no_vector_num_only.csv",
        "./dataset/movie_metadata_cleaned_cat-name_vector_no_imbd.csv",
        "./dataset/movie_metadata_cleaned_cat_vector_no_imbd.csv"
    ]
    for file in files: simple_experiment(file)

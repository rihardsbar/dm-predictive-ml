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
import seaborn as sns # More snazzy plotting library
import itertools
import warnings

##test results
#  1.looks the absolute year number gives a good colleration for prediction.
#  2. standarlistaion improves prediction by 3%
#  3. removing aspect ratio improves prediction by 2%

#open the file
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

#clasify the data for the logistic regression
def label_gross (gross):
    if (gross < 1000000) : return 1
    elif ((gross >= 1000000) & (gross < 10000000)) : return 2
    elif ((gross >= 10000000) & (gross < 50000000)) : return 3
    elif ((gross >= 50000000) & (gross < 200000000)) : return 4
    elif (gross >= 200000000) : return 5

dta_clean['gross_class'] = dta_clean.gross.apply (lambda gross: label_gross (gross))

#define regression function
def run_logistic_regression(yIn, xIn, solver, prct_split=0.3, scale=False, stadartlize=False, normalize=False, print_intr=False):
        #flatten the y vector
        yIn = np.ravel(yIn)

        #do data tranformation
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            for colname, colvalue in xIn.iteritems():
                xIn[colname] = min_max_scaler.fit_transform(xIn[colname])

        if stadartlize:
            for colname, colvalue in xIn.iteritems():
                standard_scaler = preprocessing.StandardScaler().fit(xIn[colname])
                xIn[colname] = standard_scaler.transform(xIn[colname])

        if normalize:
            for colname, colvalue in xIn.iteritems():
                nomalizer_scaler = preprocessing.Normalizer().fit(xIn[colname])
                xIn[colname] = nomalizer_scaler.transform(xIn[colname])[0]

        #train the model
        X_train, X_test, y_train, y_test = train_test_split(xIn, yIn, test_size=prct_split, random_state=0)
        model = LogisticRegression(solver=solver)
        model.fit(X_train, y_train)

        # predict class labels for the test set
        predicted = model.predict(X_test)

        # find the avarge score 10 iterations
        scores = cross_val_score(model, xIn, yIn, scoring='accuracy', cv=10)

        #print findings
        if print_intr:
            print "Running with scale: " + str(scale) + ", stadartlize: " + str(stadartlize) + ", normalize: " + str(normalize)
            print "Test data acuracy is: " + str(metrics.accuracy_score(y_test,predicted))
            print  "Avarage model score: " + str(scores.mean())

        return metrics.accuracy_score(y_test,predicted), scores.mean()

#define data trasormation itterator
def run_logc_reg_with_data_trasn(yIn,xIn,prct_split, print_intr):
    prev_test_acc = 0
    prev_avg_acc = 0
    test_acc_str = "No Result"
    avg_ac_str = "No Result"
    for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag']:
        if print_intr:
            print "######################################################"
            print "Starting regressions sequance with solver: " + solver
        for scale in [False, True]:
            for stadartlize in [False, True]:
                for normalize in [False, True]:
                    test_acc, avg_acc = run_logistic_regression(yIn,xIn,solver, prct_split, scale, stadartlize, normalize, print_intr)
                    #save the highest accuracy result and result string
                    if test_acc > prev_test_acc:
                        prev_test_acc = test_acc
                        test_acc_str = "End highest test acc " + str(test_acc) + ", achieved with solver: " + solver + ", scale: " + str(scale) + ", stadartlize: " + str(stadartlize) + ", normalize: " + str(normalize)
                    if avg_acc  > prev_avg_acc :
                        prev_avg_acc = avg_acc
                        avg_ac_str = "End highest avg model acc " + str(avg_acc) + ", achieved with solver: " + solver + ", scale: " + str(scale) + ", stadartlize: " + str(stadartlize) + ", normalize: " + str(normalize)
        if print_intr: print "######################################################"
    print test_acc_str
    print avg_ac_str
    return prev_test_acc, prev_avg_acc

def run_diff_predictors():
    headers = list(dta_clean.columns.values)
    headers.remove('gross_class')
    headers.remove('facenumber_in_poster')
    headers.remove('aspect_ratio')
    headers.remove('gross')
    itter = 0
    prev_test_acc = 0
    prev_avg_acc = 0
    best_test_acc_str = "No results"
    best_avg_acc_str = "No results"
    iterator = 0
    for L in range(4, len(headers)+1):
        for subset in itertools.combinations(headers, L):
            iterator = iterator + 1
    print "Running total " + str(iterator) + " itterators"
    for L in range(4, len(headers)+1):
        for subset in itertools.combinations(headers, L):
            string = "gross_class ~ " + " + ".join(str(x) for x in subset)
            if itter  > 0:
                print "---------------------------------------------------------"
                print str(itter) + " out of " + str(iterator) + " reg for the follow predictors:"
                print(string)
                y, X = dmatrices(string,dta_clean, return_type="dataframe")
                test_acc, avg_acc = run_logc_reg_with_data_trasn(y,X, 0.3, print_intr=False)
            #save best data
                if test_acc > prev_test_acc:
                    prev_test_acc = test_acc
                    best_test_acc_str = "End highest test acc " + str(test_acc) + ", achieved with " + string
                if avg_acc > prev_avg_acc:
                    prev_avg_acc = avg_acc
                    best_avg_acc_str = "End highest avg model acc " + str(avg_acc) + ", achieved with " + string
            itter = itter + 1

    #dump out the results
    print "Ending the predictors"
    print best_test_acc_str
    print best_avg_acc_str

#run_logc_reg_with_data_trasn(y,X,0.3, print_intr=False)
#ignore warnigs
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    run_diff_predictors()

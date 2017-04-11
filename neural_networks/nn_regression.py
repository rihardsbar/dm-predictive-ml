import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

movie = pandas.read_csv('../dataset/movie_metadata_cleaned_tfidf_num_only_min.csv')
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in movie.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
         print(colname)
# Get to the numeric columns by inversion
num_list = movie.columns.difference(str_list)

movie_num = movie[num_list]

#handle the empty values
movie_num = movie_num.fillna(value=0, axis=1)

targets = movie_num.filter(['worldwide_gross'], axis=1)
inputs = movie_num.drop(['worldwide_gross'], axis=1)

X = inputs.values
Y = targets.iloc[:, 0].values
print X
print Y

dimof_input = X.shape[1]

# define base model
def baseline_model(dimof_input=dimof_input):
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=dimof_input, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# define the model
def larger_model(dimof_input=dimof_input):
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=dimof_input, kernel_initializer='normal', activation='relu'))
	model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# fix random seed for reproducibility
seed = 7
# numpy.random.seed(seed)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=5, batch_size=5, verbose=2)

# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# #baseline model
# evaluate model with standardized dataset
# numpy.random.seed(seed)
# estimators = []
# # estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=2)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#larger model
numpy.random.seed(seed)
estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=2)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
import pandas as pd

# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
movie = pd.read_csv('../dataset/movie_metadata_cleaned_tfidf_num_only_min.csv')
# movie = pd.read_csv('/Users/ahmet/Documents/GitHub/dm-predictive-ml/dataset/pima-indians-diabetes.data', header=None)
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in movie.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
         print(colname)
# Get to the numeric columns by inversion
num_list = movie.columns.difference(str_list)

movie_num = movie[num_list]
#del movie # Get rid of movie df as we won't need it now
# movie_num = movie_num.fillna(value=0, axis=1)

targets = movie_num.filter(['worldwide_gross'], axis=1)
inputs = movie_num.drop(['worldwide_gross'], axis=1)
# targets = movie_num.iloc[:, -1]
# inputs = movie_num.iloc[:, :-1]

X = inputs.values
Y = targets.iloc[:, 0].values
# print X
# print Y

dimof_input = X.shape[1]

#create model
model = Sequential()
model.add(Dense(12, input_dim=dimof_input, activation='sigmoid'))
model.add(Dense(dimof_input, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(optimizer=optimizers.SGD(lr=0.01, clipnorm=1.), metrics=['accuracy'])

#not using adam optimizer, loss='binary_crossentropy',

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, validation_split=0.3)


#evaluate
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
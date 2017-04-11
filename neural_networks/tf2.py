from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
movie = pd.read_csv('../dataset/pima-indians-diabetes.data', header=None)
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in movie.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion
num_list = movie.columns.difference(str_list)

movie_num = movie[num_list]
#del movie # Get rid of movie df as we won't need it now
movie_num = movie_num.fillna(value=0, axis=1)

targets = movie_num.iloc[:, -1]
inputs = movie_num.iloc[:, :-1]

X = inputs.values
Y = targets.values
print X
print Y

dimof_input = X.shape[1]

#create model
model = Sequential()
model.add(Dense(12, input_dim=dimof_input, activation='sigmoid'))
model.add(Dense(dimof_input, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, validation_split=0.3)


#evaluate
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
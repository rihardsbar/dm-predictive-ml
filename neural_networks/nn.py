from numpy import exp, array, random, dot
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random as randompy



class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

graph_x = []
graph_y = []
class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            if iteration % 10000 == 0:
                print iteration
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
            # if iteration % 1000 == 0:
            #     print layer2_error
            graph_x.append(iteration)
            graph_y.append(layer2_error[0])

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        # print "    Layer 1 (4 neurons, each with 3 inputs): "
        # print self.layer1.synaptic_weights
        # print "    Layer 2 (1 neuron, with 4 inputs):"
        # print self.layer2.synaptic_weights
        pass
movie = pd.read_csv('/Users/ahmet/Documents/GitHub/dm-predictive-ml/dataset/movie_metadata_budgets.csv')
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in movie.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion
num_list = movie.columns.difference(str_list)

movie_num = movie[num_list]
#del movie # Get rid of movie df as we won't need it now
movie_num = movie_num.fillna(value=0, axis=1)

#normalise the dataframe
movie_num_shuff = movie_num / movie_num.max()
norm_coef = movie_num.max()

#shuffle the list
movie_num_norm = movie_num_shuff.reindex(np.random.permutation(movie_num_shuff.index))

inputs = []
outputs = []
# print movie_num_norm.head()
count = 0
for amovie in movie_num_norm[:100].itertuples():
    if float(amovie.duration) == 0.0 or float(amovie.production_budget) == 0.0 or float(amovie.worldwide_gross) == 0.0:
        count += 1
    else:
        outputs.append(amovie.worldwide_gross)
        inputs.append([amovie.duration, amovie.production_budget, amovie.imdb_score])

print 'Incomplete sets: ', str(count)

# inputs = [[0, 0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
# outputs = [[0, 1, 1, 1, 1, 0, 0]]

inputs_train = []
outputs_train = []
inputs_test = []
outputs_test = []

for i in range(len(inputs)):
    if randompy.random() < .8:
        inputs_train.append(inputs[i])
        outputs_train.append(outputs[i])
    else:
        inputs_test.append(inputs[i])
        outputs_test.append(outputs[i])

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(4, len(inputs_test[0]))

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)
    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print "Stage 1) Random starting synaptic weights: "
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array(inputs_train)
    training_set_outputs = array([outputs_train]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "Stage 2) New synaptic weights after training: "
    neural_network.print_weights()

    # Test the neural network with a new situation.
    # print "Stage 3) Considering a new situation [1, 1, 0] -> ?: "
    # hidden_state, output = neural_network.think(array([1, 1, 0, 1]))
    # print output
    # Test the network with the test dataset
    differences = []
    for index, item in enumerate(inputs_test):
        hidden, output = neural_network.think(array(item))
        param = norm_coef.worldwide_gross
        print 'Prediction:', output*param, ' Real: ', outputs_test[index]*param, ' Diff:', (output - outputs_test[index])*param, abs(1)/output*param
        differences.append(abs((output - outputs_test[index])*param)/outputs_test[index]*param)
    print sum(differences)/float(len(differences))

plt.plot(graph_y)
# plt.show()

import numpy as np
from neuralnetwork import NeuralNetwork

# Training data set
training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))

# OR GATE
labels = np.array([1,0,0,0]) 

nn = NeuralNetwork(2)
nn.train(training_inputs, labels, True)
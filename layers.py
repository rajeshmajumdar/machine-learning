# Now we are ready to add more layers into our neural network

import numpy as np
from nnfs.datasets import spiral_data
import nnfs

import matplotlib.pyplot as plt


nnfs.init()

inputs = [[1, 2, 3, 2.5],
        [2., 5., -1., 2],
        [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# This is out first layer, with three neurons accepting 3 inputs vectors.
# next we wish to add one more layer with same nuber of neurons

weights2 = [[0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

# this here is our second layer with 3 neurons

#layer1_output = np.dot(inputs, np.array(weights).T) + biases

# this output of layer 1 becomes the input for our second layer

#layer2_out = np.dot(layer1_output, np.array(weights2).T) + biases2

#print(layer2_out)

#X, y = spiral_data(samples=100, classes=3)
#plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
#plt.show()

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = spiral_data(samples=100, classes=3)
dense1 = Dense(2, 3) # Here we have 2 input features and 3 neurons in this layer
dense1.forward(X)

print(dense1.output[:5])

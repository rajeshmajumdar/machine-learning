# Now we have the output from our dense layer, next we add activation function to our output, to get
# desired output, depending on the need we can choose different activation functions.
# The whole purpose of using activation function is to deal with non-linear problems. Most of the problems# we encounter in daily lives and we will be trying to solve would be non-linear problems.

# To understand this clearly, if we don't use any activation function, it's like we are simply taking y=x,# which on graph is simply a straight line, but if our solution depends on multiple variables, then using
# a straight line would give us nothing, so depending on the situation we use different activation 
# since it helps our neural network to deal with non-linearity.

# A great animation to understand this better: https://nnfs.io/mvp

import numpy as np
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt

nnfs.init()

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def relu(self, output):
        return np.maximum(0, output)

    def softmax(self, output):
        exp_values = np.exp(output - np.max(output, axis=1, keepdims=True))
        # Got unnormalised probabilities
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probs

    def cross_entropy_loss(self, output, targets):
        samples = len(output)

        # Now we need to clip the data to prevent division by 0
        # We can clip both sides to not drag mean towards any value

        y_pred_clipped = np.clip(output, 1e-7, 1-1e-7)

        # Probabilities for target values
        # only if categorical values

        if len(targets.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), targets]

        elif len(targets.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * targets, axis=1)
        negative_log_likehood = -np.log(correct_confidences)
        preds = np.argmax(output, axis=1)
        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)

        acc = np.mean(preds == targets)
        return np.mean(negative_log_likehood), acc


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# X, y = spiral_data(samples=100, classes=3)

# dense1 = Dense(2, 3)
# dense1.forward(X)
# out1 = dense1.relu(dense1.output)
# dense2 = Dense(3, 3)
# dense2.forward(out1)
# out2 = dense2.softmax(dense2.output)
# loss, acc = dense2.cross_entropy_loss(out2, y)
# print(out2[:5])
# print(f"Loss: {loss}")
# print(f"Accuracy: {acc}")

# now we have the probability distribution, as we can see it's almost ~33% for each, to get the value that# network chose we can use argmax on these outputs, which will check which calsses in the output distribution has the highest confidence and returns it's index. i.e. predicted class index.

### softmax activation function: we use this activation, mostly for classification problems, since it outputs the probability.


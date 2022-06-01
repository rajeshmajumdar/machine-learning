# Until now in our neural network, we have created some dense layers with some neurons connected to each other,
# and also we have implemented some cool tricks along the way for our need such as we used relu activation function and
# softmax activation function depending on our need, next we also implemented a function to calculate the loss and accuracy
# of our neural network.

# we have already created 45% of our neural network, next we need to figure out how much we need to change our weights and biases
# for each neuron such that we can minimize our loss and this process is known as optimization.

# one thing we can do is just choose random values and hope that someday by some luck we land the lucky number and everything
# aligns, but we can try to add some checks and try to make something out of it..

# what we also keep track of the accuracy and loss after every iteration and we select the best one and keep it until we find a
# new one.

from activation import Dense
from nnfs.datasets import spiral_data
import nnfs
import numpy as np

nnfs.init()

X, y = spiral_data(100, 3)

def random_optimizer():
    dense1 = Dense(2, 3)
    dense1.forward(X)
    out1 = dense1.relu(dense1.output)
    dense2 = Dense(3, 3)
    dense2.forward(out1)
    out2 = dense2.softmax(dense2.output)
    loss, acc = dense2.cross_entropy_loss(out2, y)

    lowest_loss = 9999999 # some initial loss value

    best_dense1_weights = dense1.weights
    best_dense1_biases = dense1.biases

    best_dense2_weights = dense2.weights
    best_dense2_biases = dense2.biases

    for iterations in range(10000):
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)

        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        dense1.forward(X)
        out1 = dense1.relu(dense1.output)
        dense2.forward(out1)
        out2 = dense2.softmax(dense2.output)
        loss, acc = dense2.cross_entropy_loss(out2, y)

        if loss < lowest_loss:
            print(f"New set of weights found at iterations: {iterations}, loss: {round(loss, 2)}, acc: {round(acc, 2)}")
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()

## After 10000 iteration best accuracy we got is 39%, so it is clear we can't just use the this way of optimizing our weights & biases
# and here comes one of my favorite topics in mathematics in use i.e. derivatives.




if __name__ == '__main__':
    random_optimizer()
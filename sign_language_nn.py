import numpy as np
import pandas as pd

## I am not gonna include the datasets here, you can download the dataset from kaggle
## Dataset Link: https://www.kaggle.com/datasets/datamunge/sign-language-mnist


class NeuralNetwork:
    def __init__(self, X, y, iterations, learning_rate):
        self.X = X
        self.y = y
        self.iterations = iterations
        self.learning_rate = learning_rate

    def _init_params(self):
        W1 = np.random.rand(26, 784) - 0.5
        b1 = np.random.rand(26, 1) - 0.5
        W2 = np.random.rand(26, 26) - 0.5
        b2 = np.random.rand(26, 1) - 0.5
        
        return W1, b1, W2, b2

    def _relu(self, Z):
        return np.maximum(Z, 0)

    def _softmax(self, Z):
        exp = np.exp(Z - np.max(Z))
        return exp / exp.sum(axis=0)

    def _forward_prop(self, W1, b1, W2, b2, X = None):
        var = X if X is not None else self.X
        Z1 = W1.dot(var) + b1
        A1 = self._relu(Z1)

        Z2 = W2.dot(A1) + b2
        A2 = self._softmax(Z2)

        return Z1, A1, Z2, A2

    def _one_hot(self):
        one_hot = np.zeros((self.y.size, self.y.max() + 2))
        one_hot[np.arange(self.y.size), self.y] = 1
        one_hot = one_hot.T
        return one_hot

    def _deriv_relu(self, Z):
        return Z > 0

    def _backward_prop(self, Z1, A1, Z2, A2, W2):
        m = self.y.size
        one_hot_y = self._one_hot()
        dZ2 = A2 - one_hot_y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, 1)

        dZ1 = W2.T.dot(dZ2) * self._deriv_relu(Z1)
        dW1 = 1 / m * dZ1.dot(self.X.T)
        db1 = 1 / m * np.sum(dZ1, 1)

        return dW1, db1, dW2, db2

    def _update_param(self, W1, b1, W2, b2, dW1, db1, dW2, db2):
        W1 -= self.learning_rate * dW1
        b1 -= self.learning_rate * np.reshape(db1, (26, 1))

        W2 -= self.learning_rate * dW2
        b2 -= self.learning_rate * np.reshape(db2, (26, 1))

        return W1, b1, W2, b2

    def get_preds(self, A):
        return np.argmax(A, 0)

    def get_acc(self, preds, y=None):
        var = y if y is not None else self.y
        return np.sum(preds == var) / var.size

    def _save_weights(self, W1, b1, W2, b2):
        with open('weights/weight1.npy', 'wb') as f:
            np.save(f, W1)
        with open('weights/weight2.npy', 'wb') as f:
            np.save(f, W2)
        with open('weights/bias1.npy', 'wb') as f:
            np.save(f, b1)
        with open('weights/bias2.npy', 'wb') as f:
            np.save(f, b2)

    def run(self):
        W1, b1, W2, b2 = self._init_params()
        for i in range(self.iterations):
            Z1, A1, Z2, A2 = self._forward_prop(W1, b1, W2, b2)
            dW1, db1, dW2, db2 = self._backward_prop(Z1, A1, Z2, A2, W2)
            W1, b1, W2, b2 = self._update_param(W1, b1, W2, b2, dW1, db1, dW2, db2)
            if i % 100 == 0:
                self._save_weights(W1, b1, W2, b2)

        return W1, b1, W2, b2

    def load_weights(self, W1_path, b1_path, W2_path, b2_path):
        with open(W1_path, 'rb') as f:
            W1 = np.load(f)
        with open(b1_path, 'rb') as f:
            b1 = np.load(f)
        with open(W2_path, 'rb') as f:
            W2 = np.load(f)
        with open(b2_path, 'rb') as f:
            b2 = np.load(f)

        return W1, b1, W2, b2

    def make_predictions(self, X, W1, b1, W2, b2):
        _, _, _, A2 = self._forward_prop(W1, b1, W2, b2, X)
        preds = self.get_preds(A2)
        return preds

    
if __name__ == '__main__':
    ## Since I have already trained the model on kaggle I am simply gonna load the weights and biases
    X = 'Some random value'
    y = 'some other random value'

    X_test = 'TEST_DATA_FEATURES'
    y_test = 'TEST_DATA_LABELS'

    weight1_path = 'sign_weights/weight1.npy'
    weight2_path = 'sign_weights/weight2.npy'
    bias1_path = 'sign_weights/bias1.npy'
    bias2_path = 'sign_weights/bias2.npy'
    
    nn = NeuralNetwork(X, y, 100, 0.1)
    W1, b1, W2, b2 = nn.load_weights(weight1_path, bias1_path, weight2_path, bias2_path)
    test_preds = nn.make_predictions(X_test, W1, b1, W2, b2)
    acc = nn.get_acc(test_preds, y_test)
    print(f"Accuracy: {acc}")
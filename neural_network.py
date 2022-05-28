import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, X, y, iterations, alpha):
        self.__X = X
        self.__y = y
        self.__iterations = iterations
        self.__alpha = alpha

    def _init_params(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5

        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5

        return W1, b1, W2, b2

    def _one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def _relu(self, Z):
        return np.maximum(Z, 0)

    def _softmax(self, Z):
        exp = np.exp(Z - np.max(Z))
        return exp / exp.sum(axis=0)

    def _deriv_relu(self, Z):
        return Z > 0

    def _forward_prop(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self._relu(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self._softmax(Z2)

        return Z1, A1, Z2, A2
   
    def _back_prop(self, Z1, A1, Z2, A2, W2, X, Y):
        m = Y.size
        one_hot_Y = self._one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, 1)

        dZ1 = W2.T.dot(dZ2) * self._deriv_relu(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, 1)

        return dW1, db1, dW2, db2

    def _update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2):
        W1 -= self.__alpha * dW1
        b1 -= self.__alpha * np.reshape(db1, (10, 1))
        W2 -= self.__alpha * dW2
        b2 -= self.__alpha * np.reshape(db2, (10, 1))

        return W1, b1, W2, b2

    def _get_pred(self, A2):
        return np.argmax(A2, 0)
    
    def get_acc(self, preds, Y):
        print(preds, Y)
        return np.sum(preds == Y) / Y.size

    def run(self):
        W1, b1, W2, b2 = self._init_params()
        for i in range(self.__iterations):
            Z1, A1, Z2, A2 = self._forward_prop(W1, b1, W2, b2, self.__X)
            dW1, db1, dW2, db2 = self._back_prop(Z1, A1, Z2, A2, W2, self.__X, self.__y)
            W1, b1, W2, b2 = self._update_params(W1, b1, W2, b2, dW1, db1, dW2, db2)
            if i % 50 == 0:
                print(f"Iterations: {i}")
                print(f"Accuracy: {self.get_acc(self._get_pred(A2), self.__y)}")

        return W1, b1, W2, b2

    def make_predictions(self, X, W1, b1, W2, b2):
        _, _, _, A2 = self._forward_prop(W1, b1, W2, b2, X)
        preds = self._get_pred(A2)
        return preds


def main():
    train_data = pd.read_csv('mnist/mnist_train.csv')
    train_data = np.array(train_data).T

    test_data = pd.read_csv('mnist/mnist_test.csv')
    test_data = np.array(test_data).T

    _, ntest = test_data.shape
    _, ntrain = train_data.shape
        
    y_test = test_data[0]
    X_test = test_data[1:ntest]
    X_test = X_test / 255.

    y_train = train_data[0]
    X_train = train_data[1: ntrain]
    X_train = X_train / 255.

    print("#### Training model...")
    nn = NeuralNetwork(X_train, y_train, 500, 0.10)
    W1, b1, W2, b2 = nn.run()

    print("### Feeding test data...")
    test_preds = nn.make_predictions(X_test, W1, b1, W2, b2)
    acc = nn.get_acc(test_preds, y_test)
    print(f"Accuracy on test data: {acc}")


if __name__ == '__main__':
    main()

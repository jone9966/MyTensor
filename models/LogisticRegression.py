import numpy as np
from losses import binary_crossentropy
from optimizers import sigmoid


class LogisticRegression:
    def __init__(self, name='logistic_regression'):
        self.name = name
        self.W = None
        self.b = None

    def initialize_parameters(self, X):
        W = np.random.randn(1, X.shape[1])
        b = np.zeros((1, 1))
        return W, b

    def parameters(self):
        return self.W, self.b

    def forward_propagation(self, X, W, b):
        Z = np.dot(W, X.T) + b
        A = sigmoid(Z)
        return A

    def backward_propagation(self, X, Y, A, W, b):
        dA = -Y / A + (1 - Y) / (1 - A)
        dZ = A * (1 - A) * dA
        dW = np.sum(X.T * dZ, axis=1, keepdims=True) / X.shape[0]
        db = np.sum(dZ, axis=1, keepdims=True) / X.shape[0]
        return dW, db

    def fit(self, X, Y, epochs=10, learning_rate=0.1):
        if self.W == None:
            self.W, self.b = self.initialize_parameters(X)

        for _ in range(epochs):
            A = self.forward_propagation(X, self.W, self.b)
            loss = binary_crossentropy(Y, A)
            acc = np.sum((A > 0.5) == Y) / Y.shape[1]
            print("Train_loss : ", round(loss, 4), "Train_acc : ", round(acc, 4))

            dW, db = self.backward_propagation(X, Y, A, self.W, self.b)
            self.W = self.W - learning_rate * dW.T
            self.b = self.b - learning_rate * db

    def predict(self, X):
        return self.forward_propagation(X, self.W, self.b)
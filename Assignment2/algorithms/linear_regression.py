"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        # self.w = (num_class, D) (num_classes, D + 1) (10, 3073)
        # X_train = (N, D) (N, D + 1) (5000, 3073)
        # y_train = (N, num_class) (5000, 10)
        y_train_one_hot = -1 * np.ones((len(y_train), self.n_class))
        y_train_one_hot[np.arange(len(y_train)), y_train] = 1

        for _ in range(self.epochs):
            gradient = (2 / N) * np.dot((np.dot(X_train, self.w.T)- y_train_one_hot).T, X_train) # (num_classes, D + 1)
            # w <- w - alpha(lambda W + del L(w))
            self.w -= self.lr * (self.weight_decay * self.w + gradient)

        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        
        output_num = np.dot(X_test, self.w.T) # (N, D+1).(D+1, n_class) => (N, n_class) with calculated value
        result = np.argmax(output_num, axis=1) # (1, N)
        return result

"""
K Nearest Neighbours Model
"""
import numpy as np


class KNN(object):
    def __init__(self, num_class: int):
        self.num_class = num_class

    def train(self, x_train: np.ndarray, y_train: np.ndarray, k: int):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(self, x_test: np.ndarray, k: int = None, loop_count: int = 1):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Fill this function in
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)

        # TODO: implement me
        # predicted_label : (N, )
        # print(distance)
        distance_mat_k = np.argpartition(distance, k_test, axis=1)[:, :5]
        # print(distance_mat_k)
        predicted_label_k = self._y_train[distance_mat_k] #(500, k)
        # print(predicted_label_k)
        predicted_label = np.apply_along_axis(lambda row: np.argmax(np.bincount(row)), axis=1, arr=predicted_label_k)
        # print(predicted_label)
        return predicted_label

    def calc_dis_one_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """

        # TODO: implement me
        
        # 각 train data 마다 train data 전체를 돌려서 voting 하는 것
        # return: (number of x_test, number of x_train) (500, 5000)

        num_x_test, dim_x_test = x_test.shape
        num_x_train, dim_x_train = self._x_train.shape
        distance_mat = np.zeros((num_x_test, num_x_train))

        for _ in range(num_x_test):
            distance = np.sqrt(np.sum((self._x_train - x_test[_, :]) ** 2, axis=1))
            distance_mat[_, :] = distance
        
        return distance_mat

    def calc_dis_two_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        # TODO: implement me
        num_x_test, dim_x_test = x_test.shape
        num_x_train, dim_x_train = self._x_train.shape
        distance_mat = np.zeros((num_x_test, num_x_train))

        for i in range(num_x_test):
            for j in range(num_x_train):
                distance = np.sqrt(np.sum((self._x_train[j, :] - x_test[i, :]) ** 2))
                distance_mat[i, j] = distance

        return distance_mat
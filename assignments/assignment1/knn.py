import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def get_manhatten_distance(firstVector, secondVector): 
        return np.sum(firstVector - secondVector)
        
    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # Fill dists[i_test][i_train]
                dists[i_test][i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]))
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # [DONE]: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            dists[i_test] = np.sum(np.abs(X[i_test] - self.train_X), axis=1) 
            # TODO: explore why swap matrix and vector is possible 
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # [DONE]: Implement computing all distances with no loops!
        dists = np.sum(np.abs(X[:, None] - self.train_X), axis=2)
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        # classes of features are defined in train_Y massive, 
        # classes is the item under index which is index of minimal weight from train_X
        # the intent is to get massive with size of K filled with minimal weights from train set  
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        weights_ids = self.__calc_weights_indexes(dists, num_test)
        print(dists.shape)
        print(weights_ids.shape)
        print(self.train_y.shape)
        for sample in range(weights_ids.shape[0]): 
            frequency_zero_label = 0
            frequency_nine_label = 0
            for j in range(weights_ids[sample].size): 
                idx = weights_ids[sample][j]
                category_zero_nine = self.train_y[idx] 
                if (category_zero_nine == True):
                    frequency_zero_label += 1
                elif (category_zero_nine == False):
                    frequency_nine_label += 1
                is_last = (j + 1) >= weights_ids[sample].size
                if (is_last):
                    pred[sample] = True if (frequency_zero_label > frequency_nine_label) else False
        
#         print(weights_ids)
        
        return pred
    
    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        print("Trigger predict_labels_multiclass()")
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        weights_ids = self.__calc_weights_indexes(dists, num_test)
        for i in range(num_test):
            # DONE: Implement choosing best class based on k
            # nearest training samples
            belong_to_class = 0
            do_not_belong_to_class = 0
            observation = {}
            for j in range(weights_ids[i].size):
                idx = weights_ids[i][j]
                label = self.train_y[idx]
                if (label in observation): 
                    observation[label] += 1  
                else: 
                    observation[label] = 1
            the_most_frequent_label = -1
            frequency = -1
            for k,v in observation.items():
                if (frequency < v):
                    frequency = v
                    the_most_frequent_label = k
                    
            pred[i] = the_most_frequent_label
            
        return pred

    def __calc_weights_indexes(self, dists, num_test): 
        weights_ids = np.zeros((num_test,self.k), dtype=int)
        dists_copy = np.copy(dists)
        dists_copy_sorted = np.sort(dists, axis=1)
        for i in range(num_test):
            # DONE: Implement choosing best class based on k
            # nearest training samples
            k_neighbors_weights = dists_copy_sorted[i][:self.k] 
            for j in range(k_neighbors_weights.size):
                indexes = np.where(dists_copy[i] == k_neighbors_weights[j])[0]
                if (indexes.size != 0):
                    weight_index = indexes[0]
                    dists_copy[i][weight_index] = -1.
                    available_position = np.where(weights_ids[i] == 0)[0]
                    weights_ids[i][available_position] = weight_index 
        return weights_ids
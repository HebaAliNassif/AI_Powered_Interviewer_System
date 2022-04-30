import sys
import time

import math
import numpy as np

class DecisionStumpClassifier:

    def __init__(self, steps=400, max_parallel_processes=1):
        self.steps = steps
        self.max_parallel_processes = max_parallel_processes
        self.best_feature_index = -1
        self.best_left_direction = 1
        self.best_position = -1
        self.features_count = -1
    
    def train(self, X_, y_, W_, verbose=False):

        X = X_ if type(X_) == np.ndarray else np.array(X_)
        y = y_ if type(y_) == np.ndarray else np.array(y_)
        W = W_ if type(W_) == np.ndarray else np.array(W_)
        
        assert len(X) == len(y)
        assert len(X) == len(W)
        
        n_samples, n_features = X.shape
        
        self.features_count = n_features
        
        best_feature_index = 0
        best_left_direction = 1
        best_position = 0
        minerr = W.sum()
        
        # For each featurue, train a classifier then select the feature with the minimum error
        for i in range(n_features):
            
            err, d, p = self.tain_feature(X[:, i], y, W, self.steps)
            
            if err < minerr:
                minerr = err
                best_feature_index = i
                best_left_direction = d
                best_position = p
            
            if verbose:
                finished = i / n_features
                print('% 7.1f%%' % (finished*100), end='')
                print('\r', end='', flush=True)
                
        self.best_feature_index = best_feature_index
        self.best_left_direction = best_left_direction
        self.best_position = best_position

        return minerr

    def tain_feature(self, X, y, W, steps):

        X = X.flatten()

        min_x, max_x = X.min(), X.max()
        len_x = max_x - min_x
        
        best_left_direction = 1
        best_position = min_x
        minerr = W.sum()

        if len_x > 0.0:
            for p in np.arange(min_x, max_x, len_x/steps):
                for d in [-1, 1]:
                    gy = np.ones((len(y)))
                    gy[X*d < p*d] = -1
                    err = np.sum((gy != y) * W)
                    if err < minerr:
                        minerr = err
                        best_left_direction = d
                        best_position = p

        return minerr, best_left_direction, best_position

    def predict(self, test_set_):

        test_set = test_set_ if type(test_set_) == np.ndarray else np.array(test_set_)
        
        n_samples, n_features = test_set.shape

        assert n_features == self.features_count

        single_feature = test_set[:, self.best_feature_index]
        h = np.ones((n_samples))
        h[single_feature * self.best_left_direction < self.best_position * self.best_left_direction] = -1
        return h
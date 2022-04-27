import copy
import numpy as np
from decision_stump import DecisionStumpClassifier


class AdaboostClassifier:
    def __init__(self, weak_classifier = DecisionStumpClassifier()):
        self.features_count =0
        
        self.num_weak_clfs = 0
        self.tot_num_weak_clfs = 0
        self.weak_clfs = []
        self._weak_clf_class = weak_classifier
        
        self.alpha = np.array([])
    
    def train(self, X_, y_, n_classes, is_continue=False, verbose=False):
    
        assert len(X_) == len(y_)
        
        X = X_ if type(X_) == np.ndarray else np.array(X_)
        y = np.array(y_).flatten()
        
        y[y == 0] = -1
        n_samples, n_features = X.shape
        
        if not is_continue or self.num_weak_clfs == 0:
            self.num_weak_clfs = 0
            self.tot_num_weak_clfs = n_classes
            
            self.weak_clfs = [copy.deepcopy(self._weak_clf_class) for clf in range(self.tot_num_weak_clfs)]
            self.alpha = np.zeros((self.tot_num_weak_clfs))
            
            self.features_count = n_features
            self.sum_eval = 0
            
            W = np.ones((n_samples)) / n_samples

        else:
            self.tot_num_weak_clfs = self.num_weak_clfs + n_classes
            self.weak_clfs = np.concatenate((self.weak_clfs[0:self.num_weak_clfs], [copy.deepcopy(self._weak_clf_class) for clf in range(n_classes)]))
            self.alpha = np.concatenate((self.alpha[0:self.num_weak_clfs], np.zeros((n_classes))))
            
            W = self.W
            
        for clf_index in range(self.num_weak_clfs, self.tot_num_weak_clfs):
            if verbose:
                print('Training %d-th weak classifier from total %d weak classifier' % (clf_index, self.tot_num_weak_clfs))
                
            # Train the weak classifier
            error = self.weak_clfs[clf_index].train(X, y, W, verbose)
            
            if error == 0:
                error = 1e-6
            
            elif error == 1:
                error = 1 - 1e-6
                
            # Calculate alpha
            self.alpha[clf_index] = 0.5 * np.log((1 - error) / error)
            
            # Update the weights
            y_predict = self.weak_clfs[clf_index].predict(X).flatten()
            
            W = W * np.exp(-self.alpha[clf_index] * y * y_predict)
            
            # Normlize the weights
            W = W / W.sum()
            
            self.num_weak_clfs = clf_index + 1
            
            if verbose:
                print('%d-th weak classifier: error = %f' % (clf_index, error))
            
            if self.evaluateModel(clf_index, y_predict, y) == 0:
                print(self.num_weak_clfs, "weak classifier are enought to make error rate reach 0.0")
                break

        self.W = W
        
    def evaluateModel(self, t, y_predict, y):
        print("evaluateModel")
        print(self.sum_eval, t, y_predict, y)
        self.sum_eval = self.sum_eval + y_predict * self.alpha[t]
        print(self.sum_eval)
        yPred = np.sign(self.sum_eval)
        return np.sum(yPred != y)

    def predict(self, test_set_):
        
        hsum = self.weightedSum(test_set_)
        confidence = abs(hsum) / np.sum(abs(self.alpha))

        yPred = np.sign(hsum)
        yPred[yPred == -1] = 0

        return yPred, confidence
        
    def weightedSum(self, test_set_):
    
        test_set = test_set_ if type(test_set_) == np.ndarray else np.array(test_set_)

        assert test_set.shape[1] == self.features_count

        hsum = 0
        for i in range(self.num_weak_clfs):
            hsum += self.alpha[i] * self.weak_clfs[i].predict(test_set)

        return hsum

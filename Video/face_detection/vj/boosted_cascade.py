import os
import copy
import math
import time
import sys
import multiprocessing as mp
import numpy as np
from sklearn.utils import shuffle as skshuffle

from haar_like_feature import haar_like_feature, haar_like_feature_coord
from utils import integral_image

from adaboost import AdaboostClassifier
from decision_stump import DecisionStumpClassifier

class BoostedCascade:
    SCClass = AdaboostClassifier(weak_classifier = DecisionStumpClassifier(100))
    
    def __init__(self, false_positive_rate_target, f, d, validset_rate = 0.3, CIsteps = 0.01):

        self.false_positive_rate_target = false_positive_rate_target

        self.false_positive_rate_min_layer = f

        self.detection_rate_min_layer = d

        self.validset_rate = validset_rate
        self.CIsteps = CIsteps

        self.detectWndW, self.detectWndH = (-1, -1)
        
        self.P = np.array([])
        self.N = np.array([])
        self.validX = np.array([])
        self.validy = np.array([])
        self.P_test = np.array([])
        self.N_test = np.array([])
        
        self.features_coords = np.array([])
        self.features_types = np.array([])

        self.SCs = []
        self.thresholds = []
        self.SCn = []

    def getDetectWnd(self):
        return (self.detectWndH, self.detectWndW)

    def architecture(self):
        archi=""
        for ind, SC in enumerate(self.SCs):
            archi = archi + ("layer %d: %d weak classifier\n" % (ind, SC.nWC))
        return archi

    def __str__(self):
        return ("BoostedCascade(Ftarget=%f, "
                "f=%f, "
                "d=%f, "
                "validset_rate=%f, "
                "CIsteps=%f, "
                "detectWnd=%s, "
                "features_cnt=%d, "
                "n_strong_classifier=%d)") % (
                self.Ftarget,
                self.f,
                self.d,
                self.validset_rate,
                self.CIsteps,
                (self.detectWndH, self.detectWndW),
                self.features_cnt,
                len(self.SCs))
                
    def extract_feature_image(self, image):
        ii = integral_image(image)
        x = haar_like_feature(ii, height=self.detectWndH, width=self.detectWndW, features= self.features_coords)
        return x
        
    def parallelExtractFeatures(self, tid, range_, raw_data, result_output, schedule_output):
        assert type(range_) == tuple
        
        for n in range(range_[0], range_[1]):
            schedule_output.value = (n - range_[0]) / (range_[1] - range_[0])
            ii = integral_image(raw_data[n])
            x = haar_like_feature(ii, height=self.detectWndH, width=self.detectWndW, features= self.features_coords)
            result_output.put((n, x))

    def extractFeatures(self, raw_data, verbose=False, max_parallel_process=8):
        n_samples, height, width = np.shape(raw_data)

        assert height == self.detectWndH and width == self.detectWndW, "Height and width mismatch with current data."

        processes = [None] * max_parallel_process
        schedules = [None] * max_parallel_process
        results = mp.Queue()

        blocksize = math.ceil(n_samples / max_parallel_process)
        if blocksize <= 0: blocksize = 1
        for tid in range(max_parallel_process):
            schedules[tid] = mp.Value('f', 0.0)
            blockbegin = blocksize * tid
            if blockbegin >= n_samples: break
            blockend = blocksize * (tid+1)
            if blockend > n_samples: blockend = n_samples
            processes[tid] = mp.Process(target=self.parallelExtractFeatures, args=(tid, (blockbegin, blockend), raw_data, results, schedules[tid]))
            processes[tid].start()

        X = np.zeros((n_samples, len(self.features_coords)))
        while True:
            alive_processes = [None] * max_parallel_process
            for tid in range(max_parallel_process):
                alive_processes[tid] = processes[tid].is_alive()
            if sum(alive_processes) == 0:
                break

            while not results.empty():
                ind, x = results.get()
                X[ind] = x

            if verbose:
                for tid in range(max_parallel_process):
                    schedule = schedules[tid].value
                    print('% 7.1f%%' % (schedule * 100), end='')
                print('\r', end='', flush=True)

            time.sleep(0.2)
        sys.stdout.write("\033[K")
        return X

    def prepare(self, P_, N_, shuffle=True, verbose=False, max_parallel_process=8):
        assert np.shape(P_)[1:3] == np.shape(N_)[1:3], "Window sizes mismatch."
        _, self.detectWndH, self.detectWndW = np.shape(P_)#
        
        if shuffle:
            P_ = skshuffle(np.array(P_), random_state=1)
            N_ = skshuffle(np.array(N_), random_state=1)
        
        self.features_coords, self.features_types = haar_like_feature_coord(self.detectWndH, self.detectWndW)
        
        
        if verbose:
            print('Preparing positive data.')
        P = self.extractFeatures(P_, verbose=verbose, max_parallel_process=max_parallel_process)
        
        if verbose:
            print('Preparing negative data.')
        N = self.extractFeatures(N_, verbose=verbose, max_parallel_process=max_parallel_process)

        divideP = int(len(P)*self.validset_rate)
        divideN = int(len(N)*self.validset_rate)

        validX = np.concatenate((P[0:divideP], N[0:divideN]))
        validY = np.concatenate((np.ones(len(P[0:divideP])), np.zeros(len(N[0:divideN]))))

        P = P[divideP:len(P)]
        N = N[divideN:len(N)]

        self.P = P
        self.N = N
        self.validX = validX
        self.validy = validY
        
    def saveFeaturesData(self, filename):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        datatype = type(self.P[0][0])
        
        np.save(filename+'-variables', np.array([self.detectWndH, self.detectWndW, self.P.shape, self.N.shape, self.validX.shape, self.validy.shape, datatype], dtype=object))
        np.save(filename+'-features_coords', self.features_coords)
        np.save(filename+'-features_types', self.features_types)
        
        P = np.memmap(filename + '-P', mode='w+', dtype=datatype, shape=self.P.shape)
        P[:] = self.P[:]
        P.flush()
        
        N = np.memmap(filename + '-N', mode='w+', dtype=datatype, shape=self.N.shape)
        N[:] = self.N[:]
        N.flush()
        
        validX = np.memmap(filename + '-validX', mode='w+', dtype=datatype, shape=self.validX.shape)
        validX[:] = self.validX[:]
        validX.flush()
        
        validy = np.memmap(filename + '-validy', mode='w+', dtype=datatype, shape=self.validy.shape)
        validy[:] = self.validy[:]
        validy.flush()

    def loadFeaturesData(self, filename):
        detectWndH, detectWndW, shape1, shape2, shape3, shape4, datatype = np.load(filename + '-variables.npy', allow_pickle=True)
        self.detectWndH, self.detectWndW = int(detectWndH), int(detectWndW)
        self.features_coords = np.load(filename + '-features_coords.npy', allow_pickle=True)
        self.features_types = np.load(filename +'-features_types.npy', allow_pickle=True)
        self.P = np.memmap(filename + '-P', mode='r', dtype=datatype, shape=shape1)
        self.N = np.memmap(filename + '-N', mode='r', dtype=datatype, shape=shape2)
        self.validX = np.memmap(filename + '-validX', mode='r', dtype=datatype, shape=shape3)
        self.validy = np.memmap(filename + '-validy', mode='r', dtype=datatype, shape=shape4)
        
    def train(self, is_continue=False, autosnap_filename=None, verbose=False):
        assert self.detectWndW != -1 and self.detectWndH != -1 and len(self.P) != 0 and len(self.N) != 0 and len(self.features_types) != 0 and len(self.features_coords) != 0, "Please call prepare first."
        
        P = self.P
        N = self.N
        
        self._initEvaluate(self.validX, self.validy)

        self.P = np.array([])
        self.N = np.array([])
        
        self.validX = np.array([])
        self.validy = np.array([])

        false_positive_rate = 1.0
        detection_rate = 1.0

        if not is_continue:
            self.SCs = []
            self.thresholds = []
            self.SCn = []
        else:
            yPred = self._predictRaw(N)
            N = N[yPred == 1]

            for ind in range(len(self.SCs)):
                ySync, false_positive_rate, detection_rate, _ = self._evaluate(ind)
                self._updateEvaluate(ySync)

        features_used = len(self.features_coords)
        n_step = 1

        print('Begin training, with n_classes += %d, n_step = %d, false positive rate target = %f, false positive rate per layer = %f, detection rate per layer = %f' % (1, n_step, self.false_positive_rate_target, self.false_positive_rate_min_layer, self.detection_rate_min_layer))

        itr = 0
        while false_positive_rate > self.false_positive_rate_target:
            
            itr += 1
            aim_false_positive_rate = false_positive_rate
            
            aim_detection_rate = detection_rate
            n = 0

            print('Training iteration %d, false positive rate = %f' % (itr, false_positive_rate))

            X_train = np.concatenate((P, N))
            y_train = np.concatenate((np.ones(len(P)), np.zeros(len(N))))
            X_train, y_train = skshuffle(X_train, y_train, random_state=1)

            classifier = copy.deepcopy(self.SCClass)

            self.SCs.append(copy.deepcopy(classifier))
            self.thresholds.append(1.0)
            self.SCn.append(features_used)

            while false_positive_rate > self.false_positive_rate_min_layer * aim_false_positive_rate:
                n = n_step
                
                ind = len(self.SCs) - 1
                
                print('Itr-%d: Training %d-th AdaBoostClassifier, features count + %d, detection rate = %f, false positive rate = %f'
                    % (itr, ind, n, detection_rate, false_positive_rate))
                if verbose:
                    print('Aim detection rate : >=%f; Aim false positive rate : <=%f'
                        % (self.detection_rate_min_layer * aim_detection_rate, self.false_positive_rate_min_layer * aim_false_positive_rate))
                    print('Positive samples : %s; Negative samples : %s'
                        % (str(P.shape), str(N.shape)))

                classifier.train(X_train[:, 0:features_used], y_train, n, is_continue=True, verbose=verbose)
                if verbose:
                    for i in range(classifier.num_weak_clfs):
                        print('%d-th weak classifier select %s as its feature.'
                            % (i, str(self.features_coords[classifier.weak_clfs[i].best_feature_index])))
                
                self.SCs[ind] = copy.deepcopy(classifier)
                self.thresholds[ind] = 1.0
                self.SCn[ind] = features_used
                
                ySync, false_positive_rate, detection_rate, _ = self._evaluate(ind)
                
                print('Threshold adjusted to %f, detection rate = %f, false positive rate = %f'
                    % (self.thresholds[ind], detection_rate, false_positive_rate))
                
                while detection_rate < self.detection_rate_min_layer * aim_detection_rate:
                    self.thresholds[ind] -= self.CIsteps
                    
                    if self.thresholds[ind] < -1.0:
                        self.thresholds[ind] = -1.0
                        
                    ySync, false_positive_rate, detection_rate, _ = self._evaluate(ind)
                    
                    print('Threshold adjusted to %f, detection rate = %f, false positive rate = %f' % (self.thresholds[ind], detection_rate, false_positive_rate))
                
            self._updateEvaluate(ySync)

            if false_positive_rate > self.false_positive_rate_target:
                yPred = self._predictRaw(N)
                N = N[yPred == 1]

            if autosnap_filename:
                self.saveModel(autosnap_filename)

        print('%d cascaded classifiers, detection rate = %f, false positive rate = %f'% (len(self.SCs), detection_rate, false_positive_rate))

    def _initEvaluate(self, validset_X, validset_y):
        """Initialize before evaluating the model.

        Parameters
        ----------
        validset_X : np.array of shape = [n_samples, n_features]
            The samples of the valid set.
        validset_y : np.array of shape = [n_samples]
            The ground truth of the valid set.
        """
        class Eval: pass
        self._eval = Eval()
        self._eval.validset_X = validset_X
        self._eval.validset_y = validset_y
        self._eval.PySelector = (validset_y == 1)
        self._eval.NySelector = (validset_y == 0)
        self._eval.cP = len(validset_y[self._eval.PySelector])
        self._eval.cN = len(validset_y[self._eval.NySelector])
        self._eval.ySync = np.ones(len(validset_y)) # All exist possible positive
        pass

    def _evaluate(self, ind):
        """Evaluate the model, but won't update any parameter of the model.

        Parameters
        ----------
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the valid set.
        y_ : np.array of shape = [n_samples]
            The ground truth of the valid set.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted result.
        f : float
            The false positive rate.
        d : float
            The detection rate.
        fp : np.array
            The false positives.
        """
        X_ = self._eval.validset_X
        y_ = self._eval.validset_y

        ySync = self._eval.ySync.copy()

        yiPred, CI = self.SCs[ind].predict(X_[:, 0:self.SCn[ind]])
        CI[yiPred != 1] = -CI[yiPred != 1]
        
        # Reject those whose confidences are less that thresholds
        yiPred = (CI >= self.thresholds[ind]).astype(int)
        ySync[ySync == 1] = yiPred # Exclude all rejected

        fp = (ySync[self._eval.NySelector] == 1)
        dp = (ySync[self._eval.PySelector] == 1)
        f = (np.sum(fp) / self._eval.cN) if self._eval.cN != 0.0 else 0.0
        d = (np.sum(dp) / self._eval.cP) if self._eval.cP != 0.0 else 0.0

        return ySync, f, d, fp

    def _updateEvaluate(self, ySync):
        """Update the parameter of the evaluating model.

        Parameters
        ----------
        ySync : np.array of shape = [n_samples]
            The classifier result generated by function 'evaluate'.
        """
        self._eval.validset_X = self._eval.validset_X[ySync[self._eval.ySync == 1] == 1]
        self._eval.ySync = ySync # Update ySync
        
    def _predictRaw(self, test_set_):
        
        yPred = np.ones(len(test_set_))
        
        for ind in range(len(self.SCs)):
            yiPred, CI = self.SCs[ind].predict(test_set_[yPred == 1][:, 0:self.SCn[ind]])
            
            CI[yiPred != 1] = -CI[yiPred != 1]
            
            yiPred = (CI >= self.thresholds[ind]).astype(int)

            yPred[yPred == 1] = yiPred # Exclude all rejected
        
        return yPred
    
    def _weakPredict(self, wcself, X_):
        """Predict function for weak classifiers.

        Parameters
        ----------
        wcself : instance of WeakClassifier
            The weak classifier.
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted result of the testing samples.
        """

        coord = self.features_coords[wcself.best_feature_index]
        
        feature = np.zeros(len(X_))
        for ind in range(len(X_)):
            feature[ind] = 0
            for rect in coord:
                feature[ind] += rect.compute_feature(X_[ind])
            
        h = np.ones(len(X_))
        h[feature * wcself.best_left_direction < wcself.best_position * wcself.best_left_direction] = -1
        return h

    def _strongPredict(self, scself, X_):
        """Predict function for the strong classifier (AdaBoostClassifier).

        Parameters
        ----------
        scself : instance of self.SCClass
            The strong classifier (AdaBoostClassifier).
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted results of the testing samples.
        CI : np.array of shape = [n_samples]
            The confidence of each predict result.
        """
        hsum = 0
        for i in range(scself.num_weak_clfs):
            hsum = hsum + scself.alpha[i] * self._weakPredict(scself.weak_clfs[i], X_)

        yPred = np.sign(hsum)
        yPred[yPred == -1] = 0
        CI = abs(hsum) / np.sum(scself.alpha)

        return yPred, CI

    def predictIntegralImage(self, test_set_integral_images):
        X = test_set_integral_images
        yPred = np.ones(len(X))
        for ind in range(len(self.SCs)):
            yiPred, CI = self._strongPredict(self.SCs[ind], X[yPred == 1])
            CI[yiPred != 1] = -CI[yiPred != 1]
            yiPred = (CI >= self.thresholds[ind]).astype(int)
            # yiPred[yiPred == 1] = (CI[yiPred == 1] >= self.thresholds[ind]).astype(int)
            yPred[yPred == 1] = yiPred # Exclude all rejected
        
        return yPred

    def predict(self, test_set_):
        """Predict whether it's a face or not.

        Parameters
        ----------
        test_set_ : array-like of shape = [n_samples, height, width]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted results of the testing samples.
        """
        X = np.zeros((len(test_set_), self.detectWndH+1, self.detectWndW+1))
        for i in range(len(test_set_)):
            X[i] = utils.integral_image(test_set_[i])
        return self.predictIntegralImage(X)
    
    def predictRaw(self):
        X = np.concatenate((self.P, self.N))
        yPred = self._predictRaw(X)
        return yPred[0:len(self.P)], yPred[len(self.P):len(X)]
        
    def saveModel(self, filename):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        np.save(filename+'-variables', [
            self.false_positive_rate_target, self.false_positive_rate_min_layer, self.detection_rate_min_layer, self.validset_rate, self.CIsteps,
            self.detectWndH, self.detectWndW])
        np.save(filename+'-features_coords', self.features_coords)
        np.save(filename+'-features_types', self.features_types)
        np.save(filename+'-thresholds', self.thresholds)
        np.save(filename+'-SCs', self.SCs)
        np.save(filename+'-SCn', self.SCn)

    def loadModel(filename):
        Ftarget, f, d, validset_rate, CIsteps, \
        detectWndH, detectWndW   \
            = np.load(filename+'-variables.npy')
        model = BoostedCascade(Ftarget, f, d, validset_rate, CIsteps)
        model.detectWndH, model.detectWndW = int(detectWndH), int(detectWndW)

        model.features_coords = np.load(filename+'-features_coords.npy', allow_pickle=True)
        model.features_types = np.load(filename+'-features_types.npy', allow_pickle=True)
        model.thresholds = list(np.load(filename+'-thresholds.npy', allow_pickle=True))
        model.SCs = list(np.load(filename+'-SCs.npy',allow_pickle=True))
        model.SCn = list(np.load(filename+'-SCn.npy',allow_pickle=True))
        return model
        
    def prepareTest(self, P_, N_, verbose=False):
        
        if verbose:
            print('Preparing positive data')
        P = self.extractFeatures(P_, verbose=verbose)
        
        if verbose:
            print('Preparing negative data')
        N = self.extractFeatures(N_, verbose=verbose)
        
        self.P_test = P
        self.N_test = N
        
    def saveTestFeaturesData(self, filename):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        datatype = type(self.P_test[0][0])
        np.save(filename+'-test_variables', np.array([self.P_test.shape, self.N_test.shape, datatype], dtype=object))

        P_test = np.memmap(filename + '-P_test', mode='w+', dtype=datatype, shape=self.P_test.shape)
        P_test[:] = self.P_test[:]
        P_test.flush()
        
        N_test = np.memmap(filename + '-N_test', mode='w+', dtype=datatype, shape=self.N_test.shape)
        N_test[:] = self.N_test[:]
        N_test.flush()

    def loadTestFeaturesData(self, filename):
        shape1, shape2, datatype = np.load(filename + '-test_variables.npy', allow_pickle=True)
        
        self.P_test = np.memmap(filename + '-P_test', mode='r', dtype=datatype, shape=shape1)
        self.N_test = np.memmap(filename + '-N_test', mode='r', dtype=datatype, shape=shape2)
import os
import math
import pickle
import numpy as np
import cv2

from sklearn.svm import LinearSVC
from sklearn.model_selection  import GridSearchCV

class HogClassifier:
    def __init__(self, detectWndW, detectWndH):
        self.detectWndW = detectWndW
        self.detectWndH = detectWndH
        self.clf = None
        self.cell = [8,8]
        self.incr = [8,8]
        
    def normalize(self, vector):
        prevType = vector.dtype
        if (vector.dtype == np.float16):
            vector = vector.astype(np.float32)
        norm = np.linalg.norm(vector)
        if (norm != 0 and np.isfinite(norm)):
            vector /= norm
        return vector.astype(prevType)
        
    def calculate_histogram(self, array,weights):
        bins_range = (0, 180)
        bins = 8
        hist,_ = np.histogram(array,bins=bins,range=bins_range,weights=weights)
        
        return hist  
        
    def create_hog_features(self, grad_array, mag_array):
        max_h = int(((grad_array.shape[0]-self.cell[0])/self.incr[0])+1)
        max_w = int(((grad_array.shape[1]-self.cell[1])/self.incr[1])+1)
        cell_array = []
        
        w = 0
        h = 0
        i = 0
        j = 0
        
        #Creating 8X8 cells
        while i<max_h:
            w = 0
            j = 0
        
            while j<max_w:
                for_hist = grad_array[h:h+self.cell[0],w:w+self.cell[1]]
                for_wght = mag_array[h:h+self.cell[0],w:w+self.cell[1]]
                
                val = self.calculate_histogram(for_hist,for_wght)
                cell_array.append(val)
                j += 1
                w += self.incr[1]
        
            i += 1
            h += self.incr[0]
        
        cell_array = np.reshape(cell_array,(max_h, max_w, 8))
        #normalising blocks of cells
        block = [2,2]
        #here increment is 1
        
        max_h = int((max_h-block[0])+1)
        max_w = int((max_w-block[1])+1)
        block_list = []
        w = 0
        h = 0
        i = 0
        j = 0
        
        while i<max_h:
            w = 0
            j = 0
            while j < max_w:
                for_norm = cell_array[h:h+block[0],w:w+block[1]]
                #mag = np.linalg.norm(for_norm)
                arr_list=self.normalize(for_norm)
                #arr_list = (for_norm/mag).flatten().tolist()
                block_list += arr_list.flatten().tolist()
                
                j += 1
                w += 1
            i += 1
            h += 1
        return block_list
        
    def extractManualHogFeature(self, img):
        #Calculating Gradients (direction x and y)
        gX= cv2.Sobel(img,0, dx=1,dy=0, ksize=1)
        #gX_img= np.uint8(np.absolute(gX))
        gY= cv2.Sobel(img,0, dx=0,dy=1, ksize=1)
        #gY_img = np.uint8(np.absolute(gY))
        magnitude = np.sqrt((gX ** 2) + (gY ** 2))
        orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
        hog_features = self.create_hog_features(orientation,magnitude)
        hog_features = np.asarray(hog_features,dtype=float)
        hog_features = np.expand_dims(hog_features,axis=0)
        return hog_features.flatten()
        
    def train(self, X_train, y_train):
        grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
        grid.fit(X_train, y_train)
        self.clf = grid.best_estimator_
        self.clf.fit(X_train, y_train)
        
    def predict(self, test_set):
        return self.clf.predict([self.extractManualHogFeature(img) for img in test_set])
        
    def saveModel(self, filename):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        np.save(filename+'-variables', [self.detectWndH, self.detectWndW])
        with open(filename + '-hog.pkl', 'wb') as f:
            pickle.dump(self.clf, f) 

    def loadModel(filename):
        detectWndH, detectWndW = np.load(filename+'-variables.npy')
        model = HogClassifier(int(detectWndW), int(detectWndH))
        with open(filename + '-hog.pkl', 'rb') as f:
            model.clf = pickle.load(f) 
        return model
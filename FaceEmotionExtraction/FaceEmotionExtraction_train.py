import math

#import imutils
import numpy as np
import cv2 as cv2
import glob
import csv
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
import pickle

X = []
y = []
filename = 'AffectNet_5000_svcOnly_rpf_model.sav'        

def getImage(path):
    return cv2.imread(path)
def read(imagesPath, label):

    for filename in glob.glob(imagesPath + '/*.*'):
        # print(filename.split(imagesPath + '/')[1])
        # print(filename)
        win_size = (64, 128)
        img = getImage(filename)

        win_size = (64, 128)

        img = cv2.resize(img, win_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        d = cv2.HOGDescriptor()
        hog = d.compute(img)
        X.append(hog.transpose()[0])
        y.append(label)


read('happy_',0)
read('contempt_',1)
read('anger_',2)
read('disgust_',3)
read('fear_',4)
read('sadness_',5)
read('surprise_',6)
read('neutral_',7)

y = np.asarray(y)
X = np.asarray(X)

clf = SVC(kernel='rbf', probability=True, tol=1e-3)
clf.fit(X, y)
pickle.dump(clf, open(filename, 'wb'))

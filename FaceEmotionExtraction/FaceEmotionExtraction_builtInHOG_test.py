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

filename = 'AffectNet_5000_svcOnly_rpf_model.sav'        
clf = pickle.load(open(filename, 'rb'))
True_sum=0

def getImage(path):
    return cv2.imread(path)

def test(imagesPath, label):

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
        hog = hog.transpose()[0]
        hog = np.asarray(hog)
        predicted=int(clf.predict([hog]))
        if label == predicted:
            global True_sum
            True_sum=True_sum+1
            #print("true")
        #else:
            #print("false")    

classes = ["HAPPY", "CONTEMPT", "ANGER", "DISGUST", "FEAR", "SADNESS", "SURPRISE", "NEUTRAL"]

####################################testing on testing set########################################
#test('test_happy',0)
#test('test_contempt',1)
#test('test_anger',2)
#test('test_disgust',3)
#test('test_fear',4)
#test('test_sadness',5)
#test('test_surprise',6)
#test('test_neutral',7)
#print("accuracy: "+str((True_sum/4000)*100))
####################################testing on a specific image###################################
img = getImage('aya_angry2.jpg')
win_size = (64, 128)
img = cv2.resize(img, win_size)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
d = cv2.HOGDescriptor()
hog = d.compute(img)
hog = hog.transpose()[0]
hog = np.asarray(hog)
print(classes[int(clf.predict([hog]))])

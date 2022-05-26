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

filename = 'AffectNet_700_svcOnly_rpf_manualFeatExt_model.sav'        
clf = pickle.load(open(filename, 'rb'))
True_sum=0
cell=[8,8]
incr=[8,8]

def getImage(path):
    return cv2.imread(path)
def normalize(V):
    V_Type = V.dtype
    if (V.dtype == np.float16):
        V = V.astype(np.float32)
    V_Norm = np.linalg.norm(V)
    if (V_Norm != 0 and np.isfinite(V_Norm)):
        V /= V_Norm
    return V.astype(V_Type)	
def calculate_histogram(values,weights):
	hist,_ = np.histogram(values,bins=8,range=(0, 180),weights=weights)
	return hist  
	
def create_hog_features(gradient_values,magnitude_values):
	w = 0
	h = 0
	i = 0
	j = 0
	cell_values = []
	max_height = int(((gradient_values.shape[0]-cell[0])/incr[0])+1)
	max_width = int(((gradient_values.shape[1]-cell[1])/incr[1])+1)
	
	#Calculating 8X8 cells
	while i<max_height:
		w = 0
		j = 0
		while j<max_width:
			for_hist = gradient_values[h:h+cell[0],w:w+cell[1]]
			for_wght = magnitude_values[h:h+cell[0],w:w+cell[1]]
			cell_values.append(calculate_histogram(for_hist,for_wght))
			j += 1
			w += incr[1]
		i += 1
		h += incr[0]
	cell_values = np.reshape(cell_values,(max_height, max_width, 8))

	#Normalising the blocks  
	w = 0
	h = 0
	i = 0
	j = 0
	block = [2,2]
	max_height = int((max_height-block[0])+1)
	max_width = int((max_width-block[1])+1)
	block_values = []
	while i<max_height:
		w = 0
		j = 0
		while j<max_width:
			for_norm = cell_values[h:h+block[0],w:w+block[1]]
			normlized_values=normalize(for_norm)
			block_values += normlized_values.flatten().tolist()
			j += 1
			w += 1
		i += 1
		h += 1

	return block_values       
def feature_Extraction(img):
    #Calculating Gradients (direction x and y)
    gX= cv2.Sobel(img,0, dx=1,dy=0, ksize=1)
    #gX_img= np.uint8(np.absolute(gX))
    gY= cv2.Sobel(img,0, dx=0,dy=1, ksize=1)
    #gY_img = np.uint8(np.absolute(gY))
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
    hog_features = create_hog_features(orientation,magnitude)
    hog_features = np.asarray(hog_features,dtype=float)
    hog_features = np.expand_dims(hog_features,axis=0)
    return hog_features
def test(imagesPath, label):
    for filename in glob.glob(imagesPath + '/*.*'):
        # print(filename.split(imagesPath + '/')[1])
        # print(filename)
        win_size = (64, 128)
        img = getImage(filename)

        win_size = (64, 128)
        img = cv2.resize(img, win_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);feat=feature_Extraction(img)
        predicted=int(clf.predict([feat.flatten().tolist()]))
        if label == predicted:
            global True_sum
            True_sum=True_sum+1
              

classes = ["HAPPY", "CONTEMPT", "ANGER", "DISGUST", "FEAR", "SADNESS", "SURPRISE", "NEUTRAL"]

####################################testing on testing set########################################
test('test_happy',0)
test('test_contempt',1)
test('test_anger',2)
test('test_disgust',3)
test('test_fear',4)
test('test_sadness',5)
test('test_surprise',6)
test('test_neutral',7)
print("accuracy: "+str((True_sum/4000)*100))
####################################testing on a specific image###################################
#img = getImage('aya_angry2.jpg')
#win_size = (64, 128)
#img = cv2.resize(img, win_size)
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#feat=feature_Extraction(img)
#predicted=int(clf.predict([feat.flatten().tolist()]))
#print(predicted)

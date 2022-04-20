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
def normalize(vector):
    prevType = vector.dtype
    if (vector.dtype == np.float16):
        vector = vector.astype(np.float32)
    norm = np.linalg.norm(vector)
    if (norm != 0 and np.isfinite(norm)):
        vector /= norm
    return vector.astype(prevType)	
def calculate_histogram(array,weights):
	bins_range = (0, 180)
	bins = 8
	hist,_ = np.histogram(array,bins=bins,range=bins_range,weights=weights)

	return hist  
	
def create_hog_features(grad_array,mag_array):
	max_h = int(((grad_array.shape[0]-cell[0])/incr[0])+1)
	max_w = int(((grad_array.shape[1]-cell[1])/incr[1])+1)
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
			for_hist = grad_array[h:h+cell[0],w:w+cell[1]]
			for_wght = mag_array[h:h+cell[0],w:w+cell[1]]
			
			val = calculate_histogram(for_hist,for_wght)
			cell_array.append(val)
			j += 1
			w += incr[1]

		i += 1
		h += incr[0]

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

		while j<max_w:
			for_norm = cell_array[h:h+block[0],w:w+block[1]]
			#mag = np.linalg.norm(for_norm)
			arr_list=normalize(for_norm)
			#arr_list = (for_norm/mag).flatten().tolist()
			block_list += arr_list.flatten().tolist()
			
			j += 1
			w += 1

		i += 1
		h += 1

	#returns a vextor array list of 288 elements
	return block_list      
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
import math
import pickle

import numpy as np
import cv2
#Import required modules
import cv2
#import dlib
import numpy as np
import math
import math

#import imutils
import numpy as np
import cv2 as cv2
import glob
import csv

#from imutils import face_utils
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
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.svm import SVC
import pickle

#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='rbf', probability=True, tol=1e-3)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cell=[8,8]
incr=[8,8]

cap = cv2.VideoCapture(0+ cv2.CAP_DSHOW)
filename = 'AffectNet_5000_svcOnly_rpf_manualFeatExt_model.sav'

face_cascade = cv2.CascadeClassifier('C:\\Users\\THINK\\Desktop\\College\\GP\\emotion extraction\\Emotion-Recognition-From-Facial-Expressions-master\\Trail1\\haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


clf = pickle.load(open(filename, 'rb'))
classes = ["HAPPY", "CONTEMPT", "ANGER", "DISGUST", "FEAR", "SADNESS", "SURPRISE", "NEUTRAL"]
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
while(True):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    crop_img = frame
    if len(faces) == 0:
        crop_img = frame
    else:
        crop_img = frame[y:y + h, x:x + w]

    win_size = (64, 128)

    resized_img = cv2.resize(crop_img, win_size)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    feat=feature_Extraction(resized_img)
    pred_value=int(clf.predict([feat.flatten().tolist()]))
    proba = clf.predict_proba([feat.flatten().tolist()])
    #print(proba)
    #print(math.floor((proba[0][0]*1000000))/10000)

    if(pred_value == 0):
        cv2.putText(frame, 'Happy: ' + str(math.floor((proba[0][0]*1000000))/10000) + '%', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Happy: ' + str(math.floor((proba[0][0] * 1000000)) / 10000) + '%', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if(pred_value == 1):
        cv2.putText(frame, 'Contempt: ' + str(math.floor((proba[0][0]*1000000))/10000) + '%', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Contempt: ' + str(math.floor((proba[0][0] * 1000000)) / 10000) + '%', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if (pred_value == 2):
         cv2.putText(frame, 'ANGER: ' + str(math.floor((proba[0][1]*1000000))/10000), (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'ANGER: ' + str(math.floor((proba[0][1] * 1000000)) / 10000), (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if (pred_value == 3):
        cv2.putText(frame, 'DISGUST: ' + str(math.floor((proba[0][2]*1000000))/10000), (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'DISGUST: ' + str(math.floor((proba[0][2] * 1000000)) / 10000), (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if(pred_value == 4):
        cv2.putText(frame, 'FEAR: ' + str(math.floor((proba[0][3]*1000000))/10000), (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'FEAR: ' + str(math.floor((proba[0][3] * 1000000)) / 10000), (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if (pred_value == 5):
        cv2.putText(frame, 'SADNESS: ' + str(math.floor((proba[0][4]*1000000))/10000), (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'SADNESS: ' + str(math.floor((proba[0][4] * 1000000)) / 10000), (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if(pred_value == 6):
        cv2.putText(frame, 'SURPRISE: ' + str(math.floor((proba[0][5]*1000000))/10000), (30, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'SURPRISE: ' + str(math.floor((proba[0][5] * 1000000)) / 10000), (30, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if (pred_value == 7):
        cv2.putText(frame, 'NEUTRAL: ' + str(math.floor((proba[0][6]*1000000))/10000), (30, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'NEUTRAL: ' + str(math.floor((proba[0][6] * 1000000)) / 10000), (30, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
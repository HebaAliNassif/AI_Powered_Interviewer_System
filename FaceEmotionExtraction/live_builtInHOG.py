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
"""
def get_landmark_positions(img):
    detections = detector(img, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(img, d)  # Draw Facial Landmarks with the predictor class
        shape2 = face_utils.shape_to_np(shape)
        ch = cv2.convexHull(shape2[48:68])
        M = cv2.moments(shape2[48:68])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        sum = 0
        for p in ch:
            i, j = p[0]
            v = (j - cY) / (i - cX)
            if ((i - cX) != 0):
                sum = sum + v


        xlist = []
        ylist = []
        (x, y, w, h) = cv2.boundingRect(np.array([shape2[48:68]]))
        roi = img[y:y + h, x:x + w]

        win_size = (64, 128)

        img = cv2.resize(img, win_size)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        d = cv2.HOGDescriptor()
        hog = d.compute(img)

        hog = hog.transpose()[0]
        hog = np.asarray(hog)


        for i in range(1, 68):  # Store X and Y coordinates in two lists
            # if(i >= 49 and i <= 68):
            #     print(shape.part(i))
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
    return xlist, ylist, hog,sum
def get_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    clahe_image = clahe.apply(gray)
    xlist, ylist, hog, sum = get_landmark_positions(clahe_image)
    features2 = []
    features2.extend(hog)
"""

cap = cv2.VideoCapture(0+ cv2.CAP_DSHOW)
filename = 'AffectNet_5000_svcOnly_rpf_model.sav'

face_cascade = cv2.CascadeClassifier('C:\\Users\\THINK\\Desktop\\College\\GP\\emotion extraction\\Emotion-Recognition-From-Facial-Expressions-master\\Trail1\\haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


clf = pickle.load(open(filename, 'rb'))
classes = ["HAPPY", "CONTEMPT", "ANGER", "DISGUST", "FEAR", "SADNESS", "SURPRISE", "NEUTRAL"]

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

    #feat = get_features(crop_img)
    #proba = clf.predict_proba([feat])
    #pred_value = clf.predict([feat])[0]
    resized_img = cv2.resize(crop_img, win_size)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    d = cv2.HOGDescriptor()
    hog = d.compute(resized_img)
    hog = hog.transpose()[0]
    hog = np.asarray(hog)
    pred_value=int(clf.predict([hog]))
    proba = clf.predict_proba([hog])
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
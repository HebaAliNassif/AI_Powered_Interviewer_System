import cv2
import shutil
import os
import numpy as np
import scipy as sp
import scipy.misc as spmisc
from face_detector import FaceDetector
from mergerect import mergeRects
import config

import pickle
from skimage import data, color, feature
from skimage import transform
if __name__ == '__main__':
    patch_size = (62, 47)
    
    with open('hog.pkl', 'rb') as fid:
        model = pickle.load(fid)
    
    video_capture = cv2.VideoCapture(0)
    if config.CBCL_DATASET:
        faceDetector = FaceDetector(config.MODEL_PATH + "/CBCL")
    else:
        faceDetector = FaceDetector(config.MODEL_PATH + "/LFW")
    
    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces, totalTiles = faceDetector.detect(
            gray,
            min_size=0.0, max_size=0.3,
            step=0.9, detectPad=(2,2),
            verbose=True,
            getTotalTiles=True
        )
        faces = mergeRects(
            faces,
            overlap_rate=0.82,
            min_overlap_cnt=4
        )
        print("----faces detected----")
        for x, y, w, h in faces:
            face = color.rgb2gray(frame[y:y+h, x:x+w])
            face = transform.resize(face, patch_size)
            pred = model.predict([feature.hog(face)])
            if pred:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        
        frame = cv2.resize(frame, (0,0), fx=1.0/0.4, fy=1.0/0.4)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    faceDetector.stopParallel()
    video_capture.release()
    cv2.destroyAllWindows()

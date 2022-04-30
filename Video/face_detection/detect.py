import cv2
import shutil
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "vj"))    
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
    
    video_capture = cv2.VideoCapture(0)
    if config.CBCL_DATASET:
        faceDetector = FaceDetector(config.VJ_MODEL_PATH + "/CBCL")
    else:
        faceDetector = FaceDetector(config.VJ_MODEL_PATH + "/LFW")
    
    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces, totalTiles = faceDetector.detect(
            gray,
            min_size=0.0, max_size=0.3,
            step=0.9, detectPad=(2,2),
            verbose=False,
            getTotalTiles=True
        )
        faces = mergeRects(
            faces,
            overlap_rate=0.82,
            min_overlap_cnt=4
        )

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        
        frame = cv2.resize(frame, (0,0), fx=1.0/0.4, fy=1.0/0.4)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    # When everything is done, release the capture
    faceDetector.stopParallel()
    video_capture.release()
    cv2.destroyAllWindows()

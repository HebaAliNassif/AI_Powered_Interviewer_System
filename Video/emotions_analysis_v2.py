from time import time
import os
import sys
import cv2
import numpy as np
from skimage import color, feature
import config
sys.path.append(os.path.join(os.path.dirname(__file__), "face_alignment"))
sys.path.append(os.path.join(os.path.dirname(__file__), "face_detection"))
sys.path.append(os.path.join(os.path.dirname(__file__), "face_detection/vj"))
sys.path.append(os.path.join(os.path.dirname(__file__), "face_detection/hog"))
#from hog import HogClassifier
#from face_detector import FaceDetector
from mergerect import mergeRects
#from face_aligner import FaceAligner
#print(os.getcwd())




def openCV_detectface(image):
    cascPath = "haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30))
    #print(len(faces))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        return True, face
    else:
        return False, None
        
def vj_hog_detectface(image,faceDetector,hogModel,faceAligner):

    image = cv2.resize(image, (0,0), fx=0.4, fy=0.4)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces, totalTiles = faceDetector.detect(
        gray,
        min_size=0.0, max_size=0.3,
        step=0.9, detectPad=(2,2),
        verbose=False,
        getTotalTiles=True
    )
    
    faces = mergeRects(faces,overlap_rate=0.82,min_overlap_cnt=4)
    for x, y, w, h in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (hogModel.detectWndW, hogModel.detectWndH))
        pred = hogModel.clf.predict([feature.hog(face)])
        if pred[0]:
            #face = faceAligner.align_face(image, (x, y, w, h))
            return True, face
    return False, None
def analyse_emotions(video_path,faceDetector,hogModel,faceAligner, opencv_fd=True, frames_path=None):        
    ListOfFaceDetectedImages=[]
    video = cv2.VideoCapture(video_path)
    sucess, frame = video.read()
    count = 0
    while sucess:
        count += 1
        
        if frames_path:
            os.makedirs(os.path.split(frames_path+"/")[0], exist_ok=True)
            cv2.imwrite(frames_path+"/frame%d.jpg"%count, frame)
            
        found = False
        
        if opencv_fd:
            found, face = openCV_detectface(frame)
        else:
            found, face = vj_hog_detectface(frame,faceDetector,hogModel,faceAligner)
            
        if found:
            #os.makedirs(os.path.split(faces_path+"/")[0], exist_ok=True)
            #print("ListOfFaceDetectedImages append image:"+str(count))
            ListOfFaceDetectedImages.append(face)
            #cv2.imwrite(faces_path+"/face%d.jpg"%count, face)
        
        sucess, frame = video.read()
    return ListOfFaceDetectedImages    


''''        
if __name__ == '__main__':
    print("-" * 80)
    print("Loading Models")
    print("-" * 80)
    
    faceDetector = FaceDetector(config.VJ_MODEL_PATH)
    hogModel = HogClassifier.loadModel(config.HOG_MODEL_PATH)
    faceAligner = FaceAligner(desiredFaceWidth=100)
    print("...done.\n" + "-" * 80, "\n")
    
    print("-" * 80)
    print("Testing")
    print("-" * 80)
    
    t_start = time()
    
    # To visualize farmes and detected faces, use
    #analyse_emotions(config.VIDEO_PATH+"/"+config.VIDEO_NAME, config.OPENCV_FACE_DETECTION, config.FRAMES_DIRECTORY, faces_path=config.FACES_DIRECTORY)
    analyse_emotions(config.VIDEO_PATH+"/"+config.VIDEO_NAME, config.OPENCV_FACE_DETECTION, faces_path=config.FACES_DIRECTORY)
    # else, use
    #analyse_emotions(config.VIDEO_PATH+"/"+config.VIDEO_NAME, config.OPENCV_FACE_DETECTION)
        
    time_full_face_detection = time() - t_start
    print("...done.\n\tTime Elapsed:", time_full_face_detection)
    print("-" * 80, "\n")
    faceDetector.stopParallel()
'''
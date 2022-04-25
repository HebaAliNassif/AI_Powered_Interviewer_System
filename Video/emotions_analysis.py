import cv2
import config
import os
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
        face = image[y:y+h, x:x+w]
        return True, face
    else:
        return False, None

def analyse_emotions(video_path, opencv_fd=True, frames_path=None, faces_path=None):
    video = cv2.VideoCapture(video_path)
    sucess, frame = video.read()
    count = 0
    while sucess:
        count += 1
        
        if frames_path:
            os.makedirs(os.path.split(frames_path+"/")[0], exist_ok=True)
            cv2.imwrite(frames_path+"/frame%d.jpg"%count, frame)
        
        found, face = openCV_detectface(frame)
        
        if faces_path and found:
            os.makedirs(os.path.split(faces_path+"/")[0], exist_ok=True)
            cv2.imwrite(faces_path+"/face%d.jpg"%count, face)
        
        sucess, frame = video.read()
        
if __name__ == '__main__':
    # To visualize farmes and detected faces, use
    #analyse_emotions(config.VIDEO_PATH+"/"+config.VIDEO_NAME, config.OPENCV_FACE_DETECTION, config.FRAMES_DIRECTORY, config.FACES_DIRECTORY)
    
    # else, use
    analyse_emotions(config.VIDEO_PATH+"/"+config.VIDEO_NAME, config.OPENCV_FACE_DETECTION)
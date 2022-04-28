import numpy as np
import cv2
import math
from eye_center_locator import getEyesCenter

class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def rotate_image(self, image, angle, face):
        x, y, w, h = face
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        x, y = (int(rot_mat[0,0]*x+ rot_mat[0,1]*y + rot_mat[0,2]) , int(rot_mat[1,0]*x+ rot_mat[1,1]*y + rot_mat[1,2]))
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)[y :y + h , x :x + w ]
        return result
        
    def align(self, image, leftEyeCenter, rightEyeCenter, face):
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))- 180
        result = self.rotate_image(image, angle, face)
        return result
        
    def align_face(self, image, face_):
        x, y, w, h = face_
        face = image[y :y + h , x :x + w ]
        leftEyeCenter, rightEyeCenter = getEyesCenter(face)
        faceAligned = self.align(image, rightEyeCenter, leftEyeCenter, face_)
        return faceAligned
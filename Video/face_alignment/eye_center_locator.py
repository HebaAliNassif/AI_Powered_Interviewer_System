import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import config
def getEyesCenter(face_BGR):
    
    eyeCenterLocator = EyeCenterLocator()
    img_GRAY = cv2.cvtColor(face_BGR, cv2.COLOR_BGR2GRAY)
    
    eye_region_width = face_BGR.shape[0] * (config.K_EYE_PERCENT_WIDTH/100.0)
    eye_region_height = face_BGR.shape[0] * (config.K_EYE_PERCENT_HEIGHT/100.0)
    eye_region_top = face_BGR.shape[1] * (config.K_EYE_PERCENT_TOP/100.0)
    
    leftEyeRegion = (1,1,1,1)
    rightEyeRegion = (1,1,1,1)

    leftEyeRegion = int(face_BGR.shape[0]*(config.K_EYE_PERCENT_SIDE/100.0)), int(eye_region_top), int(eye_region_width), int(eye_region_height)
    rightEyeRegion = int(face_BGR.shape[0] - eye_region_width - face_BGR.shape[0]*(config.K_EYE_PERCENT_SIDE/100.0)), int(eye_region_top),int(eye_region_width),int(eye_region_height)
    
    x1, y1, w1, h1 = (leftEyeRegion)
    region_left = img_GRAY[int(y1) :int(y1) + int(h1) , int(x1) :int(x1) + int(w1)]
    leftEyeCenter = eyeCenterLocator.locate(region_left)
    
    x2, y2, w2, h2 = (rightEyeRegion)
    region_right = img_GRAY[int(y2) :int(y2) + int(h2) , int(x2) :int(x2) + int(w2)]
    rightEyeCenter = eyeCenterLocator.locate(region_right)
    
    return (int(leftEyeCenter[1])+int(x1) , int(leftEyeCenter[0])+int(y1)), (int(rightEyeCenter[1])+int(x2), int(rightEyeCenter[0])+int(y2))
    
class EyeCenterLocator:

    def __init__(self, blur = 3, minrad = 2, maxrad = 20):
        self.blur = blur
        self.minrad = minrad
        self.maxrad = maxrad
    
    def locate(self, image_BGR):
        image_BGR = gaussian_filter(image_BGR, sigma=self.blur)
        image_BGR = (image_BGR.astype('float') - np.min(image_BGR))
        image_BGR = image_BGR / np.max(image_BGR)
        
        Ly, Lx = np.gradient(image_BGR)
        Lyy, Lyx = np.gradient(Ly)
        Lxy, Lxx = np.gradient(Lx)
        Lvv = Ly**2 * Lxx - 2*Lx * Lxy * Ly + Lx**2 * Lyy
        Lw =  Lx**2 + Ly**2
        Lw[Lw==0] = 0.001
        Lvv[Lvv==0] = 0.001
        k = - Lvv / (Lw**1.5)
        
        Dx =  -Lx * (Lw / Lvv)
        Dy =  -Ly * (Lw / Lvv)
        displacement = np.sqrt(Dx**2 + Dy**2)
        
        curvedness = np.absolute(np.sqrt(Lxx**2 + 2 * Lxy**2 + Lyy**2))
        center_map = np.zeros(image_BGR.shape, image_BGR.dtype)
        for i in range(center_map.shape[0]):
            for j in range(center_map.shape[1]):
                if Dx[i][j] == 0 and Dy[i][j] == 0:
                    continue
                if (j + Dx[i][j])>0 and (i + Dy[i][j])>0:
                    if (j + Dx[i][j]) < center_map.shape[1] and (i + Dy[i][j]) < center_map.shape[0] and k[i][j]<0:
                        if displacement[i][j] >= self.minrad and displacement[i][j] <= self.maxrad:
                            center_map[int(i+Dy[i][j])][int(j+Dx[i][j])] += curvedness[i][j]
        center_map = gaussian_filter(center_map, sigma=self.blur)
        blurred = gaussian_filter(image_BGR, sigma=self.blur)
        center_map = center_map * (1-blurred)
        position = np.unravel_index(np.argmax(center_map), center_map.shape)
        return position
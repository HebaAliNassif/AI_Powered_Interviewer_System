from time import time
import progressbar
import cv2
import numpy as np
import os
from sklearn.datasets import fetch_lfw_people
from itertools import chain
import pickle
import datetime
from skimage import data, color, feature

import utils
import config

from hog import HogClassifier

if __name__ == '__main__':
    clf = HogClassifier(config.WINDOW_SIZE[1], config.WINDOW_SIZE[0])
    
    print("-" * 80)
    print("Loading Dataset")
    print("-" * 80)
    DatasetName = ""
    if config.CBCL_DATASET:
        DatasetName = "CBCL"
        
        print('Loading face images for training...')
        cbcl_faces_data = utils.load_images(config.CBCL_TRAIN_FACE_PATH)
        cbcl_faces_training = np.asarray([cv2.resize(img, (config.WINDOW_SIZE[1], config.WINDOW_SIZE[0])) for img in cbcl_faces_data])
        print('...done. \n' + str(len(cbcl_faces_training)), 'faces with size =', cbcl_faces_training[0].shape,'are loaded.\n')
        
        print('Loading non face images for training...')
        cbcl_non_faces_data = utils.load_images(config.CBCL_TRAIN_NON_FACE_PATH)
        cbcl_non_faces_training = np.asarray([cv2.resize(img, (config.WINDOW_SIZE[1], config.WINDOW_SIZE[0])) for img in cbcl_non_faces_data])
        print('...done. \n' + str(len(cbcl_non_faces_training)), 'non faces with size =', cbcl_non_faces_training[0].shape, 'are loaded.\n')
        
        P_train = cbcl_faces_training
        N_train = cbcl_non_faces_training

    else:
        DatasetName = "LFW"
        print('Loading face images for training...')
        lfw = fetch_lfw_people().images
        faces_lfw = np.asarray([cv2.resize(img, (config.WINDOW_SIZE[1], config.WINDOW_SIZE[0])) for img in lfw])
        print('...done. \n' + str(len(faces_lfw)), 'faces with size =', faces_lfw[0].shape,'are loaded.')
        print(faces_lfw.shape)
        
        imgs_to_use = ['camera', 'text', 'coins', 'moon',
               'page', 'clock', 'immunohistochemistry',
               'chelsea', 'coffee', 'hubble_deep_field']
        non_face_images = []
        for name in imgs_to_use:
            img = getattr(data, name)()
            try:
                img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                non_face_images.append(img2)
            except:
                non_face_images.append(img)
                
        print('Loading non face images for training...')
        non_faces_skimage = np.vstack([utils.extract_patches(img, 1000, faces_lfw[0].shape,scale) for img in non_face_images for scale in [0.5, 1.0, 2.0]])
        print('...done.\n' + str(len(non_faces_skimage)), 'faces with size =', non_faces_skimage.shape,'are loaded.')
        print(non_faces_skimage.shape)
        
        P_train = faces_lfw
        
        N_train = non_faces_skimage
    
    print("...done.\n" + "-" * 80, "\n")

    print("-" * 80)
    print("Begin feature extraction...")
    print("-" * 80)
    
    t_start = time()
    #X_train = np.array([feature.hog(im) for im in chain(P_train, N_train)], dtype=object)
    X_train = np.array([clf.extractManualHogFeature(im) for im in chain(P_train, N_train)], dtype=object)
    y_train = np.zeros(X_train.shape[0])
    y_train[:P_train.shape[0]] = 1
    time_full_feature_comp = time() - t_start
    
    print("...done.\n\tTime Elapsed:", time_full_feature_comp)
    print("-" * 80, "\n")
    
    ModelFile = config.MODELS_PATH + "/" + "model-"+ str(datetime.datetime.now().date()) + "/"+ DatasetName
    #ModelFile = config.MODELS_PATH + "/" + "model-hog" + "/"+ DatasetName

    print("-" * 80)
    print("Begin Training")
    print("-" * 80)
    
    t_start = time()
    clf.train(X_train, y_train)
    time_full_train = time() - t_start
    
    print("...done.\n\tTime Elapsed:", time_full_train)
    print("-" * 80, "\n")
    
    print("-" * 80)
    print("Saving Model")
    print("-" * 80)
    clf.saveModel(ModelFile)
    print("...done.\n" + "-" * 80, "\n")
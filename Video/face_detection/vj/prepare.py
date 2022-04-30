from time import time
import utils
import numpy as np
import cv2

import config
from boosted_cascade import BoostedCascade
from sklearn.datasets import fetch_lfw_people
from skimage import data, color

if __name__ == '__main__':
    print("-" * 80)
    print("Loading Dataset")
    print("-" * 80)
    
    DatasetName = ""
    if config.CBCL_DATASET:
        DatasetName = "CBCL"
        
        print('Loading face images for training...')
        cbcl_faces_data = utils.load_images(config.CBCL_TRAIN_FACE_PATH)
        cbcl_faces_training = np.asarray([cv2.resize(img, config.WINDOW_SIZE) for img in cbcl_faces_data])
        print('...done. \n' + str(len(cbcl_faces_training)), 'faces with size =', cbcl_faces_training[0].shape,'are loaded.\n')
        
        print('Loading non face images for training...')
        cbcl_non_faces_data = utils.load_images(config.CBCL_TRAIN_NON_FACE_PATH)
        cbcl_non_faces_training = np.asarray([cv2.resize(img, config.WINDOW_SIZE) for img in cbcl_non_faces_data])
        print('...done. \n' + str(len(cbcl_non_faces_training)), 'non faces with size =', cbcl_non_faces_training[0].shape, 'are loaded.\n')
        
        P_train = cbcl_faces_training
        N_train = cbcl_non_faces_training

    else:
        DatasetName = "LFW"
        print('Loading face images for training...')
        lfw = fetch_lfw_people().images
        faces_lfw = np.asarray([cv2.resize(img, config.WINDOW_SIZE) for img in lfw])
        print('...done. \n' + str(len(faces_lfw)), 'faces with size =', faces_lfw[0].shape,'are loaded.\n')
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
        print('...done.\n' + str(len(non_faces_skimage)), 'faces with size =', non_faces_skimage.shape,'are loaded.\n')
        print(non_faces_skimage.shape)
        
        P = faces_lfw
        N = []
        
        N = non_faces_skimage
        testset_rate = 0.3
        divlineP = int(len(P)* testset_rate)
        divlineN = int(len(N)* testset_rate)
        
        P_train = P[divlineP:len(P)]
        N_train = N[divlineN:len(N)]
        #P_test = P[0:divlineP]
        #N_test = N[0:divlineN]
        
    print("-" * 80, "\n")
    
    clf = BoostedCascade(0.07, 0.60, 0.94)
    
    print("-" * 80)
    print("Begin feature extraction...")
    print("-" * 80)
    
    t_start = time()
    clf.prepare(P_train, N_train, shuffle=True, verbose=True, max_parallel_process=4)
    
    time_full_feature_comp = time() - t_start
    
    print("...done.\n\tTime Elapsed:", time_full_feature_comp)
    print("-" * 80, "\n")
    
    print("-" * 80)
    print("Saving Features...")
    print("-" * 80)
    clf.saveFeaturesData(config.FEATURES_PATH +"/"+ DatasetName)
    print("...done.\n")
    print("-" * 80, "\n")
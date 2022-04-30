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
    
    print("-" * 80)
    print("Loading Model")
    print("-" * 80)
    DatasetName = ""
    if config.CBCL_DATASET:
        DatasetName = "CBCL"
        
    else:
        DatasetName = "LFW"
    
    ModelFile = config.MODEL_PATH + "/"+ DatasetName
    clf = HogClassifier.loadModel(ModelFile)

    print("...done.\n" + "-" * 80, "\n")
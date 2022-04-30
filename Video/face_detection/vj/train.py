from time import time
import utils
import numpy as np
import cv2

import config
from boosted_cascade import BoostedCascade
import datetime
if __name__ == '__main__':
    clf = BoostedCascade(0.07, 0.6, 0.9)

    print("-" * 80)
    print("Loading Features")
    print("-" * 80)
    DatasetName = ""

    if config.CBCL_DATASET:
        DatasetName = "CBCL"
    else:
        DatasetName = "LFW"
        
    clf.loadFeaturesData(config.FEATURES_PATH +"/"+ DatasetName)

    print("...done.\n" + "-" * 80, "\n")
    ModelFile = config.MODELS_PATH + "/" + "model-"+ str(datetime.datetime.now().date()) + "/"+ DatasetName

    print("-" * 80)
    print("Begin Training")
    print("-" * 80)
    clf.train(is_continue=False, autosnap_filename=ModelFile, verbose=True)
    print("...done.\n" + "-" * 80, "\n")
    
    print("-" * 80)
    print("Saving Model")
    print("-" * 80)
    clf.saveModel(ModelFile)
    print("...done.\n" + "-" * 80, "\n")

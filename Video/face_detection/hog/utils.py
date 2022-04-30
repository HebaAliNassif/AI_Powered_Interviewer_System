import os
import imageio
import numpy as np
import cv2

from sklearn.feature_extraction.image import PatchExtractor
from skimage import transform

def load_images(path, normalize=False, verbose=False):
    """
    Loads images in a given directory.
    
    Parameters
    ---------- 
    path : str
      Directory Path.
    normalize : bool
      Weather to normalize images using  variance normalization or not.
    
    Returns
    -------
    images: numpy.ndarray
      loaded images in grayscale
    """
    images = []
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):
            images = images + load_images(abs_path, verbose)
        else:
            # pilmode=F --> 32-bit floating point pixels
            # as_gray --> flatten
            img = imageio.imread(abs_path, as_gray=False, pilmode="F")
            if normalize:
                img = (img - img.mean(axis=0)) / img.std(axis=0)
                #img = cv2.normalize(img,  None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(img)
            if verbose:
                sys.stdout.write("\r\033[K")
                print('%d images loaded.' % len(images), end='', flush=True)
    if verbose: 
        sys.stdout.write("\r\033[K")
    return np.asarray(images)
    
def extract_patches(img, N, patch_size, scale=1.0):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                               max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches

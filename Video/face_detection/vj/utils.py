import os
import imageio
import numpy as np
import cv2

from sklearn.feature_extraction.image import PatchExtractor
from skimage import transform

def integral_image(img_arr):
    """
    Calculates the integral image based on the original image data.
    The integral image contains the sum of all elements above and to the left of it.

    Parameters
    ---------- 
    img_arr : numpy.ndarray
      Input image.

    Returns
    -------
    ii: numpy.ndarray
      Integral image for the input image.

    """
    """
    cumsum: returns the cumulative sum of the elements along a given axis
      axis=0 --> sum over rows
      axis=1 --> sum over columns
      So, first sum over rows then sum of columns
    """    
    ii = img_arr
    for i in range(img_arr.ndim):
      ii = ii.cumsum(axis=i)
    return ii

def sum_rect(ii, top_left, dimention):
    """
    Calculates the sum of image in the rectangle specified by the given tuples.
    
    Parameters
    ---------- 
    ii : numpy.ndarray
      Integral image.
    top_left-(y, x): (int, int)
      Rectangle's top left corner.
    dimention-(h, w): (int, int)
      Rectangle's dimentions.
      
    Returns
    -------
    sum_val: int
      The sum of all image pixels in the given rectangle.
      
    """
    assert type(top_left) == tuple
    assert type(dimention) == tuple
    bottom_right = (top_left[0]+dimention[0], top_left[1]+dimention[1])
    if top_left == bottom_right:
        return ii[top_left]
    top_right = (top_left[0], top_left[1]+dimention[1])
    bottom_left = (top_left[0]+dimention[0], top_left[1])
    #print(top_left, top_right, bottom_left, bottom_right)
    sum_val = ii[top_left] + ii[bottom_right] - ii[top_right] - ii[bottom_left]
    return sum_val

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
import cv2
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def read_images(ref_im_path:str, query_im_path:str, is_debug:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    read two images given their path, will plot the images if 'is_debug' is set to 'True'
    """
    ref_image = cv2.imread(ref_im_path)
    query_image = cv2.imread(query_im_path)
    assert ref_image.shape == query_image.shape, "images shapes missmatch!"
    if is_debug:
        # Display the images to verify that they have been read correctly
        cv2.imshow('Reference Image', ref_image)
        cv2.waitKey()
        cv2.imshow('Query Image', query_image)
        cv2.waitKey()
    
    # Convert to grayscale
    ref_image     = cv2.cvtColor(ref_image   , cv2.COLOR_BGR2GRAY)
    query_image   = cv2.cvtColor(query_image , cv2.COLOR_BGR2GRAY)
    return ref_image, query_image


def estimate_sigma(image:np.ndarray) -> float:
    """
    This function estimates the std (sigma) of the white additive Gaussian
    that was added to the image using the 'Median Absolute Deviation (MAD) method'
    """
    # Convert the image to a float32 array
    img = np.float32(image)

    # Compute the median of the image
    median = np.median(img)

    # Compute the absolute deviation of each pixel from the median
    abs_dev = np.abs(img - median)

    # Compute the median absolute deviation (MAD)
    MAD = np.median(abs_dev)

    # Compute the estimate of sigma based on the MAD
    sigma = 1.4826 * MAD

    return sigma

def find_transformation(ref_im:np.ndarray,
                              query_im:np.ndarray,
                              is_debug:bool=False) -> np.ndarray:
    """
    Finds a rigid transformation between the query image to the ref image
    and return the transformation.
    """

    ref_img   = apply_LoG(ref_im)
    query_img = apply_LoG(query_im)

    # Define the template size
    y, x = ref_img.shape
    min_size = min(x, y)
    size = min_size // 2
    template_size = (size, size)

    # Define the template from the reference image
    top_left = (ref_img.shape[0] // 2 - template_size[0] // 2, ref_img.shape[1] // 2 - template_size[1] // 2)
    template = query_img[top_left[0]:top_left[0]+template_size[0], top_left[1]:top_left[1]+template_size[1]]

    # Perform template matching to determine the shift between the two images
    result = cv2.matchTemplate(ref_img, template, cv2.TM_CCOEFF)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Get the x and y shift from the template matching result
    shift_x = top_left[1] - max_loc[0]
    shift_y = top_left[0] - max_loc[1]

    # Create a transformation matrix to shift the target image
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    return M

def apply_LoG(image:np.ndarray) -> np.ndarray:
    # Apply Gaussian blur to the image
    im = cv2.GaussianBlur(image, (3, 3), 0)
    # Apply Laplacian to sharpen it
    im = cv2.Laplacian(im, cv2.CV_16S, ksize=3)
    # Back to uint8
    im = cv2.convertScaleAbs(im)
    return im

def apply_morphological(binary_mask:np.ndarray) -> np.ndarray:

    # Define a kernel for the morphological filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply a morphological opening operation to the thresholded image
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    return opening

def parse_to_bool(word:str) -> bool:
    if word == "False":
        return False
    elif word == "True":
        return True
    else:
        raise ValueError("Please pass 'False' or 'True'")
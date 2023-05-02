import cv2
from typing import Tuple
import numpy as np


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

def find_rigid_transformation(ref_im:np.ndarray,
                              query_im:np.ndarray,
                              is_debug:bool=False)\
    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds a rigid transformation between the query image to the ref image
    and return the ref image, transformed query image and the 
    transformation it self.
    """
    # Convert the images to grayscale
    ref_img     = cv2.cvtColor(ref_im   , cv2.COLOR_BGR2GRAY)
    query_img   = cv2.cvtColor(query_im , cv2.COLOR_BGR2GRAY)

    use_ECC = True
    if use_ECC:
        # Define the motion model to use (MOTION_TRANSLATION for translation only)
        motion_model = cv2.MOTION_TRANSLATION

        # Set the termination criteria for the iterative algorithm (max number of iterations and epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)

        # Estimate the 2D rigid transformation using cv2.findTransformECC()
        cc, transformation = cv2.findTransformECC(ref_img, query_img, np.eye(2, 3, dtype=np.float32),
                                                motion_model, criteria, None, 1)
    else:
        # Find keypoints and descriptors in the two images using ORB
        orb = cv2.ORB_create()
        ref_kp, ref_desc = orb.detectAndCompute(ref_img, None)
        query_kp, query_desc = orb.detectAndCompute(query_img, None)

        # Find matches between the descriptors using brute force matching
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(ref_desc, query_desc)

        # Sort the matches by their distances
        matches = sorted(matches, key=lambda x: x.distance)

        # Use the first 10% of matches to find the homography between the images
        num_matches = int(len(matches) * 0.1)
        ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches[:num_matches]]).reshape(-1, 1, 2)
        query_pts = np.float32([query_kp[m.trainIdx].pt for m in matches[:num_matches]]).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(query_pts, ref_pts, cv2.RANSAC)
        transformation = np.array([[1, 0, homography[0, 2]], [0, 1, homography[1, 2]]])
    
    # Transform the query image to the reference image using cv2.warpAffine()
    transformed_query_img = cv2.warpAffine(query_img, transformation, (ref_img.shape[1], ref_img.shape[0]))

    if is_debug:
        # Display the transformed query image and the reference image side by side
        result_img = np.hstack((ref_img, transformed_query_img))
        cv2.imshow('Transformed query image (right) and reference image (left)', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ref_im, transformed_query_img, transformation
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
    # query_img = ref_img

    # ref_img   = apply_LoG(ref_img)
    # query_img = apply_LoG(query_img)

    use_sift = False
    if use_sift:
        # Create a SIFT feature detector object
        sift = cv2.xfeatures2d.SIFT_create()

        # Detect and compute keypoints and descriptors for both images
        kp1, des1 = sift.detectAndCompute(ref_img, None)
        kp2, des2 = sift.detectAndCompute(query_img, None)

        # Create a brute force matcher object
        bf = cv2.BFMatcher()

        # Match the descriptors between the two images
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply the ratio test to filter out ambiguous matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract the corresponding keypoint coordinates for the good matches
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate the homography matrix between the two images using RANSAC algorithm
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        # Warp the second image to align it with the first image using the estimated homography matrix
        warped_img = cv2.warpPerspective(query_img, H, (ref_img.shape[1], ref_img.shape[0]))

        # Display the results
        cv2.imshow("Image 1", ref_img)
        cv2.imshow("Image 2", query_img)
        cv2.imshow("Warped Image 2", warped_img)
        res = ref_img - warped_img
        min_res = res.min()
        res = res + min_res
        cv2.imshow("diff", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:

        # Define the template size
        template_size = (100, 100)

        # Define the template from the reference image
        top_left = (query_img.shape[0] // 2 - template_size[0] // 2, query_img.shape[1] // 2 - template_size[1] // 2)
        template = ref_img[top_left[0]:top_left[0]+template_size[0], top_left[1]:top_left[1]+template_size[1]]

        # Perform template matching to determine the shift between the two images
        result = cv2.matchTemplate(query_img, template, cv2.TM_CCOEFF)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Get the x and y shift from the template matching result
        shift_x = top_left[1] - max_loc[0]
        shift_y = top_left[0] - max_loc[1]

        # Create a transformation matrix to shift the target image
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Apply the transformation matrix to the target image
        img_aligned = cv2.warpAffine(query_img, M, (query_img.shape[1], query_img.shape[0]))

        # # Display the aligned image
        # cv2.imshow('Aligned Image', img_aligned)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
         # Display the results
        cv2.imshow("Image 1", ref_img)
        cv2.imshow("Image 2", query_img)
        cv2.imshow("Warped Image 2", img_aligned)
        res = ref_img - img_aligned
        min_res = res.min()
        res = res + min_res
        cv2.imshow("diff", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ref_im, warped_img, H

def apply_LoG(image:np.ndarray) -> np.ndarray:
    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (9, 9), 0)

    # # Apply Laplacian of Gaussian (LoG) filter to the blurred image
    # log = cv2.Laplacian(blurred, cv2.CV_64F)

    # # Convert the result to uint8 and normalize the values to the range [0, 255]
    # log = cv2.convertScaleAbs(log)
    # cv2.normalize(log, log, 0, 255, cv2.NORM_MINMAX)

    # Display the result
    # cv2.imshow("LoG Filtered Image", blurred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return blurred
import argparse
import cv2
from typing import Tuple
import numpy as np

def read_images(ref_im_path:str, query_im_path:str, is_debug:bool) -> Tuple(np.ndarray, np.ndarray):
    """
    read two images given their path, will plot the images if 'is_debug' is set to 'True'
    """
    ref_image = cv2.imread(ref_im_path)
    query_image = cv2.imread(query_im_path)
    assert ref_image.shape == query_image.shape, "images shapes missmatch!"
    if is_debug:
        # Display the images to verify that they have been read correctly
        cv2.imshow('Reference Image', ref_image)
        cv2.imshow('Query Image', query_image)
        cv2.waitKey()

    return ref_image, query_image

def main():
    # Create an ArgumentParser object to get the image paths as input arguments
    parser = argparse.ArgumentParser(description='Read two images using OpenCV')
    parser.add_argument('ref_image_path', type=str, help='Path to the reference image')
    parser.add_argument('query_image_path', type=str, help='Path to the query image')
    parser.add_argument('is_debug', type=bool, default=False, help='Debug falg, set true for debug properties')
    args = parser.parse_args()

    # Read the images using OpenCV
    ref_im, query_im = read_images(args.ref_image_path, args.query_image_path, args.is_debug)
    # ref_image = cv2.imread(args.ref_image_path)
    # query_image = cv2.imread(args.query_image_path)


if __name__ == '__main__':
    main()

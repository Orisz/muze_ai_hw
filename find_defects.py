import argparse
from utils.im_utils import *

def run_pipe(args):
    # Read the images
    ref_im, query_im = read_images(args.ref_image_path, args.query_image_path)

    # Estimate the noise level of the image
    # ref_sigma   = estimate_sigma(ref_im)
    # query_sigma = estimate_sigma(query_im)

    # Find transformation between the query image to the ref image
    _, transformed_query_img, trans = find_rigid_transformation(ref_im, query_im, args.is_debug)

def main():
    # Create an ArgumentParser object to get the image paths as input arguments
    parser = argparse.ArgumentParser(description='Defect Finder')
    parser.add_argument('--ref_image_path', type=str, help='Path to the reference image')
    parser.add_argument('--query_image_path', type=str, help='Path to the query image')
    parser.add_argument('--is_debug', type=bool, default=False, help='Debug falg, set true for debug properties')
    args = parser.parse_args()

    # Run analysis pipeline
    run_pipe(args)


if __name__ == '__main__':
    main()

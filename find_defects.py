import argparse
from utils.im_utils import *
from utils.hist_thresh import GHT

def find_defects(ref_im, query_img_aligned):
    diff_im = ref_im - query_img_aligned
    diff_im = cv2.normalize(diff_im  , None, 0, 255, cv2.NORM_MINMAX)
    ret, otsu = cv2.threshold(diff_im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
def run_pipe(args):
    # Read the images
    ref_im, query_im = read_images(args.ref_image_path, args.query_image_path)

    # Estimate the noise level of the image
    # ref_sigma   = estimate_sigma(ref_im)
    # query_sigma = estimate_sigma(query_im)

    # Find transformation between the query image to the ref image
    ref_im, query_img_aligned = align_images(ref_im, query_im, args.is_debug)
    find_defects(ref_im, query_img_aligned)

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

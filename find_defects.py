import argparse
from utils.im_utils import *
import scipy.stats as stats
import os


def save_mask(binary_mask:np.ndarray, dst:str, im_path:str, conf_lvl:int):
    """
    saves the musk given some paras such as destination image name etc.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    im_name = im_path.split("/")[-1]
    im_full_loc = os.path.join(dst, f"mask_{conf_lvl}%_conf_lvl_"+im_name)
    cv2.imwrite(im_full_loc, binary_mask)


def find_defects(ref_im, query_im, M, confidence_level):
    """
    Given the images, the transformation, and the desired confidence level
    align the images
    =>
    Find the defects with:
    p(pixel_defect | x,y) >= confidence_level
    =>
    Return a binary musk of the defects
    """
    assert type(confidence_level)==int and confidence_level>=0 and confidence_level<=100,\
          "'confidence_level' must be an integer in range [0,100]"
    
    # Apply the transformation matrix to the target image
    img_aligned = cv2.warpAffine(ref_im, M, (ref_im.shape[1], ref_im.shape[0]))

    # Estimate the noise level of the image
    query_sigma = estimate_sigma(query_im)

    # Pad the blank area after translation to avoid false detection
    x_shift     = int(M[0, 2])
    y_shift     = int(M[1, 2])
    if x_shift <= 0:
        img_aligned[:, x_shift:] = query_im[:, x_shift:]
    else:
        img_aligned[:, :x_shift] = query_im[:, :x_shift]
    if y_shift <= 0:
        img_aligned[y_shift:, :] = query_im[y_shift:, :]
    else:
        img_aligned[:y_shift, :] = query_im[:y_shift, :]

    # Find the diff image
    diff = cv2.absdiff(query_im, img_aligned)

    # Find the threshold w.r.t. the desired confidence level 
    thresh = stats.norm.ppf((100 + confidence_level) / 200, scale=query_sigma)

    # Threshold the image to get the binary mask
    _, binary_mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    return binary_mask

def run_pipe(args):
    # Read the images
    ref_im, query_im = read_images(args.ref_image_path, args.query_image_path)

    # Find transformation between the query image to the ref image
    M = find_transformation(ref_im, query_im, args.is_debug)

    # Find the defects
    binary_mask = find_defects(ref_im, query_im, M, args.confidence_level)

    save_mask(binary_mask, args.dst_path, args.query_image_path, args.confidence_level)

def main():
    # Create an ArgumentParser object to get the image paths as input arguments
    parser = argparse.ArgumentParser(description='Defect Finder')
    parser.add_argument('--ref_image_path', type=str, help='Path to the reference image')
    parser.add_argument('--query_image_path', type=str, help='Path to the query image')
    parser.add_argument('--confidence_level', type=int, help='desired conf level, provide an integer within the range [0,100]')
    parser.add_argument('--dst_path', type=str, help='location folder to save the results in')
    parser.add_argument('--is_debug', type=bool, default=False, help='Debug falg, set true for debug properties')
    args = parser.parse_args()

    # Run analysis pipeline
    run_pipe(args)


if __name__ == '__main__':
    main()

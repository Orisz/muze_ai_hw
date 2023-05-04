import os
import fire
import sys
from tqdm import tqdm


def run_on_all_images(confidence_level:int):
    assert type(confidence_level)==int and confidence_level>=0 and confidence_level<=100,\
        "'confidence_level' must be an integer in range [0,100]"
    
    PYTHON_PATH = sys.executable
    BASE = os.getcwd()
    MAIN_SCRIPT = os.path.join(BASE, 'find_defects.py')
    im_base_def = os.path.join(BASE, 'data', 'defective_examples')
    im_base_non_def = os.path.join(BASE, 'data', 'non_defective_examples')
    ref =   [os.path.join(im_base_def     , "case1_reference_image.tif"),
             os.path.join(im_base_def     , "case2_reference_image.tif"),
             os.path.join(im_base_non_def , "case3_reference_image.tif")]
    query = [os.path.join(im_base_def     , "case1_inspected_image.tif"),
             os.path.join(im_base_def     , "case2_inspected_image.tif"),
             os.path.join(im_base_non_def , "case3_inspected_image.tif")]
    dst_path = os.path.join(BASE, 'output')
    cmds = [f"{PYTHON_PATH} {MAIN_SCRIPT}\
             --ref_image_path {ref_im}\
             --query_image_path {query_im}\
             --confidence_level {confidence_level}\
             --dst_path {dst_path}"\
             for ref_im, query_im in zip(ref, query)]

    for cmd in tqdm(cmds):
        print(f'run: {cmd}')
        exist_status = os.WEXITSTATUS(os.system(cmd))
        print(f'exit status: {exist_status}')

if __name__ == '__main__':
    fire.Fire(run_on_all_images)
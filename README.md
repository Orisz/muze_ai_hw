# muze_ai_hw
a coding interview given by the company

## Assumptions
1. I assumed that the miss alignment between the images originates from translation only. This assumptions was made based on inspection of the images and talking with the one assigned me this exercise.
2. I assumed the images were degraded using white additive Gaussian noise. A reasonable assumption with this type of problems.
3. I assumed the images are of the same size.

## Work Flow
1. I used opencv as the backbone package for this exercise.
2. Since I was told the defect can be within a size of a single pixel there is no point in denosing the images when trying to find the binary mask. This issue made things more challenging.
3. For the alignment part I first tried a SIFT approach following a RANSAC for the matching points. Although I tried several optimizers and feature extractors the images seems to be too noisy for that approach. Finally I turned to a template matching approach which in it's core uses cross correlation score. To make the results more robust to the already extreme noise the transformation was calculated on the Laplacian of Gaussian of the images.
4. Finally for the detection of the defects I used the absolute difference of the aligned images. Than I used the 'Median Absolute Deviation (MAD)' method to estimate the std of the query image. And finally, I utilized the white additive Gaussian noise assumption to derive a probabilistic scheme that given the desired confidence level and the estimated std decides on the threshold. This threshold in turn was used to acquire the binary mask of the query image.

## Results
* all the output images can be found in the 'output' folder. I supplied the results for two levels of confidence (95% and 99%).
* The masks are not ideal and one must consider which is more important to him, low False Negative or low False Positive. The higher the confidence level we seek the lower False positive occurrences we get (and vice versa for the False Negative). One can also plot the ROC curve the get a better understanding for the desired threshold.

## Discussion and possible improvements
* If we had many images of the same "pattern" (which we should have), we cloud have average many images together and have a better understanding of the added noise.
* Also since the sensor should be known to use (along with its specifications) we should have access to at least the first and second moment which would have made things easier.
* A possible explanations to the not ideal appearance of the masks is that the noise assumption is not acurate.
* A deep learning approach can be utilized e.g.:
    1. deploy a generative model that can train on a single image such as the CVPR best paper winner SinGAN by Rott Shaham et al. https://arxiv.org/pdf/1905.01164.pdf. This can provide us with many images that has the same defects as the ones we have in the given defected images. Than we can train a simple detection/segmentation model (mask rcnn, yolo, detr, etc.)
    2. If we had many images but only some are labeled we could have utilize some weakly supervised approach.

## Regenerate the results
from the root folder just run
```bash
python run_for_all_images.py <confidence_level>
```
where: confidence_level$\in[0,100]$ is an integer.

To run the core algorithm on some other images run:
```bash
python find_defects.py --ref_image_path <path to ref image> --query_image_path <path to query image> --confidence_level <desired confidence level> --dst_path <location (folder) to save the binary mask to>
```





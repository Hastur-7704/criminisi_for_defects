import argparse
import cv2
import numpy
import os

import criminisi
import utils

image_path = r'C:\Users\86139\Desktop\paperData\size_256\image\002.png'
mask_path = r'C:\Users\86139\Desktop\paperData\size_256\mask\002.png'
save_path = r'C:\Users\86139\Desktop\paperData\size_256\ablation\002.png'

patch_size = 9


# get image
image_origin = cv2.imread(image_path)
mask = cv2.imread(mask_path)

# preprocessing
image_masked = numpy.copy(image_origin).astype(float)
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if numpy.array_equal(mask[i][j], [255, 255, 255]):
            image_masked[i][j] = [0, 0, 0]
        image_masked[i][j] = utils.RGB2Lab(image_masked[i][j])
# cv2.imwrite(r'C:\Users\86139\Desktop\001.png', image_masked)

image_inpaint = criminisi.criminisi(image_masked, patch_size, alpha=2, beta=0)
for i in range(image_inpaint.shape[0]):
    for j in range(image_inpaint.shape[1]):
        image_inpaint[i][j] = utils.Lab2RGB(image_inpaint[i][j])
cv2.imwrite(save_path, image_inpaint)
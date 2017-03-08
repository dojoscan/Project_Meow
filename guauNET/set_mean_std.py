# CALCULATES MEAN AND STD IMAGE FOR A DATASET

import os
import parameters as p
import numpy as np
from PIL import Image

SAVE_PATH = 'C:/Master Chalmers/2 year/volvo thesis/code0/test'

def calc_mean_std(path_to_images):
    """
    Args:
        path_to_images: full path to image folder
    Returns:
        mean_image: average image across all images in path_to_images
        std_image: standard deviation across all images in path_to_images
    """

    mean_image = np.zeros((p.IMAGE_HEIGHT, p.IMAGE_WIDTH, 3))
    std_image = np.zeros((p.IMAGE_HEIGHT, p.IMAGE_WIDTH, 3))
    image_list = os.listdir(path_to_images)
    no_images = len(image_list)
    image_list = [path_to_images + s for s in image_list]
    for image in image_list:
        im = Image.open(image)
        # Image object is 4d, RGB are first 3 channels
        im = np.array(im.resize((p.IMAGE_WIDTH, p.IMAGE_HEIGHT), Image.BILINEAR))
        mean_image = np.add(im[:, :, 0:3], mean_image)
    mean_image = np.divide(mean_image, no_images)
    for image in image_list:
        im = Image.open(image)
        im = np.array(im.resize((p.IMAGE_WIDTH, p.IMAGE_HEIGHT), Image.BILINEAR))
        std_image += np.square(np.subtract(im[:, :, 0:3],mean_image))
    std_image = np.divide(std_image, no_images-1)
    std_image = np.sqrt(std_image)
    return mean_image, std_image

mean_image, std_image = calc_mean_std(p.PATH_TO_IMAGES)
with open(os.path.join(SAVE_PATH, 'KITTI_mean.txt'), 'wb') as temp_file:
    np.save(temp_file, mean_image)
with open(os.path.join(SAVE_PATH, 'KITTI_std.txt'), 'wb') as temp_file:
    np.save(temp_file, std_image)



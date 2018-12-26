#############################################
# Date          : 2018.03.25
# Programmer    : Seounggyu Kim
# description   : util 관련 함수
# Update Date   : 2018.04.04
# Update        : util 관련 함수
#############################################

import scipy.misc
import numpy as np
from six.moves import xrange
import tensorflow.contrib.slim as slim


############ get image ##################
def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64,crop=False):
    crop = False
    image = imread(image_path)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=False):
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


############# save image ############
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
  return (images+1.)/2.

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')
    return img

######## Get Image Size ################
def image_manifold_size(num_images):
  # manifold_h = int(np.floor(np.sqrt(num_images)))
  # manifold_w = int(np.ceil(np.sqrt(num_images)))
    manifold_h = int(num_images) // 4
    manifold_w = 4
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w

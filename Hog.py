import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure


def HOG(image_name):
    image = cv.imread(location + '/' + image_name, 0)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(6, 6),
                        cells_per_block=(2, 2), visualise=True)

    hog_image_rescaled = hog_image

    plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # plt.show()
    plt.imsave(location+'/HOG_'+image_name, hog_image_rescaled, cmap=plt.cm.gray)
    return hog_image_rescaled


import cv2 as cv
import os

location = './UCMerced_LandUse/training_images/runway'

for filename in os.listdir(location):
    if filename.endswith(".tif"):
        hog_of_image = HOG(filename)

# img = cv.imread(location + '/airplane/airplane41.tif', 0)
# # img = cv.imread(location + 'negative_examples/baseballdiamond17.tif', 0)
# # cv.imshow('grayscale image', img)
# hog_of_image = HOG(img)
#
# # # cv.imshow('image', hog_of_image)
# #
# # image = color.rgb2gray(data.astronaut())
# # HOG(image)
#
# cv.imwrite(location + '/HOG/airplane1.tif', hog_of_image)

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure

def HOG(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(6, 6),
                        cells_per_block=(2, 2), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    # hog_image_rescaled = hog_image

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    # plt.show()
    return hog_image_rescaled


import cv2 as cv
import os

# location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/Images/'
location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/training_images/negative_examples'

for filename in os.listdir(location):
    if filename.endswith(".tif"):
        hog_of_image = HOG(cv.imread(location + '/' + filename, 0))
        cv.imwrite(location + '/HOG_' + filename, hog_of_image)
        # print filename


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

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import exposure

import cv2 as cv
import os
import numpy as np
from sklearn import svm
import pickle

location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/training_images/negative_examples'
testing_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/HOG_Images/airplane'
base_dir = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/Images'
target_dir = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/HOG_Images'
fds_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/FD_vectors/'

def HOG(filename):
    image = cv.imread(location + '/' + filename, 0)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    # ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 255))
    # hog_image_rescaled = hog_image

    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # ax1.set_adjustable('box-forced')
    # plt.show()
    plt.imsave(location + '/../HOG/' + filename, hog_image_rescaled, cmap=plt.cm.gray)

    # return hog_image_rescaled


def save_HOG_of_all_images(directory):
    '''To make a new directory of HOG images'''

    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            image = cv.imread(directory + '/' + filename, 0)
            fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 255))

            index = directory.index("Images") + 6
            plt.imsave(target_dir + directory[index: ] + "/" + filename, hog_image_rescaled, cmap=plt.cm.gray)

        else:
            if not filename.startswith("."):
                os.mkdir(target_dir + "/" + filename)
                save_HOG_of_all_images(directory + '/' + filename)

def save_HOG_of_negative_examples():
    # To make HOG images
    for filename in os.listdir(location):
        if filename.endswith(".tif"):
            HOG(filename)

def train_svm():

    training_input = []
    training_output = []
    testing_input = []
    second_testing_input = np.empty((0, 256*256), int)

    # To read images for training
    for filename in os.listdir(location):
        if filename.endswith(".tif"):
            training_input += [cv.imread(location + '/../HOG/' + filename, 0)]
            if filename.startswith("airplane"):
                training_output += [1]
            else:
                training_output += [0]

    # training_input = get_training_set()
    # training_input = add_positive_training_example(training_input)
    # training_output = [0] * len(training_input)

    test_count = 0
    for filename in os.listdir(testing_location):
        if filename.endswith(".tif"):
            if cv.imread(testing_location + '/' + filename, 0).shape == (256, 256):
                # testing_input += [cv.imread(testing_location + '/' + filename, 0).reshape(1, -1)]
                second_testing_input = np.append(second_testing_input, cv.imread(testing_location + '/' + filename, 0).reshape(1, 256*256))
                test_count += 1

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    input = np.np.array(training_input)
    output = np.np.array(training_output)

    input = input.reshape(len(training_input), 256 * 256)
    second_testing_input = second_testing_input.reshape(test_count, -1)

    classifier.fit(input, output)
    predicted = classifier.predict(second_testing_input)
    print predicted

def get_positive_samples():
    list_of_fds = []
    testing_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/training_images/training_airplane_images/'
    for filename in os.listdir(testing_location):
        if filename.endswith(".tif"):
            image = cv.imread(testing_location + filename, 0)
            if image.shape != (256, 256):
                image = cv.resize(image, (256, 256))
            fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True, block_norm='L2-Hys')
            list_of_fds += [fd]

        # JUST ONE AIRPLANE IMAGE TO TRAIN
        break
    return list_of_fds

def get_negative_samples(object_name):
    list_of_fds = []
    images_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/Images'
    for filename in os.listdir(images_location + '/' + object_name):
        image = cv.imread(images_location + '/' + object_name + '/' + filename, 0)
        if image.shape != (256, 256):
            image = cv.resize(image, (256, 256))
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True,
                            block_norm='L2-Hys')
        list_of_fds += [fd]
    print "Directory " + object_name + " done."
    return list_of_fds


def save_fds():
    images_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/Images'
    for directoryname in os.listdir(images_location):
        if not directoryname.startswith("."):
            fds = get_negative_samples(directoryname)
            f = file(fds_location + directoryname, 'w')
            pickle.dump(fds, f)
            f.close()


def train_svm(positive_fds, negative_fds):
    training_vectors_input = positive_fds + negative_fds
    training_output = [1] * len(positive_fds) + [0] * len(negative_fds)
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001, probability=True)
    classifier.fit(training_vectors_input, training_output)
    return classifier

def get_scores_of_airplanes(classifier):
    classes = []
    probab = []
    testing_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/training_images/training_airplane_images/'
    for filename in os.listdir(testing_location):
        if filename.endswith(".tif"):
            image = cv.imread(testing_location + filename, 0)
            if image.shape != (256, 256):
                image = cv.resize(image, (256, 256))
            fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True,
                                block_norm='L2-Hys')
            predicted = classifier.predict([fd])
            probability = classifier.predict_proba([fd])
            probab += [probability]
            classes += [predicted]
    probabilities_with_indexes = list(enumerate(probab))
    sorted_scores = sorted(probabilities_with_indexes, reverse=True, key=lambda x: x[1][0][1])
    return sorted_scores

def load_fds():
    list_of_fds = []
    for filename in os.listdir(fds_location):
        if not filename.startswith("."):
            f = file(fds_location + filename, 'r')
            list_of_fds += pickle.load(f)
            f.close()
    return list_of_fds


def get_updated_positive_fds(sorted_scores):
    list_of_fds = []
    testing_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/training_images/training_airplane_images/'
    airplanes = os.listdir(testing_location)
    print "Selecting top airplanes ---- "
    for i in range(5):
        image = cv.imread(testing_location + airplanes[sorted_scores[i][0]], 0)
        print "selected: " + airplanes[sorted_scores[i][0]]
        if image.shape != (256, 256):
            image = cv.resize(image, (256, 256))
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True,
                            block_norm='L2-Hys')
        list_of_fds += [fd]
    return list_of_fds

import datetime
starttime = datetime.datetime.now()

# iteration 1
positive_fds = get_positive_samples()
negative_fds = load_fds()
svm_classifier = train_svm(positive_fds, negative_fds)
sorted_scores = get_scores_of_airplanes(svm_classifier)

# iteration 2
positive_fds = get_updated_positive_fds(sorted_scores)
svm_classifier = train_svm(positive_fds, negative_fds)
sorted_scores = get_scores_of_airplanes(svm_classifier)

# iteration 3
positive_fds = get_updated_positive_fds(sorted_scores)
svm_classifier = train_svm(positive_fds, negative_fds)
sorted_scores = get_scores_of_airplanes(svm_classifier)


# positive_fds = get_positive_samples()

endtime = datetime.datetime.now()
print "Time: " + (endtime - starttime)

# save_HOG_of_all_images(base_dir)
# save_HOG_of_negative_examples()
# train_svm()

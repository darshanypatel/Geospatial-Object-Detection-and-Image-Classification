import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import exposure

import cv2 as cv
import os
import numpy as np
from sklearn import svm
import pickle
import datetime

location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/training_images/negative_examples'
testing_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/HOG_Images/airplane'
base_dir = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/Images'
target_dir = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/HOG_Images'
fds_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/FD_vectors/'
seed_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/training_images/cropped_airplane_seeds/'

# setting up the default seed image shape
seed_name = 'airplane86.tif'
image = cv.imread(seed_location + seed_name, 0)
seed_shape = image.shape

# this is not used
def HOG(filename):
    image = cv.imread(location + '/' + filename, 0)
    # image = cv.imread(seed_location + seed_name, 0)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True,
                        block_norm='L2-Hys')

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

# this is not used
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

# this is not used
def save_HOG_of_negative_examples():
    # To make HOG images
    for filename in os.listdir(location):
        if filename.endswith(".tif"):
            HOG(filename)

# this is not used
# def train_svm():
#
#     training_input = []
#     training_output = []
#     testing_input = []
#     second_testing_input = np.empty((0, 256*256), int)
#
#     # To read images for training
#     for filename in os.listdir(location):
#         if filename.endswith(".tif"):
#             training_input += [cv.imread(location + '/../HOG/' + filename, 0)]
#             if filename.startswith("airplane"):
#                 training_output += [1]
#             else:
#                 training_output += [0]
#
#     # training_input = get_training_set()
#     # training_input = add_positive_training_example(training_input)
#     # training_output = [0] * len(training_input)
#
#     test_count = 0
#     for filename in os.listdir(testing_location):
#         if filename.endswith(".tif"):
#             if cv.imread(testing_location + '/' + filename, 0).shape == (256, 256):
#                 # testing_input += [cv.imread(testing_location + '/' + filename, 0).reshape(1, -1)]
#                 second_testing_input = np.append(second_testing_input, cv.imread(testing_location + '/' + filename, 0).reshape(1, 256*256))
#                 test_count += 1
#
#     # Create a classifier: a support vector classifier
#     classifier = svm.SVC(gamma=0.001)
#
#     input = np.np.array(training_input)
#     output = np.np.array(training_output)
#
#     input = input.reshape(len(training_input), 256 * 256)
#     second_testing_input = second_testing_input.reshape(test_count, -1)
#
#     classifier.fit(input, output)
#     predicted = classifier.predict(second_testing_input)
#     print predicted


def get_positive_seed():
    # returns the HOG vector of the seed image in a list

    list_of_fds = []
    global seed_name
    image = cv.imread(seed_location + seed_name, 0)

    fd2, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True,
                        block_norm='L2-Hys')
    list_of_fds += [fd2]

    return list_of_fds


def get_negative_samples(object_name):
    # returns a list of HOG vectors of negative samples

    global seed_shape

    list_of_fds = []
    images_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/Images'
    for filename in os.listdir(images_location + '/' + object_name):
        image = cv.imread(images_location + '/' + object_name + '/' + filename, 0)

        if image.shape != seed_shape:
            image = cv.resize(image, seed_shape)
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True,
                            block_norm='L2-Hys')
        list_of_fds += [fd]
    print "Directory " + object_name + " done."
    return list_of_fds


def save_fds():
    # saves HOG vectors to the filesystem

    images_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/Images'
    for directoryname in os.listdir(images_location):
        if not directoryname.startswith("."):
            fds = get_negative_samples(directoryname)
            f = file(fds_location + directoryname, 'w')
            pickle.dump(fds, f)
            f.close()


def train_svm(positive_fds, negative_fds):
    # returns a linear svm

    training_vectors_input = positive_fds + negative_fds
    training_output = [1] * len(positive_fds) + [0] * len(negative_fds)
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001, probability=True, kernel='linear')
    classifier.fit(training_vectors_input, training_output)
    return classifier


def get_scores_of_airplanes(classifier):
    # returns a list of probabilities of detecting an airplane in a image

    classes = []
    probab = []
    object_name = []

    global seed_shape

    testing_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/training_images/cropped_airplane_images/'
    for filename in os.listdir(testing_location):
        if filename.endswith(".tif"):
            image = cv.imread(testing_location + filename, 0)
            # if image.shape != (256, 256):
            #     image = cv.resize(image, (256, 256))

            if image.shape != seed_shape:
                image = cv.resize(image, seed_shape)

            fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True,
                                block_norm='L2-Hys')
            predicted = classifier.predict([fd])
            probability = classifier.predict_proba([fd])
            probab += [probability]
            classes += [predicted]
            object_name += [filename[:-4]]

    z = zip(object_name, probab)
    probabilities_with_indexes = list(enumerate(z))
    sorted_scores = sorted(probabilities_with_indexes, reverse=True, key=lambda x: x[1][1][0][1])
    return sorted_scores


def load_fds():
    # returns the saved HOG vectors

    list_of_fds = []
    for filename in os.listdir(fds_location):
        if not filename.startswith("."):
            f = file(fds_location + filename, 'r')
            list_of_fds += pickle.load(f)
            f.close()
    return list_of_fds


def get_updated_positive_fds(sorted_scores, k):
    # returns the updated positive samples

    list_of_fds = []

    global seed_shape

    testing_location = '/Users/darshanypatel/Desktop/Sem_3/Spatial and Temporal Data Mining/Project/Geospatial-Object-Detection-and-Image-Classification/UCMerced_LandUse/training_images/cropped_airplane_images/'
    airplanes = os.listdir(testing_location)
    print "Selecting top airplanes ---- "
    for i in range(k):
        image = cv.imread(testing_location + airplanes[sorted_scores[i][0]], 0)
        print "selected: " + airplanes[sorted_scores[i][0]][:-4]
        # if image.shape != (256, 256):
        #     image = cv.resize(image, (256, 256))

        if image.shape != seed_shape:
            image = cv.resize(image, seed_shape)

        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True,
                            block_norm='L2-Hys')
        list_of_fds += [fd]
    return list_of_fds


def check_results_for_negative_samples(classifier):
    # returns the probabilities of detecting the object in negative samples

    classes = []
    probab = []
    object_name = []

    for filename in os.listdir(location):
        if filename.endswith(".tif"):
            image = cv.imread(location + '/' + filename, 0)
            # if image.shape != (256, 256):
            #     image = cv.resize(image, (256, 256))

            if image.shape != seed_shape:
                image = cv.resize(image, seed_shape)

            fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True,
                                block_norm='L2-Hys')
            predicted = classifier.predict([fd])
            probability = classifier.predict_proba([fd])
            probab += [probability]
            classes += [predicted]
            object_name += [filename[:-4]]

    z = zip(object_name, probab)
    probabilities_with_indexes = list(enumerate(z))
    sorted_scores = sorted(probabilities_with_indexes, reverse=True, key=lambda x: x[1][1][0][1])
    return sorted_scores


starttime = datetime.datetime.now()

# saving the HOG vectors in the filesystem
save_fds()


# loading the saved HOG vectors from the filesystems
negative_fds = load_fds()

# number of iterations for refining the part detectors
k = 5

# iteration 1
positive_fds = get_positive_seed()
svm_classifier = train_svm(positive_fds, negative_fds)
sorted_scores = get_scores_of_airplanes(svm_classifier)
for i in sorted_scores:
    print i[1]

# iteration 2
positive_fds = get_updated_positive_fds(sorted_scores, k)
svm_classifier = train_svm(positive_fds, negative_fds)
sorted_scores = get_scores_of_airplanes(svm_classifier)
for i in sorted_scores:
    print i[1]

# iteration 3
positive_fds = get_updated_positive_fds(sorted_scores, k)
svm_classifier = train_svm(positive_fds, negative_fds)
sorted_scores = get_scores_of_airplanes(svm_classifier)
for i in sorted_scores:
    print i[1]


# Just checking what the classifier is giving probabilities for negative images
positive_fds = get_updated_positive_fds(sorted_scores, k)
sorted_scores = check_results_for_negative_samples(svm_classifier)
for i in sorted_scores:
    print i[1]


# Saving the classifier
f = file(location + '/../svm_classifier_object', 'w')
pickle.dump(svm_classifier, f)


endtime = datetime.datetime.now()
print "Time: "
print endtime - starttime

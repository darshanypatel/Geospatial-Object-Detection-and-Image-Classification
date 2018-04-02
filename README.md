# Geospatial-Object-Detection-and-Image-Classification

The reference paper for this project is [here](https://github.com/kira0992/Geospatial-Object-Detection-and-Image-Classification/blob/master/isprs-COPD.pdf)

The dataset used for object classification is [here](https://github.com/kira0992/Geospatial-Object-Detection-and-Image-Classification/tree/master/UCMerced_LandUse/Images)

### Procedure to follow for implementation:

1. We cropped the images of objects which we want to detect for removing the surroundings and get only the objects. The cropped images are [here](https://github.com/kira0992/Geospatial-Object-Detection-and-Image-Classification/tree/master/UCMerced_LandUse/training_images/cropped_airplane_images).

2. Select one image as seed for every object you want to detect: Taking example of an airplane, we used the below image of airplane as our seed image.

![airplane86](https://user-images.githubusercontent.com/8282522/38204868-859b403e-3672-11e8-880d-6b4747ec51d5.jpg)

3. Create Histogram of Oriented Gradients(HOG) of every image using [Dalal and Triggs](https://github.com/kira0992/Geospatial-Object-Detection-and-Image-Classification/blob/master/Reference%20Papers/Dalal%20and%20Triggs.pdf) reference paper. You can also follow [this](https://www.learnopencv.com/histogram-of-oriented-gradients/) technique for creating HOGs. We have used the parameters for creating a HOG which works best for our referenced paper:- 

`orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True, block_norm='L2-Hys'`

The HOG of our seed image is as shown below -

![seed_hog](https://user-images.githubusercontent.com/8282522/38204757-27534990-3672-11e8-8581-0cff4a521cc3.png)

4. Build a linear SVM using the HOG feature vector of the seed image as positive example and all the other images which does not have the target object in them as the negative examples. Note: we have resized all the images to the size of our seed image in order to keep the HOG feature vector of equal dimension.

5. Get top 5 predictions of the target object using the above SVM and build a new model using these 5 images as positive examples for training.

6. Repeat this process multiple times to get final set of 5 positive examples. We have used 3 iterations as it gives the maximum accuracy according to the reference paper.

7. Scan through the satellite image with a window size same as size of the objects in positive set. Make a pyramid of this satellite image and scan for objects in all those levels of the satellite image. Make HOG feature vectors of this sliding window and give it to the SVM to get the prediction of that window. We have used 1.25 scale to reduce the image into smaller sizes.

9. Perform non-maximum suppression (remove more than 50% overlapping detections) to get the results from the image.

Result on using this airplane LinearSVM classifier on the below image:

![original image](https://user-images.githubusercontent.com/8282522/38206855-f39b45b4-3679-11e8-9e2c-e7c997b1a189.png)

Image before removing the overlapping detections:

![before removing overlapping detections](https://user-images.githubusercontent.com/8282522/38205760-b9909a76-3675-11e8-9396-2946c6b247bf.png)

Image after removing the overlapping detections:

![after removing overlapping detections](https://user-images.githubusercontent.com/8282522/38205778-c958e27e-3675-11e8-9972-55eff065d8c1.png)


### Libraries Used:

1. OpenCV
2. skimage
3. numpy
4. pickle
5. sklearn.svm

### Details of the python code:

#### Hog.py:

This code uses one seed image of an airplane and follows the procedure given above to train a classifier and stores the classifier object in a file.

#### Object_detection.py:

This code uses the classifier stored from the above code and predicts the object locations using a sliding window. Then it removes the repeated detections of a single object using Non-Maximum Suppression.

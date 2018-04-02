# Geospatial-Object-Detection-and-Image-Classification

The reference paper for this project is [here](https://github.com/kira0992/Geospatial-Object-Detection-and-Image-Classification/blob/master/isprs-COPD.pdf)

DATASET USED FOR OBJECT CLASSIFICATION : [here](https://github.com/kira0992/Geospatial-Object-Detection-and-Image-Classification/tree/master/UCMerced_LandUse/Images)

Procedute to follow for implementation:
1. Select one image as seed for every object:
  Taking example of an airplane we cropped the image from it's surroundings to get only the airplane. [cropped images]( )

2. Create HOG of every image using [Dalal and Triggs](http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=1467360) references


3. Following [this](https://www.learnopencv.com/histogram-of-oriented-gradients/) technique for creating HOGs

4. Build a linear SVM using the seed images as positive examples in HOG feature Space and other images as the negative examples.

5. Get top 5 predictions using the above SVM and build a new model using these 5 images as positive examples for training.

6. Repeat the process again to get final set of 5 positive examples.

7. Scan through the satellite image with a window size same as size of the objects in positive set.

8. Detect all matching objects in the satellite images

9. Perform non-maxima suppression to get the top results from the image.


Libraries Used:

openCV
skimage. -- HOG
numpy
pickle
sklearn.svm


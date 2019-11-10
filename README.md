# SVM-for-face-detection
The face detection model is constructed by implementing support vector machines. 
SVM is trained using hard negative mining algorithm to recognize face features.
Raw image intensity values are not robust features for classification. Hence Histogram of Oriented Gradient (HOG) is used as image features. HOG uses the gradient information instead of intensities, and this is more robust to changes in color and illumination conditions.
VLFeat library is used to do HOG calclation. Th coding is done in Matlab.

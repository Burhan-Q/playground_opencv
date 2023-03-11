'''Python function that calculates feature similarity for images of any size'''

import numpy as np
from scipy.spatial import distance
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import cv2


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    '''Function that returns HOG features and visualization (optionally)'''

    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=False, feature_vector=feature_vec)
        return features

    
def binSpatial(img, size=(32, 32)):
    '''Function that computes binned color features'''

    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()

    return np.hstack((color1, color2, color3))

    
def colorHist(img, nbins=32):    #bins range from 0 to 256 
    '''Function that computes color histogram features'''

    # Compute the histogram of the color channels separately 
    channel1Hist = np.histogram(img[:,:,0], bins=nbins)  #channels are in RGB order 
    channel2Hist = np.histogram(img[:,:,1], bins=nbins)  #channels are in RGB order 
    channel3Hist = np.histogram(img[:,:,2], bins=nbins)  #channels are in RGB order 

    # Concatenate the histograms into a single feature vector 
    histFeatures = np.concatenate((channel1Hist[0], channel2Hist[0], channel3Hist[0])) 

    return histFeatures

    
def extractFeatures(imgs):   #extracts all features for all images in imgs list 

    # Create a list to append feature vectors to 
    features = []  

    # Iterate through the list of images 
    for file in imgs:  

        # Read in each one by one 
        image = mpimg.imread(file)  

        # apply color conversion if other than 'RGB' 
        if cspace != 'RGB':  

            if cspace == 'HSV':  

                featureImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  

            elif cspace == 'LUV':  

                featureImage = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)  

            elif cspace == 'HLS':  

                featureImage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  

            elif cspace == 'YUV':  

                featureImage = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  

            elif cspace == 'YCrCb':  

                featureImage = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)  

        else: featureImage = np.copy(image)      

        spatialFeatures = binSpatial(featureImage)   #get spatial features from image 

        histFeatures = colorHist(featureImage)       #get histogram features from image 

        hogFeatures = getHogFeatures(featureImage)   #get HOG features from image 

        # Append the new feature vector to the features list 
        features.append(np.concatenate((spatialFeatures, histFeatures)))
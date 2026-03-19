# Imports
import os
import sys
import time
import copy
import numpy as np
import cv2 as cv
import glob
from matplotlib import pylab as plt
from math import pi, sqrt
import math

import pandas as pd



# Settings
thetaStart = 0
thetaEnd = 45
thetaStep = 5






#-------------------------------------------------------------------------------
#                             reconstruction_image
#-------------------------------------------------------------------------------
class reconstruction_image(object):

    #---------------------------------------------------------------------------
    #                              __init__
    #---------------------------------------------------------------------------
    def __init__(self, filePath):

        # Convert to an absolute path
        filePath = os.path.abspath(filePath)

        # Check to see if the path exist
        if os.path.exists(filePath) == False:
            raise Exception(f'ERROR: File not found ({filePath})')
        
        # Make sure the path is a file
        if os.path.isfile(filePath) == False:
            raise Exception(f'ERROR: Path needs to be to an image file ({filePath})')

        # Save the file path
        self.filePath = filePath

        # Load the image
        self._load_image()

        # Detect the features within the image
        self._detect_features()

        return


    #---------------------------------------------------------------------------
    #                              _load_image
    #---------------------------------------------------------------------------
    def _load_image(self):
        self.pic = cv.imread(self.filePath)

        # Convert to grayscale
        self.pic_gray = cv.cvtColor(self.pic, cv.COLOR_BGR2GRAY)

        # Get the size of the images (they are the same)
        self.h,self.w = np.shape(self.pic_gray)
        
        return


    #---------------------------------------------------------------------------
    #                           _detect_features
    #---------------------------------------------------------------------------
    def _detect_features(self):

        # TODO:  Parameterize these features... for now, hard code...

        # use SIFT to detect features:  https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html
        numFeatures = 10000         # default = 0
        numOctaveLayers = 75        # default = 3.  Higher numbers seem to increase the number of detected features
        contrastThreshold = 0.04    # default = 0.04
        edgeThreshold = 10          # default = 10
        sigma = 1.6                 # default = 1.6

        sift = cv.SIFT_create(nfeatures=numFeatures, nOctaveLayers=numOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)
        self.kp, self.desc = sift.detectAndCompute(self.pic_gray, None)

        return
    



#-------------------------------------------------------------------------------
#                                  dist
#-------------------------------------------------------------------------------
def dist(pt1, pt2):
    ''' Helper function to measure the distance between two points. '''
    x1 = pt1[0]
    y1 = pt1[1]

    x2 = pt2[0]
    y2 = pt2[1]

    d = sqrt((x1-x2)**2 + (y2-y1)**2)
    return d


#-------------------------------------------------------------------------------
#                                y_dist
#-------------------------------------------------------------------------------
def y_dist(pt1, pt2):
    ''' Helper function to measure the y-distance between two points '''
    return abs(pt2[1] - pt1[1])


#-------------------------------------------------------------------------------
#                             match_features
#-------------------------------------------------------------------------------
def match_features(reconImg1, reconImg2, yMovementThreshold=10, xyMinMovementThreshold=0):

    '''
    * reconImg1 - an object of the "reconstruction_image" class.
    * reconImg2 - an object of the "reconstruction_image" class.
    * yMovementThreshold - maximum number of pixels the y-coordinate can change
                           between the two reconstruction images in order to be
                           considered valid.  (ideally, this number is 0)
                           [default = 2]
    * xyMinMovementThreshold - Minimim number of pixels a coordinate should have 
                           moved (combined x/y distance)
                           [default = 5]
    '''

    # FLANN parameters (SIFT)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100) # or pass empty dictionarys

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(reconImg1.desc, reconImg2.desc, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    goodMatches = 0

    distances = []

    for i,(m,n) in enumerate(matches):
        # This is the "hamming distance"
        if m.distance < 0.7*n.distance:    # <--- This threshold impacts the results. Investigate it.
            # The match is good
            index1 = m.queryIdx
            index2 = m.trainIdx

            try:
                pt1 = reconImg1.kp[index1].pt
                pt2 = reconImg2.kp[index2].pt
            except:
                print(index1, index2)
                continue

            # The y-distance shouldn't have moved.  Compare it and see how close it is.
            if y_dist(pt1, pt2) < yMovementThreshold:
                
                # Find the total distance the pixel moved
                d = dist(pt1, pt2)

                distances.append(d)

                if d >= xyMinMovementThreshold:

                    # The points should be valid...
                    x1.append(pt1[0])
                    y1.append(pt1[1])

                    x2.append(pt2[0])
                    y2.append(pt2[1])

                    # Keep track of the good matches
                    matchesMask[i]=[1,0]
                    goodMatches += 1


    print(f'Found {goodMatches} good matches...')


    # Compute the mean & standard deviation of the distances moved
    std = np.std(distances)
    mean = np.mean(distances)

    # Remove matches that move too far (mean + 1 std dev)
    numRemoved = 0
    removeIndexes = []
    for i,d in enumerate(distances):
        if (d == 0) or (d > (mean + std)):
            removeIndexes.append(i)
            numRemoved +=1

    for i in sorted(removeIndexes, reverse=True):
        del x1[i]
        del x2[i]
        del y1[i]
        del y2[i]

    print(f'Removed {numRemoved}...')


    # Save the points into a numpy vector
    x1 = np.matrix(x1).T
    y1 = np.matrix(y1).T
    x2 = np.matrix(x2).T
    y2 = np.matrix(y2).T

    initialPoints = np.hstack((x1, y1))
    finalPoints = np.hstack((x2, y2))


    



    #fig = plt.figure()
    #plt.plot(distances)
    #counts, bins = np.histogram(distances)
    #plt.stairs(counts, bins)


    plt.show()

    return (initialPoints, finalPoints)


#-------------------------------------------------------------------------------
#                             reconstruct_depth
#-------------------------------------------------------------------------------
def reconstruct_depth(initialPoints, finalPoints, rotationAmount):
    '''
    * initialPoints -  A numpy matrix with the initial values. These values are
                       the (x,y) coordinates for features matched between two
                       images. These coordinates are for the feature in the 
                       first image.
    * finalPoints -    This is similar to the initialPoints, but after the shape
                       has been rotated by some amount.
    * rotationAmount - this is the amount that the points were rotated, in
                       degrees.

    !!! ASSUMPTION:  This assumes perfect rotation along the [0, 1, 0] vector
                     (rotating about the y axis). Future versions of this function
                     should take the axis of rotation into account.
    '''

    x1 = initialPoints[:,0]
    y1 = initialPoints[:,1]
    x2 = finalPoints[:,0]
    y2 = finalPoints[:,1]
    theta = rotationAmount * pi / 180.0

    z1 = (x2 - x1 * np.cos(theta)) / np.sin(theta)
    z2 = -x1 * np.sin(theta) + z1 * np.cos(theta)
    
    # invert z2...   
    #z2 *= -1

    x2 -= 350
    z2 += 50


    # Invert the y axis
    y2 = 640 - y2

    coords = np.hstack((x2, y2, z2))


    # For some reason ours is rotated by 180 degrees
    coords = rotate_3D(coords, 180)
    
    # Scale down the coordinates (the actual bunny object is of the order 0.01 not 500)
    coords /= 3300

    return coords


#-------------------------------------------------------------------------------
#                             structure_from_motion
#-------------------------------------------------------------------------------
def structure_from_motion(initialPoints, finalPoints, fieldOfView=25.0, w=800, h=600):

    fov = math.radians(fieldOfView)  # The field of view used to capture the images in opengl
    AR = w / h                # The aspect ratio of the viewport (image width / image height)

    # Calculate the focal length
    f = 1 / math.tan(fov / 2)

    # Calculate the optical centers
    cx = AR / 2
    cy = 0.5

    # Create the 3x3 camera matrix
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    # Find the essential matrix
    E, mask = cv.findEssentialMat(initialPoints, finalPoints, K, cv.RANSAC, 0.999, 1.0)

    matchesMask = mask.ravel().tolist()

    # Reover the pose
    points, R, t, mask = cv.recoverPose(E, initialPoints, finalPoints)

    #print(R)
    #print(t)

    # TEST!!!  Overwrite the rotation matrix with the fixed one...
    #theta = thetaStep
    #R = np.matrix([[np.cos(theta),     0,                  np.sin(theta)],\
    #               [0,                 1,                  0],\
    #               [-np.sin(theta),    0,                  np.cos(theta)]])
    #t = np.zeros((3, 1))
    #t[0,0] = -1


    #calculate projection matrix for both camera
    M_r = np.hstack((R, t))
    M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))  # Assume no rotation/traslation for the "left" "eye"

    P_l = np.dot(K,  M_l)
    P_r = np.dot(K,  M_r)

    # undistort points
    p1 = initialPoints[np.asarray(matchesMask)==1,:]
    p2 = finalPoints[np.asarray(matchesMask)==1,:]

    p1_un = cv.undistortPoints(p1, K, None)
    p2_un = cv.undistortPoints(p2, K, None)
    p1_un = np.squeeze(p1_un)
    p2_un = np.squeeze(p2_un)

    #triangulate points this requires points in normalized coordinate
    point_4d_hom = cv.triangulatePoints(P_l, P_r, p1_un.T, p2_un.T)
    point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_3d[:3, :].T

    return point_3d


#-------------------------------------------------------------------------------
#                                  rotate_3D
#-------------------------------------------------------------------------------
def rotate_3D(points=None, rotation=0, axis=(0,1,0)):
    ''' rotate points in 3D.  rotation specified in degrees. '''
    # https://en.wikipedia.org/wiki/Rotation_matrix

    # TODO: incorporate axis of rotation...

    theta = rotation * pi / 180.0

    Rx = np.matrix([[1,                 0,                  0],\
                    [0,                 np.cos(theta),      -np.sin(theta)],\
                    [0,                 np.sin(theta),      np.cos(theta)]])


    Ry = np.matrix([[np.cos(theta),     0,                  np.sin(theta)],\
                    [0,                 1,                  0],\
                    [-np.sin(theta),    0,                  np.cos(theta)]])


    Rz = np.matrix([[np.cos(theta),     -np.sin(theta),     0],\
                    [np.sin(theta),     np.cos(theta),      0],\
                    [0,                 0,                  1]])


    newPoints = Ry * points.T

    return newPoints.T



#-------------------------------------------------------------------------------
#                                   main
#-------------------------------------------------------------------------------
def main():

    bunny_df = pd.DataFrame(columns = ['x', 'y', 'z'])

    images = []

    for i, degree in enumerate(range(thetaStart, thetaEnd+thetaStep, thetaStep)):
        
        # Define the path to the file
        filePath = f'C:/Dropbox (Meta)/Jupyter/Crystal_Rotation/bunny_pics/Bunny_{degree : 04d}deg.jpg'

        # Print a status update
        print(f'Processing [{i}]: {degree} degrees ({filePath})...')

        # Load the file
        img = reconstruction_image(filePath)
        images.append(img)

        if i == 0:
            continue
        elif i > 1:
            # Remove the first one
            images.pop(0)

        # Match features
        initialPoints,finalPoints = match_features(images[0], images[1])

        # Reconstruct the depth info given the 5 degree rotation
        coords = reconstruct_depth(initialPoints, finalPoints, thetaStep)
        #coords = structure_from_motion(initialPoints, finalPoints)

        # Rotate the coordinates before storing
        #coords = rotate_3D(coords, degree)             # This doesn't seem to work.  It may be because there is some translation that needs to happen.
        #coords = rotate_3D(coords, thetaStep)
        #coords = rotate_3D(coords, degree - thetaStep)

        # Store the coordinates into a dataframe
        df = pd.DataFrame(data=coords.astype(float))
        df.columns = ['x', 'y', 'z']
        
        # Add to the larger dataframe
        bunny_df = pd.concat((bunny_df, df))

        #if i == 3:
        #    break


    # Save the dataframe to a file
    #bunny_df.to_csv('coords_0deg_5deg.csv', sep=',', header=True, float_format='%.3f', index=False)
    bunny_df.to_csv('Crystal_Rotation/full_bunny.csv', sep=',', header=True, float_format='%.3f', index=False)


    # Get the coordates to plot    
    x = bunny_df['x']
    y = bunny_df['y']
    z = bunny_df['z']

    # Define the plot area
    plotArea = 5

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(coords[:,0], coords[:,2], coords[:,1])
    ax.scatter(x, z, y, s=plotArea)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.axis('image')

    #ax.axes.set_xlim3d(left=0, right=800) 
    #ax.axes.set_ylim3d(bottom=0, top=100) 
    #ax.axes.set_zlim3d(bottom=0, top=640)

    plt.show()

    return


if __name__ == '__main__':
    main()

# End of File

#!/usr/local/bin/python3
'''
File Name:      undistort.py
Author:         Jon Peterson - jrp@meta.com
Description:    This file takes as input an image and a camera calibration file
                (distortion values and camera matrix). It then undistorts the
                image.
'''

# Imports
import os
import sys
from glob import glob
from optparse import OptionParser

# Imports (3rd party packages - pip3 install numpy, opencv-python, matplotlib)
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



#-------------------------------------------------------------------------------
#                               _load_camera_cal 
#-------------------------------------------------------------------------------
def _load_camera_cal(path):
    ''' "Internal" helper function. '''

    mtx = []
    dist = []
    processingMtx = True

    file = open(path, 'r')

    for line in file:
        line = line.strip()

        if len(line) == 0:
            processingMtx = False
            continue

        cols = line.split(',')

        if processingMtx == True:
            mtx.append(cols)  
        else:
            dist.append(cols)

    file.close()

    mtx = np.asarray(mtx, dtype=np.float32)
    dist = np.asarray(dist, dtype=np.float32)

    return (mtx, dist)


#-------------------------------------------------------------------------------
#                                 undistort
#-------------------------------------------------------------------------------
def undistort(imagePath='image.png', pathToCameraCalibration='camera_calibration.csv', outputImagePath='undistorted.png', debug=False):

    # Get the camera calibration
    mtx,dist = _load_camera_cal(pathToCameraCalibration)

    if debug == True:
        print(f'{mtx=}')
        print(f'{dist=}')

    # Load the image
    img = cv.imread(imagePath)
    height,width,channels = np.shape(img)

    # Create a new camera matrix
    newcameramatrix, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

    # Undistort the image
    undistorted_image = cv.undistort(img, mtx, dist, None, newcameramatrix)

    # Write the output to a file
    cv.imwrite(outputImagePath, undistorted_image)

    if debug == True:
        cv.imshow(f'{imagePath} --> Undistorted', undistorted_image)
        cv.waitKey(1000)

    return outputImagePath


#-------------------------------------------------------------------------------
#                                    main
#-------------------------------------------------------------------------------
def main():

    # Parse the command-line parameters
    parser = OptionParser()
    parser.add_option("-i", "--image", dest="imagePath", help="path to the image file to undistort")
    parser.add_option("-f", "--file", dest="calFilePath", help="path to the camera calibration file.")
    parser.add_option("-o", "--output", dest="outputImagePath", help="path to save the undistorted image to")
    parser.add_option("-d", "--debug", dest="debug", action="store_true", help="enable debug output, such as plots")
    (options, args) = parser.parse_args()

    # Make sure a path was specified
    if options.imagePath is None:
        print(f'ERROR: Please specify a path to an image file to undistort.\ne.g. >> python3 {__file__} -i distortedImage.png -f cal_file.csv')
        sys.exit()

    outputPath = undistort(options.imagePath, options.calFilePath, options.outputImagePath, options.debug)
    print(outputPath)

    return

if __name__ == '__main__':
    main()

# End of File

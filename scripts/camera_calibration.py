#!/usr/bin/local/python3
'''
File Name:      camera_calibration.py
Author:         Jon Peterson - jrp@meta.com
Description:    This file accepts a folder path with pictures of a checkerboard calibration
                target, then outputs a calibration file.
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
#                                calibrate_camera
#-------------------------------------------------------------------------------
def calibrate_camera(folderPath='folder path', calFilePath='camera_calibration.csv', debug=False):
    '''
    Function Name:      calibrate_camera
    Arguments:          1. folderPath [string] - a path to the location of the 
                            calibration images. There should be at least one
                            image in the folder, preferably many images with 
                            different perspectives (e.g. move the checkerboard)
                            around.

                        2. calFilePath [string/None] - if specified, this should
                            be a path to the output CSV file. This is where the
                            function will store the camera matrix (3x3 array) and 
                            the distortion coefficients (5 values). If no path is 
                            specified, it will default to 'camera_calibration.csv'.

                        3. debug [boolean] - if this is True, then debug print 
                            statements and OpenCV visualizations will be used. 
                            This can be helpful to see if the calibration is 
                            working as expected.
    
    Return Values:      1. string - the path to the calibration CSV file.
    '''

    # Get the path to the calibration images
    absPath = os.path.abspath(folderPath)

    # Make sure the folder path exists
    if (os.path.exists(absPath) == False) or (os.path.isfile(absPath) == True):
        raise Exception(f'ERROR: Path is incorrect. Must be a valid path to a folder which contains image files. ({absPath})')

    # Get a list of all files in the folder
    supportedExtensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
    imageFiles = []
    for ext in supportedExtensions:
        files = glob(f'{absPath}/{ext}')
        imageFiles += files

    # Make sure there were some files
    if len(imageFiles) == 0:
        raise Exception('ERROR: folder path needs to include at least one image file... preferably many image files.')

    # Get the path to the output file
    if calFilePath is None:
        outputFilePath = 'camera_calibration.csv'
    else:
        outputFilePath = calFilePath

    # Sort the files
    imageFiles.sort()

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Define the number of rows and columns in the checkerboard
    numRows = 9
    numCols = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((numRows * numCols,3), np.float32)
    objp[:,:2] = np.mgrid[0:numRows,0:numCols].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Load each file and contribute toward the calibration
    for imageNum,imageName in enumerate(imageFiles):

        if debug == True:
            print(f'Calibrating "{imageName}" [{imageNum+1} / {len(imageFiles)}]...')

        # Load the calibration image
        img = cv.imread(imageName)

        # Convert the image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        try:
            ret, corners = cv.findChessboardCorners(gray, (numRows,numCols), None)
        except:
            print(f'ERROR Processing "{imageName}" (findChessboardCorners). Skipping file.')
            continue

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            try:
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            except:
                print(f'ERROR Processing "{imageName}" (cornerSubPix). Skipping file.')
                continue
            
            imgpoints.append(corners2)

            if debug == True:
                try:
                    # Draw and display the corners
                    cv.drawChessboardCorners(img, (numRows,numCols), corners2, ret)
                    cv.imshow(imageName, img)
                    cv.waitKey(1000)
                except:
                    print(f'ERROR drawing chessboard for "{imageName}"')

        if debug == True:
            cv.destroyAllWindows()


    # Define the camera matrix
    mtx = None

    # Define the distortion coeffs
    dist = None

    # Run the calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Convert numpy matricies to strings
    s1 = np.array2string(mtx, precision=6, separator=',')
    s2 = np.array2string(dist, precision=6, separator=',')
    
    # Remove the brackets [,], and space
    s1 = s1.replace('[', '')
    s1 = s1.replace('],', '')
    s1 = s1.replace(']', '')
    s1 = s1.replace(' ', '')

    s2 = s2.replace('[', '')
    s2 = s2.replace('],', '')
    s2 = s2.replace(']', '')
    s2 = s2.replace(' ', '')

    # Write the calibration to the file
    file = open(outputFilePath, 'w')
    file.write(f'{s1}\n\n{s2}')
    file.close()

    if debug == True:
        print('Camera Matrix:')
        print(mtx)

        print('\nDistortion Coefficients:')
        print(dist)

    return outputFilePath


#-------------------------------------------------------------------------------
#                                    main
#-------------------------------------------------------------------------------
def main():

    # Parse the command-line parameters
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="folderPath", help="path to folder with calibration images")
    parser.add_option("-f", "--file", dest="calFilePath", help="path to write the calibration output file to")
    parser.add_option("-d", "--debug", dest="debug", action="store_true", help="enable debug output, such as plots")
    (options, args) = parser.parse_args()

    # Make sure a path was specified
    if options.folderPath is None:
        print(f'ERROR: Please specify a path to the calibration images.\ne.g. >> python3 {__file__} -p calibrationFolder')
        sys.exit()

    # Calibrate the camera
    calFilePath = calibrate_camera(options.folderPath, options.calFilePath, options.debug)
    print(f'{calFilePath}')

    return

if __name__ == '__main__':
    main()

# End of File

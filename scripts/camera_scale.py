#!/usr/local/bin/python3
'''
File Name:      camera_scale.py
Author:         Jon Peterson - jrp@meta.com
Description:    This file gets the scale factor between an undistorted checkerboard (grid) and
                the camera.  This provides the real-world distance relationship from pixels to
                millimeters.
'''


# Imports
import os
import sys
from glob import glob
from optparse import OptionParser
import copy

# Imports (3rd party packages - pip3 install numpy, opencv-python, matplotlib)
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt





#-------------------------------------------------------------------------------
#                                _draw_cube
#-------------------------------------------------------------------------------
def _draw_cube(drawImg, corners, imgpts):
    '''
    Internal helper function for debug...
    '''

    # Open CV uses "BGR" instead of "RGB"
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)

    lineThickness = 5

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    basePoints = np.array([imgpts[:4]], dtype=int)
    drawImg = cv.drawContours(drawImg, basePoints, -1, green, -3)
    
    
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        pt1 = np.array(imgpts[i], dtype=int)
        pt2 = np.array(imgpts[j], dtype=int)
        drawImg = cv.line(drawImg, pt1, pt2, blue, lineThickness)
    
    # draw top layer in red color
    topPoints = np.array([imgpts[4:]], dtype=int)
    drawImg = cv.drawContours(drawImg, topPoints, -1, red, lineThickness)

    return drawImg


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
#                                 get_scale
#-------------------------------------------------------------------------------
def get_scale(imagePath='image.png', pathToCameraCalibration='camera_calibration.csv', debug=False, numRows=6, numCols=9, box_size=8.0):
    '''
    * imagePath [string] - path to an undistorted image which includes the checkerboard pattern.
    * pathToCameraCalibration [string] - path to the camera calibration CSV file.
    * debug [boolean] - Outputs debug information or not...

    * numRows [int] - number of rows in the checkerboard pattern
    * numCols [int] - number of cols in the checkerboard pattern
    * box_size [float] - the size of the checkerboard box, in millimeters

    * Returns (float):  scale in pixels per millimeter
    '''

    # Load the undistorted image
    img = cv.imread(imagePath)
    height,width,channels = np.shape(img)

    # Get the camera calibration
    mtx,dist = _load_camera_cal(pathToCameraCalibration)

    if debug == True:
        print(f'{mtx=}')
        print(f'{dist=}')

    # Create a new camera matrix
    newcameramatrix, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    mtx = newcameramatrix

    # Make sure the image is grayscale
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the checkerboard
    ret, corners = cv.findChessboardCorners(grayImg, (numRows, numCols), None)

    # Overwrite the distortion matrix, since this is an undistorted image
    dist[dist != 0] = 0


    if ret == True:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Define the checkerboard object
        objp = np.zeros((numRows * numCols, 3), np.float32)
        objp[:,:2] = np.mgrid[0:numRows,0:numCols].T.reshape(-1,2)
        
        # Scale to millimeters (ensures the "solvePnP" algorithm returns values in real-world units)
        objp *= box_size

        # Axis for the cube (this is drawn when debug is True)
        axisLength = 10  # millimeters
        axis = np.float32([[0,0,0], [0,axisLength,0], [axisLength,axisLength,0], [axisLength,0,0], [0,0,-axisLength], [0,axisLength,-axisLength], [axisLength,axisLength,-axisLength], [axisLength,0,-axisLength]])

        # Refine the position estimate
        corners2 = cv.cornerSubPix(grayImg, corners, (11,11), (-1,-1), criteria)

        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)


        # Create a 3x4 rotation / translation matrix
        rmat,jac = cv.Rodrigues(rvecs)
        RT = np.hstack((rmat, tvecs))

        # Get the camera matrix
        K = newcameramatrix

        if debug == True:
            print(f'{K=}')
            print(f'{RT=}')

        # Create a point in the real world
        p1 = np.matrix([0,0,0,1]).T
        p2 = np.matrix([1,1,0,1]).T

        # Scale a real-world point to a camera-world point
        c1 = K * (RT * p1)
        #c2 = K * (RT * p2)

        s = c1[2,0]
        u = c1[0,0] / s
        v = c1[1,0] / s
        
        # https://en.wikipedia.org/wiki/Perspective-n-Point
        if debug == True:
            print(f'{s=}\t{u=}\t{v=}')

        # This is a check...
        #s2 = c2[2,0]
        #u2 = c2[0,0] / s2
        #v2 = c2[1,0] / s2
        #print(f'{s2=}\t{u2=}\t{v2=}')
        #print(u2 - u)
        #print(v2 - v)

        # Define coordinates in the pixel space (0,0) and (1,1) and see where that would fall in the real world (in millimeters)
        c1 = np.linalg.pinv(RT) * (np.linalg.pinv(K) * np.matrix([0,0,s]).T)
        c2 = np.linalg.pinv(RT) * (np.linalg.pinv(K) * np.matrix([s,s,s]).T)

        # Get the distance (diagonal line)
        d = np.linalg.norm(c2-c1)

        # Divide by sqrt(2) since this is the hypoteneus (get x-scale / y-scale)
        scale = d / np.sqrt(2)

        # Convert to pixels per millimeter instead of mm/px
        scale = 1/scale

        '''
        # Get the reference distance (in checkerboard boxes)
        p1 = (imgpts[0] / axisLength)[0]
        p2 = (imgpts[2] / axisLength)[0]

        # Compute the distance in X and Y directions (distance in pixels)
        d_x = abs(p2[0] - p1[0])
        d_y = abs(p2[1] - p1[1])

        # Compute the scale (pixels per millimeter)
        #scale_x = d_x / size_x
        #scale_y = d_y / size_y
        scale_x = d_x
        scale_y = d_y
        '''

        if debug == True:
            # Draw the red/gree/blue lines indicating where the grid is
            img2 = _draw_cube(img, corners2, imgpts)

            cv.imshow('img',img2)
            cv.waitKey(2000)
            
            #cv.waitKey(0)
            #cv.destroyAllWindows()

            # OpenCV stores images in BGR order instead of RGB, so convert
            #fig = plt.figure()
            #plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
            #plt.show()

    else:
        print("ERROR:  Couldn't find the chessboard...")
        scale = None


    return scale


#-------------------------------------------------------------------------------
#                                    main
#-------------------------------------------------------------------------------
def main():

    # Parse the command-line parameters
    parser = OptionParser()
    parser.add_option("-i", "--image", dest="imagePath", help="path to the undistorted image file")
    parser.add_option("-f", "--file", dest="calFilePath", help="path to the camera calibration file.")
    parser.add_option("-d", "--debug", dest="debug", action="store_true", help="enable debug output, such as plots")
    (options, args) = parser.parse_args()

    # Make sure a path was specified
    if options.imagePath is None:
        print(f'ERROR: Please specify a path to an image file with the checkerboard.\ne.g. >> python3 {__file__} -i myfile.png')
        sys.exit()

    scale = get_scale(options.imagePath, options.calFilePath, options.debug)
    
    if (scale is not None):
        print(f'Scale factor: {scale:.4f} pixels per millimeter')
    else:
        print('ERROR: Could not get scale factor.')

    return

if __name__ == '__main__':
    main()

# End of File

import numpy as np
import cv2 as cv
import glob
#import yaml
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--directory", help="calibration directory path")
    args = parser.parse_args()

    if args.directory != None:
        print("Directory Loaded : "+str(args.directory))
    else:
        print("No directory loaded used -d argurment")
        quit()

    return args.directory

# Define the chess board rows and columns
rows = 7
cols = 6
# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)
# Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
# Create the arrays to store the object points and the image points
objpoints = []
imgpoints = []
platform=get_args()
images = glob.glob(str(platform)+'/*.jpg')
for fname in images:
    # Load the image and convert it to gray scale
    img = cv.imread(fname)
    img = cv.resize(img,(640,480))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (rows,cols), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (rows,cols), corners2, ret)
        cv.imshow('chess', img)
        cv.waitKey(500)
cv.destroyAllWindows()

# Calibrate the camera and save the results
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez(str(platform)+'calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
#fname = "calibration.yml"
#with open(fname, "w") as f:
#    yaml.dump(ret, f)

# Print the camera calibration error
error = 0

for i in range(len(objpoints)):
    point, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv.norm(imgpoints[i], point, cv.NORM_L2) / len(point)

print("Total error: ", error / len(objpoints))

# Load one of the test images
img2 = cv.imread(images[0])
img2 = cv.resize(img2,(img.shape[1],img.shape[0]))
h, w = img.shape[:2]

# Obtain the new camera matrix and undistort the image
newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistortedImg = cv.undistort(img2, mtx, dist, None, newCameraMtx)
# x, y, w, h = roi
# undistortedImg = undistortedImg[y:y+h, x:x+w]
# cv.imwrite('calibresult.png', undistortedImg)

# Display the final result
cv.imshow('chess', np.hstack((img2, undistortedImg)))
cv.waitKey(0)

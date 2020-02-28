import numpy as np
import cv2 as cv
import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--directory", help="directory path")
    args = parser.parse_args()

    if args.directory != None:
        print("Directory Loaded : "+str(args.directory))
    else:
        print("No directory loaded used -d argurment")
        quit()

    return args.directory

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

def main():
    calibration_dir=get_args()
    # Load previously saved data
    with np.load(str(calibration_dir)+'/calibration.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    # Define the chess board rows and columns
    rows = 7
    cols = 6
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                    [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

    for fname in glob.glob(str(calibration_dir)+'/*.jpg'):
        img = cv.imread(fname)
        img = cv.resize(img,(512,512))
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows,cols),None)
        if ret == True:
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            print(str(objp.shape)+', '+str(corners2.shape))
            ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img,corners2,imgpts)
            cv.imshow('img',img)
            k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
                cv.imwrite(fname[:6]+'.png', img)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
import numpy as np
import cv2 as cv
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--calibration", help="calibration file")
    args = parser.parse_args()

    if args.calibration != None:
        print("Calibration Loaded : "+str(args.calibration))
    else:
        print("No calibration loaded used -c argurment")
        quit()

    return args.calibration


def main():
    #region Initialization
    calibration_file=get_args()

    # Load calibration saved
    with np.load(str(calibration_file)) as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    
    # Descriptor features
    orb = cv.ORB_create()
    # find the keypoints and compute descriptor with ORB
    marker=cv.imread('markers/fiducial.png',1)
    #marker = cv.cvtColor(marker,cv.COLOR_BGR2GRAY)
    kp_marker, des_marker = orb.detectAndCompute(marker, None) 
    marker2=np.array(marker)
    # draw only keypoints location,not size and orientation
    cv.drawKeypoints(marker,kp_marker,marker2,color=(0,255,0), flags=0)
    cv.imshow("Marker",marker2)
    #endregion

    cap = cv.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return        
        
        # Display the resulting frame
        cv.imshow('Camera',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Pose Estimation
        #@TODO use mtx & co + papier prof   cv find homography     





    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
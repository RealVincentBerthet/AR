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
    # Initialization
    calibration_file=get_args()

    # Load previously calibration saved
    with np.load(str(calibration_file)) as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    #homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    #camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
    #orb = cv.ORB_create()
    # create BFMatcher object based on hamming distance  
    #bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    #dir_name = os.getcwd()
    #model = cv.imread(os.path.join(dir_name, 'reference/detail.png'), 0)
    # Compute model keypoints and its descriptors
    #kp_model, des_model = orb.detectAndCompute(model, None)
    # Load 3D model from OBJ file
    #obj = OBJ(os.path.join(dir_name, 'models/Pikachu.obj'), swapyz=True) 


    cap = cv.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return 

        # Display the resulting frame
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Descriptor features
        





    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
import numpy as np
import cv2 as cv
import argparse
import ARTools
from ARTools import *

def get_args(calibration,marker,minMatches,maxMatches,video):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--calibration', help='calibration filepath')
    parser.add_argument('-m','--marker', help='marker image path')
    parser.add_argument('-min','--minmatches', help='min matches')
    parser.add_argument('-max','--maxmatches', help='max matches')
    parser.add_argument('-v','--video',help='video path')
    parser.add_argument('-r','--real',help='play video at fps rate in real time',action='store_true')
    args = parser.parse_args()

    if args.calibration != None :
        calibration=str(args.calibration)
    if args.marker != None :
        marker=str(args.marker)
    if args.minmatches != None :
        minMatches=int(args.minmatches)
    if args.maxmatches != None :
        maxMatches=int(args.maxmatches)
    if args.video != None :
        video=str(args.video)
    
    return calibration,marker,minMatches,maxMatches,video,not(args.real)

def main():
    #region Initialization
    video=0 #0 to use camera
    calibration_file='./videos/genius_F100/calibration/calibration.npz'
    marker_file='./markers/natural.png'
    minMatches=5
    maxMatches=20
    calibration_file,marker_file,minMatches,maxMatches,video,realMode=get_args(calibration_file,marker_file,minMatches,maxMatches,video)
    moveWindows=True

    # ARPipeline
    pipeline=ARPipeline(video=video,realMode=realMode)
    pipeline.LoadCamCalibration(calibration_file)
    pipeline.LoadMarker(marker_file)
    pikachu=OBJ('./models/wolf.obj', swapyz=True)  
    #endregion
    
    while(True): 
        frame=pipeline.GetFrame()
       
        matches,frame_kp=pipeline.ComputeMatches(frame)
        homography,_=pipeline.ComputeHomography(matches,frame_kp,minMatches=minMatches)
        
        matches_refined=pipeline.RefineMatches(matches,frame_kp,minMatches=minMatches)
        homography_refined,_=pipeline.ComputeHomography(matches_refined,frame_kp,minMatches=minMatches)
       
        warped,homography_warped=pipeline.WarpMarker(frame,homography_refined,minMatches=minMatches)
        rvecs, tvecs=pipeline.ComputePose(frame,homography_warped) #@TODO pnp

        #region Rendering
        ar=frame.copy()
        ar=pipeline.renderer.Draw2DRectangle(ar,homography,color=(255,0,0))
        ar=pipeline.renderer.Draw2DRectangle(ar,homography_refined,color=(0,255,0))
        ar=pipeline.renderer.Draw2DRectangle(ar,homography_warped,color=(0,0,255))
        ar=pipeline.renderer.DrawObj(ar,homography_warped,pikachu,eye=0.6)

        cv.imshow('AR Camera',ar)
        cv.imshow('Keypoints',pipeline.renderer.DrawKeypoints(frame,frame_kp))
        img_matches=pipeline.renderer.DrawMatches(frame,frame_kp,matches,maxMatches=maxMatches)
        img_matches = cv.resize(img_matches,(frame.shape[1],frame.shape[0]))
        cv.imshow('Matches',img_matches)
        img_matches_refined=pipeline.renderer.DrawMatches(frame,frame_kp,matches_refined,maxMatches=maxMatches)
        img_matches_refined = cv.resize(img_matches_refined,(frame.shape[1],frame.shape[0]))
        cv.imshow('Matches refined',img_matches_refined)
        cv.imshow('Warp',warped)

        if moveWindows==True :
            # init position windows once
            cv.moveWindow('AR Camera',0,0)
            cv.moveWindow('Keypoints',frame.shape[1],0)
            cv.moveWindow('Matches',2*frame.shape[1],0)
            cv.moveWindow('Matches refined',2*frame.shape[1],frame.shape[0])
            cv.moveWindow('Warp',frame.shape[1],frame.shape[0])
            moveWindows=False
        #endregion

if __name__ == '__main__':
    main()
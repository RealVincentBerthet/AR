#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""main.py: AR program"""
__author__      = 'Vincent Berthet'
__license__     = 'MIT'
__email__       = 'vincent.berthet42@gmail.com'
__website__     = 'https://realvincentberthet.github.io/vberthet/'

import numpy as np
import cv2 as cv
import argparse
import ARTools
from ARTools import *

def get_args():
    """
    get_args function return optional parameters.

    :return: argurments set by default or overriden
    """
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Process of pose estimation to display an AR Object. More information in the dedicated README.')
    parser.add_argument('-s','--source',type=str, default='0', help='Capture device index or video filepath')
    parser.add_argument('-x', type=int, default=640, help='Witdh of the image source')
    parser.add_argument('-y', type=int, default=480, help='Height of the image source')
    parser.add_argument('-c','--calibration', type=str, default='./videos/genius_F100/calibration/calibration.npz', help='Path to the camera calibration')
    parser.add_argument('-m','--marker', type=str, default='./markers/natural.png', help='Path to image marker')
    parser.add_argument('-o','--object',type=str,default='./models/wolf.obj', help='3D Object to display (.obj)')
    parser.add_argument('-min','--minMatches', type=int, default=5, help='min matches')
    parser.add_argument('-max','--maxMatches',type=int, default=20, help='max matches')
    parser.add_argument('--all', default=False, action='store_true', help='Show any step of the pipeline')
    parser.add_argument('--unmove', default=False, action='store_true', help='Don\'t move position of windows as start')
    parser.add_argument('--log',type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING','ERROR','CRITICAL'], help='Choose logging level')
    args = parser.parse_args()

    return args

def main(opt):
    #region Initialization
    ARTools.log.setLevel(getattr(logging, opt.log.upper(), None))
    pipeline=ARPipeline(capture=opt.source,width=opt.x,height=opt.y,calibration=opt.calibration)
    pipeline.LoadMarker(opt.marker)
    model=OBJ(opt.object, swapyz=True) 
    #endregion

    #region AR Pipeline
    while(True):
        frame=pipeline.GetFrame()
        if frame is not None:
            # Compute pose estimation
            matches,frame_kp=pipeline.ComputeMatches(frame)
            homography,_=pipeline.ComputeHomography(matches,frame_kp,minMatches=opt.minMatches)
            matches_refined=pipeline.RefineMatches(matches,frame_kp,minMatches=opt.minMatches)
            homography_refined,_=pipeline.ComputeHomography(matches_refined,frame_kp,minMatches=opt.minMatches)
            warped,homography_warped=pipeline.WarpMarker(frame,homography_refined,minMatches=opt.minMatches)
            rvecs, tvecs=pipeline.ComputePose(frame,homography_warped) #@TODO pnp

            # Rendering
            ar=frame.copy()
            #ar=pipeline.renderer.Draw2DRectangle(ar,homography,color=(255,0,0))
            #ar=pipeline.renderer.Draw2DRectangle(ar,homography_refined,color=(0,255,0))
            #ar=pipeline.renderer.Draw2DRectangle(ar,homography_warped,color=(0,0,255))
            ar=pipeline.renderer.DrawObj(ar,homography_warped,model,eye=0.4)
            cv.imshow('AR Camera (press "q" to quit)',ar)

            if opt.all :
                cv.imshow('Keypoints',pipeline.renderer.DrawKeypoints(frame,frame_kp))
                img_matches=pipeline.renderer.DrawMatches(frame,frame_kp,matches,maxMatches=opt.maxMatches)
                img_matches = cv.resize(img_matches,(frame.shape[1],frame.shape[0]))
                cv.imshow('Matches',img_matches)
                img_matches_refined=pipeline.renderer.DrawMatches(frame,frame_kp,matches_refined,maxMatches=opt.maxMatches)
                img_matches_refined = cv.resize(img_matches_refined,(frame.shape[1],frame.shape[0]))
                cv.imshow('Matches refined',img_matches_refined)
                cv.imshow('Warp',warped)

                # Initialize position of window once
                if not opt.unmove :
                    opt.unmove=True
                    ypadding=60
                    cv.moveWindow('AR Camera (press "q" to quit)',0,0)
                    cv.moveWindow('Keypoints',frame.shape[1],0)
                    cv.moveWindow('Matches',2*frame.shape[1],0)
                    cv.moveWindow('Matches refined',2*frame.shape[1],frame.shape[0]+ypadding)
                    cv.moveWindow('Warp',frame.shape[1],frame.shape[0]+ypadding)
            
            Camera.checkKey()
        else:
            # No frame available
            break
    #endregion

if __name__ == '__main__':
    opt = get_args()
    main(opt)
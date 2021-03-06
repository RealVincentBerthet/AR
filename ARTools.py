#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARTools.py: ARTools AR pipeline step and rendering"""
__author__      = 'Vincent Berthet'
__license__     = 'MIT'
__email__       = 'vincent.berthet42@gmail.com'
__website__     = 'https://realvincentberthet.github.io/vberthet/'

import numpy as np
import cv2 as cv
import os
import math
import logging
import logging.config
from CameraOpenCV import Camera

# Logging
logging.config.fileConfig('logger.conf')
log = logging.getLogger()

class FrameTracking:
    def __init__(self,n=10,popTreshold=5):
        self.n=n
        self.frames=[]
        self.pop=0
        self.popTreshold=popTreshold

    def Add(self,object):
        if len(self.frames)==self.n :
            self.frames.pop()
        self.frames.insert(0,object)
    
    def Pop(self):
        res=None
        if len(self.frames)>0 :
            res=self.frames.pop(0)
            self.pop=self.pop+1
            if self.pop<=self.popTreshold :
                self.Clear()

        return res

    def Mean(self):
        res=None
        if len(self.frames) > 0 :
            res=self.frames[0]

        for f in range(1,len(self.frames)) :
            res=res*f

        return res

    def Clear(self):
        self.frames=[]

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
 
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            # elif values[0] in ('usemtl', 'usemat'):
            #     material = values[1]
            # elif values[0] == 'mtllib':
            #     self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

class Renderer:
    def __init__(self,pipeline):
        """ Use to render pipeline"""
        self.pipeline=pipeline
    
    def computeProjectionMatrix(self,homography):
        """
        computeProjectionMatrix Method compute projection matrix
        :param homography: homography provided to compute projection matrix
        :return: projection matrix
        """
        # Compute rotation along the x and y axis as well as the translation
        homography = homography * (-1)
        rot_and_transl = np.dot(np.linalg.inv(self.pipeline.CAP.mtx), homography)
        col_1 = rot_and_transl[:, 0]
        col_2 = rot_and_transl[:, 1]
        col_3 = rot_and_transl[:, 2]
        # normalise vectors
        l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / l
        rot_2 = col_2 / l
        translation = col_3 / l
        # compute the orthonormal basis
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_3 = np.cross(rot_1, rot_2)
        # finally, compute the 3D projection matrix from the model to the current frame
        projection = np.stack((rot_1, rot_2, rot_3, translation)).T

        return np.dot(self.pipeline.CAP.mtx, projection)
        
    def drawObj(self,img, homography,obj,color=(255,255,255),line=True,eye=1):
        """
        drawObj Method render obj object on a image

        :param img: input image
        :param homography: homography computed
        :param obj: obj to render
        :param obj: obj to render
        :param color: color use to render obj
        :param line: type of drawing for polyline
        :param eye: scale
        :return: image with obj drawn
        """
        if homography is not None :
            self.pipeline.tracker.Add(homography)
        else :
            self.pipeline.tracker.Pop()
        homography=self.pipeline.tracker.Mean()
        if homography is None :
            return img

        projection=self.computeProjectionMatrix(homography)
        vertices = obj.vertices
        scale_matrix = np.eye(3) * eye
        h = self.pipeline.marker_image.shape[0]
        w = self.pipeline.marker_image.shape[1]

        image=img.copy()
        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            # render marker in the middle of the reference surface. To do so,
            # marker points must be displaced
            points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
            dst = cv.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)

            if line == True :
                image = cv.polylines(image, [imgpts], True, color, 1, cv.LINE_AA)
            else :
                cv.fillConvexPoly(image, imgpts, (0, 0, 255),lineType=4)

        return image

    def drawKeypoints(self,img,keypoints):
        """
        drawKeypoints Method render keypoints on a image

        :param img: input image
        :param keypoints: keypoints object list
        :param obj: obj to render
        :return: image with keypoints
        """
        tmp=img.copy()
        cv.drawKeypoints(img,keypoints,tmp ,color=(0,255,0), flags=0)
        size=(int(img.shape[1]),int(img.shape[0]))
        tmp = cv.resize(tmp,size)

        return tmp

    def drawMatches(self,img,img_kp,matches,maxMatches=None):
        """
        drawMatches Method render matches on a image

        :param img: input image
        :param img_kp: keypoints of the image
        :param matches: matches found
        :param maxMatches: maximum amount of matches 
        :return: image with matches
        """
        keypoints=self.drawKeypoints(self.pipeline.marker_image,self.pipeline.marker_kp)
        if matches is None :
            tmp=cv.drawMatchesKnn(keypoints,self.pipeline.marker_kp,img,img_kp,None,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        else:
            if maxMatches == None or maxMatches>len(matches)-1:
                maxMatches=len(matches)-1
                if maxMatches < 0 :
                    maxMatches=0
            tmp=cv.drawMatchesKnn(keypoints,self.pipeline.marker_kp,img,img_kp,matches[:maxMatches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return tmp

    def draw2DRectangle(self,img,homography,color=(255,0,0)):
        """
        draw2DRectangle Method render a 2D rectangle

        :param img: input image
        :param homography: homography computed
        :param color: color of the line
        :return: image with 2D rectangle drawn
        """
        frame=img.copy()
        if homography is not None :
            # Draw a rectangle that marks the found model in the frame
            h=self.pipeline.marker_image.shape[0]
            w=self.pipeline.marker_image.shape[1]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # project corners into frame
            dst = cv.perspectiveTransform(pts, homography)
            imgpts=np.int32(dst)
            # connect them with lines  
            frame = cv.polylines(frame, [imgpts], True, color, 3, cv.LINE_AA)

        return frame  

    def draw3DCube(self,img, rvecs,tvecs):
        """
        draw3DCube Method render a 3D  cube

        :param img: input image
        :param rvecs: computed from pnp
        :param tvecs: computed from pnp
        :return: image with 3D cube drawn
        """
        frame=img.copy()
        if rvecs is not None and tvecs is not None :
            axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],[0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])
            # project 3D points to image plane
            imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, self.pipeline.CAP.mtx, self.pipeline.CAP.dist)

            imgpts = np.int32(imgpts).reshape(-1,2)

            # draw ground floor in green
            frame = cv.drawContours(frame, [imgpts[:4]],-1,(0,255,0),-3)
            # draw pillars in blue color
            for i,j in zip(range(4),range(4,8)):
                frame = cv.line(frame, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
            # draw top layer in red color
            frame = cv.drawContours(frame, [imgpts[4:]],-1,(0,0,255),3)

        return frame

class ARPipeline:
    def __init__(self,capture=0,width=640,height=480,calibration=None,detector=cv.ORB_create(),matcher=cv.BFMatcher(normType=cv.NORM_HAMMING)):
        """
        ARPipeline class use to manage pipeline to process pose estimation

        :param capture: Capture use, it can be camera index of your device (0 for first webcam) or video path
        :param width: width set for the capture resolution (set if available on the device)
        :param height: heigjt set for the capture resolution (set if available on the device)
        :param calibration:  calibration file
        :param detector: detector used by the pipeline [ORB, SIFT, AKAZE]
        :param matcher: matcher used by the pipeline [HAMMING, L2,FLANN]
        """
        self.CAP=Camera(capture,calibration,(width,height))
        self.detector=detector
        self.matcher=matcher #https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        self.tracker=FrameTracking()
        self.marker_image=None
        self.marker_kp=None
        self.marker_des=None
        self.marker_points2D=None
        self.marker_points3D=None
    
    def loadMarker(self,marker_path):
        """
        loadMarker Method load natural marker to use

        :param marker_path: path to the natural marker used (.png/.jpg)
        """
        # Extract marker features
        self.marker_image=cv.imread(marker_path,cv.IMREAD_COLOR)
        self.marker_kp, self.marker_des = self.detector.detectAndCompute(cv.cvtColor(self.marker_image, cv.COLOR_BGR2GRAY), None)        
        
        # Compute marker points
        h=self.marker_image.shape[0]
        w=self.marker_image.shape[1]
        h_norm=h/max(h,w)
        w_norm=w/max(h,w)
        self.marker_points2D = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
        self.marker_points3D = np.float32([[-w_norm,-h_norm,0],[w_norm,-h_norm,0],[w_norm,h_norm,0],[-w_norm,h_norm,0]])
        
        self.renderer=Renderer(self)
        log.info('Marker used is \"'+str(marker_path)+'\"')
    
    def getFrame(self):
        """
        getFrame Method get next frame from the capture

        :return: next frame
        """
        return self.CAP.getFrame()
    
    def computeMatches(self,frame):
        """
        computeMatches Method compute matches on a frame

        :return: good matches and frames keypoints
        """
        good_matches = []
        k=2

        if len(frame.shape)==3 :
            gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray=frame

        frame_kp, frame_des = self.detector.detectAndCompute(gray, None)
        if frame_des is not None and len(frame_kp)>=k:
            if type(self.matcher)==cv.FlannBasedMatcher :
                matches=self.matcher.knnMatch(np.asarray(self.marker_des,np.float32),np.asarray(frame_des,np.float32),k=k) # CV32_F
            else :
                matches=self.matcher.knnMatch(self.marker_des,frame_des,k=k)
            # Ratio test
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])
        else:
            log.debug('Not enough descriptors found in the frame ('+str(len(frame_kp))+' found at least '+str(k)+' are required)')

        return good_matches,frame_kp

    def computeHomography(self,matches,frame_kp,minMatches=10):
        """
        computeHomography Method compute homography given matches and descriptor of the frame
        :param matches: frame matches
        :param frame_kp: frame keypoints
        :param minMatches: minimum amount of matches requested
        :return: the homograpy and mask computed
        """
        homography=None
        mask=None
        
        if len(matches) >minMatches :      
            src_points = np.float32([self.marker_kp[m[0].queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_points = np.float32([frame_kp[m[0].trainIdx].pt for m in matches]).reshape(-1,1,2)
            homography, mask=cv.findHomography(src_points,dst_points,cv.RANSAC,ransacReprojThreshold=5.0)
        else :
            log.debug('Cannot compute homography without enough matches ('+str(len(matches))+' found at least '+str(minMatches)+' are required)')

        return homography,mask
    
    def refineMatches(self,matches,frame_kp,minMatches=10):
        """
        refineMatches Method refine matches by applying a mask to remove outliers
        :param matches: frame matches
        :param minMatches: minimum amount of matches requested
        :return: the correct matches found
        """
        correct_matches=[]
        homography,mask=self.computeHomography(matches,frame_kp,minMatches=minMatches)
        
        if homography is not None :
            # Get inliers mask
            correct_matches = [matches[i] for i in range(len(matches)) if mask[i]]

        return correct_matches
    
    def warpMarker(self,frame,homography,minMatches=10):
        """
        warpMarker Method warp marker and refine homography
        :param frame: frame
        :param homography: homography computed
        :param minMatches: minimum amount of matches requested
        :return: warp perspective and homography refined
        """
        warped=np.zeros((frame.shape[0],frame.shape[1]))
        homography_res=homography

        if homography is not None :
            #Applies a perspective transformation to warp image using homography found
            warped=cv.warpPerspective(cv.cvtColor(self.marker_image, cv.COLOR_BGR2GRAY),homography,(frame.shape[1],frame.shape[0]),cv.WARP_INVERSE_MAP | cv.INTER_CUBIC)
            warped_matches,warped_kp=self.computeMatches(warped)
            warped_correct_matches=self.refineMatches(warped_matches,warped_kp,minMatches=minMatches)
           
            if len(warped_correct_matches)>0 :
                homography_warped,_=self.computeHomography(warped_correct_matches,warped_kp,minMatches=minMatches)
                
                if homography_warped is not None :
                    homography_res=homography_warped

        return warped,homography_res

    def computePose(self,frame,homography):
        """
        computePose Method compute pose according to pnp
        :param frame: frame
        :param homography: homography computed
        :return: results of pnp
        """
        rvecs=None
        tvecs=None
        
        if homography is not None :
            # Transform frame edge based on new homography
            dst = cv.perspectiveTransform(self.marker_points2D, homography) 
            # Estimate the camera pose, objectPoints=3D points, imagePoints=2D points
            _,rvecs, tvecs = cv.solvePnP(objectPoints=self.marker_points3D, imagePoints=dst, cameraMatrix=self.CAP.mtx, distCoeffs=self.CAP.dist)

            #rotMat=cv.Rodrigues(rvecs)
        else :
            log.debug('Cannot compute pose without homography')  

        return rvecs, tvecs



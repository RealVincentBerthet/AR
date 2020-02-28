import numpy as np
import cv2 as cv
import os
import math
from objloader_simple import *

_debug=False
print('[INFO] Debug ARTools is : '+str(_debug))

def Log(message):
     if _debug==True :
        print(message)

class Struct(object):
    def __getattr__(self, name):
        setattr(self, name, None)

class FrameTracking:
    def __init__(self,n=15):
        self.n=n
        self.frames=[]

    def Add(self,object):
        if len(self.frames)==self.n :
            self.frames.pop()
        self.frames.insert(0,object)
    
    def Clear(self):
        self.frames=[]

class Renderer:
    def __init__(self,pipeline):
        self.cam=pipeline.cam
        self.marker=pipeline.marker
    
    def ComputeProjectionMatrix(self,homography):
        # Compute rotation along the x and y axis as well as the translation
        homography = homography * (-1)
        rot_and_transl = np.dot(np.linalg.inv(self.cam.mtx), homography)
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

        return np.dot(self.cam.mtx, projection)
        
    def DrawObjHomography(self,img, homography,obj):
        if homography is not None :
            projection=self.ComputeProjectionMatrix(homography)
            vertices = obj.vertices
            scale_matrix = np.eye(3) * 3
            h = self.marker.img.shape[0]
            w = self.marker.img.shape[1]

            for face in obj.faces:
                face_vertices = face[0]
                points = np.array([vertices[vertex - 1] for vertex in face_vertices])
                points = np.dot(points, scale_matrix)
                # render marker in the middle of the reference surface. To do so,
                # marker points must be displaced
                points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
                dst = cv.perspectiveTransform(points.reshape(-1, 1, 3), projection)
                imgpts = np.int32(dst)
                cv.fillConvexPoly(img, imgpts, (137, 27, 211))

        return img

    def DrawKeypoints(self,img,keypoints):
        tmp=img.copy()
        cv.drawKeypoints(img,keypoints,tmp ,color=(0,255,0), flags=0)
        size=(int(img.shape[1]),int(img.shape[0]))
        tmp = cv.resize(tmp,size)

        return tmp

    def DrawMatches(self,img,img_kp,matches,maxMatches=None):
        keypoints=self.DrawKeypoints(self.marker.img,self.marker.kp)
        if matches is None :
            tmp=cv.drawMatchesKnn(keypoints,self.marker.kp,img,img_kp,None,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        else:
            if maxMatches == None or maxMatches>len(matches)-1:
                maxMatches=len(matches)-1
                if maxMatches < 0 :
                    maxMatches=0
            tmp=cv.drawMatchesKnn(keypoints,self.marker.kp,img,img_kp,matches[:maxMatches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return tmp

    def Draw2DRectangle(self,img,homography,color=(255,0,0)):
        frame=img.copy()
        if homography is not None :
            # Draw a rectangle that marks the found model in the frame
            h=self.marker.img.shape[0]
            w=self.marker.img.shape[1]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # project corners into frame
            dst = cv.perspectiveTransform(pts, homography)
            # connect them with lines  
            frame = cv.polylines(frame, [np.int32(dst)], True, color, 3, cv.LINE_AA)

        return frame  

    def Draw3DCube(self,img, rvecs,tvecs):
        frame=img.copy()
        if rvecs is not None and tvecs is not None :
            axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],[0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])
            # project 3D points to image plane
            imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, self.cam.mtx, self.cam.dist)

            imgpts = np.int32(imgpts).reshape(-1,2)

            # draw ground floor in green
            frame = cv.drawContours(frame, [imgpts[:4]],-1,(0,255,0),-3)
            # draw pillars in blue color
            for i,j in zip(range(4),range(4,8)):
                frame = cv.line(frame, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
            # draw top layer in red color
            frame = cv.drawContours(frame, [imgpts[4:]],-1,(0,0,255),3)

        return frame

    def Draw3DCubeTest(self,img, rvecs, tvecs):
        frame=img.copy()
        if rvecs is not None and tvecs is not None :
            # Cube corner points in world coordinates
            axis = np.float32([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, -1], [1, 0, -1], [1, 1, -1],
                                [0, 1, -1]]).reshape(-1, 3)

            # Project corner points of the cube in image frame
            imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, self.cam.mtx, self.cam.dist)

            # Render cube in the video
            # Two faces (top and bottom are shown. They are connected by red lines.
            imgpts = np.int32(imgpts).reshape(-1, 2)
            face1 = imgpts[:4]
            face2 = np.array([imgpts[0], imgpts[1], imgpts[5], imgpts[4]])
            face3 = np.array([imgpts[2], imgpts[3], imgpts[7], imgpts[6]])
            face4 = imgpts[4:]

            # Bottom face
            frame = cv.drawContours(img, [face1], -1, (255, 0, 0), -3)

            # Draw lines connected the two faces
            frame = cv.line(img, tuple(imgpts[0]), tuple(imgpts[4]), (0, 0, 255), 2)
            frame = cv.line(img, tuple(imgpts[1]), tuple(imgpts[5]), (0, 0, 255), 2)
            frame = cv.line(img, tuple(imgpts[2]), tuple(imgpts[6]), (0, 0, 255), 2)
            frame = cv.line(img, tuple(imgpts[3]), tuple(imgpts[7]), (0, 0, 255), 2)

            # Top face
            frame = cv.drawContours(img, [face4], -1, (0, 255, 0), -3)

        return frame

class ARPipeline:
    def __init__(self,escapeKey='q',width=640,height=480,video=0,loop=True,realMode=False):
        self.cam=Struct()
        self.marker=Struct()
        self.escapeKey=escapeKey
        self.cam.width=width
        self.cam.height=height
        self.cam.video=video
        self.cam.loop=loop
        self.cam.realMode=realMode
        print('[INFO] Press \"'+str(self.escapeKey)+'\" to quit')

        if video==0 :
            self.cam.capture = cv.VideoCapture(video,cv.CAP_DSHOW)
            print('[INFO] Video capture is \"camera\"')
        else :
            self.cam.capture = cv.VideoCapture(video)
            self.cam.video=video

            print('[INFO] Video capture is \"'+str(video)+'\", '+str(int(self.cam.capture.get(cv.CAP_PROP_FPS)))+' FPS')
        
        self.cam.capture.set(cv.CAP_PROP_FRAME_WIDTH, self.cam.width)
        self.cam.capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.cam.height)
        self.descriptor = cv.ORB_create()
        self.matcher= cv.BFMatcher() #https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

    def LoadCamCalibration(self,calibrationPath):
        with np.load(str(calibrationPath)) as X:
            self.cam.mtx, self.cam.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        print('[INFO] Calibration file used is \"'+str(calibrationPath)+'\"')
    
    def LoadMarker(self,markerPath):
        # Extract marker features
        self.marker.img=cv.imread(markerPath,cv.IMREAD_COLOR)
        self.marker.kp, self.marker.des = self.descriptor.detectAndCompute(cv.cvtColor(self.marker.img, cv.COLOR_BGR2GRAY), None)        
        
        # Compute marker points
        h_norm=self.cam.height/max(self.cam.height,self.cam.width)
        w_norm=self.cam.width/max(self.cam.height,self.cam.width)
        self.marker.points2D = np.float32([[0,0],[self.cam.width,0],[self.cam.width,self.cam.height],[0,self.cam.height]]).reshape(-1, 1, 2)
        self.marker.points3D = np.float32([[-w_norm,-h_norm,0],[w_norm,-h_norm,0],[w_norm,h_norm,0],[-w_norm,h_norm,0]])
        
        self.renderer=Renderer(self)
        print('[INFO] Marker used is \"'+str(markerPath)+'\"')
    
    def GetFrame(self):
        # Capture frame-by-frame
        if self.cam.capture.isOpened() :
            ret, frame = self.cam.capture.read()

            if self.cam.video !=0 :
                # Video from file
                if not(self.cam.realMode==True) :
                    for i in range(int(self.cam.capture.get(cv.CAP_PROP_FPS))) :
                        if  cv.waitKey(1) & 0xFF == ord(str(self.escapeKey)):
                            # Stop with key
                            self.cam.capture.release()
                            cv.destroyAllWindows()
                            quit()

                if not ret :
                    if self.cam.loop==True :
                        self.cam.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
                        return self.GetFrame()
                    else :
                        # End of video file
                        self.cam.capture.release()
                        cv.destroyAllWindows()
                        quit()
                else:
                    # resize video frame
                    frame = cv.resize(frame,(self.cam.width,self.cam.height))
            else :
                # Video from camera check return
                if not ret :
                    print('[ERROR] Unable to capture video')
                    self.cam.capture.release()
                    cv.destroyAllWindows()
                    quit()    

            if  cv.waitKey(1) & 0xFF == ord(str(self.escapeKey)):
                # Stop with key
                self.cam.capture.release()
                cv.destroyAllWindows()
                quit()
        else :
            print('[ERROR] Unable to capture video source')
            self.cam.capture.release()
            cv.destroyAllWindows()
            quit()

        return frame
    
    def ComputeMatches(self,frame):
        good_matches = []
        k=2

        if len(frame.shape)==3 :
            gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray=frame

        frame_kp, frame_des = self.descriptor.detectAndCompute(gray, None)

        if frame_des is not None and len(frame_kp)>=k:
            matches=self.matcher.knnMatch(self.marker.des,frame_des,k=k)
            # Ratio test
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])
        else:
            Log('[Warning] Not enough descriptors found in the frame')

        return good_matches,frame_kp

    def ComputeHomography(self,matches,frame_kp,minMatches=10):
        homography=None
        mask=None
        
        if len(matches) >minMatches :      
            src_points = np.float32([self.marker.kp[m[0].queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_points = np.float32([frame_kp[m[0].trainIdx].pt for m in matches]).reshape(-1,1,2)
            homography, mask=cv.findHomography(src_points,dst_points,cv.RANSAC,ransacReprojThreshold=5.0)
        else :
            Log('[Warning] Cannot compute homography without enough matches')

        return homography,mask
    
    def RefineMatches(self,matches,frame_kp,minMatches=10):
        correct_matches=[]
        homography,mask=self.ComputeHomography(matches,frame_kp,minMatches=minMatches)
        
        if homography is not None :
            # Get inliers mask
            correct_matches = [matches[i] for i in range(len(matches)) if mask[i]]

        return correct_matches
    
    def WarpMarker(self,frame,homography,minMatches=10):
        warped=np.zeros((frame.shape[0],frame.shape[1]))
        homography_res=homography

        if homography is not None :
            #Applies a perspective transformation to warp image using homography found
            warped=cv.warpPerspective(cv.cvtColor(self.marker.img, cv.COLOR_BGR2GRAY),homography,(frame.shape[1],frame.shape[0]),cv.WARP_INVERSE_MAP | cv.INTER_CUBIC)
            warped_matches,warped_kp=self.ComputeMatches(warped)
            warped_correct_matches=self.RefineMatches(warped_matches,warped_kp,minMatches=minMatches)
           
            if len(warped_correct_matches)>0 :
                homography_warped,_=self.ComputeHomography(warped_correct_matches,warped_kp,minMatches=minMatches)
                
                if homography_warped is not None :
                    homography_res=homography_warped

        return warped,homography_res

    def ComputePose(self,frame,homography):
        rvecs=None
        tvecs=None
        #@TODO NOK
        if homography is not None :
            # Transform frame edge based on new homography
            dst = cv.perspectiveTransform(self.marker.points2D, homography) 
            # Estimate the camera pose from frame corner points in world coordinates and image frame
            _,rvecs, tvecs = cv.solvePnP(objectPoints=self.marker.points3D, imagePoints=dst, cameraMatrix=self.cam.mtx, distCoeffs=self.cam.dist)

            #rotMat=cv.Rodrigues(rvecs)
        else :
            Log('[Warning] Cannot compute pose without homography')  

        return rvecs, tvecs

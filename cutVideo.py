import cv2 as cv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f','--filepath', help='video filepath')
args = parser.parse_args()



nb_cut=40
cam=cv.VideoCapture(args.filepath)
nb_cut=40
frame_start=0.1 #%
frame_end=0.9 #%
if frame_end<frame_start :
    print('[ERROR] frame_start should be < than frame_end')
    quit()

frame_start=int(cam.get(cv.CAP_PROP_FRAME_COUNT)*frame_start)
frame_end=int(cam.get(cv.CAP_PROP_FRAME_COUNT)*frame_end)
interval=int((frame_end-frame_start)/nb_cut)

print('[INFO] Video loaded : \"'+str(args.filepath)+'\", '+str(cam.get(cv.CAP_PROP_FRAME_COUNT))+' frames')
print('[INFO] Cut '+str(nb_cut)+' times, from frame '+str(frame_start)+' to frame '+str(frame_end))
if cam.isOpened() :
    print('*********** WRITE ***************')
    for i in range(nb_cut) :
        cam.set(cv.CAP_PROP_POS_FRAMES,frame_start+i*interval)
        ret, frame = cam.read()
        frame=cv.resize(frame,(640,480))
        cut=os.path.dirname(args.filepath)+'\cut_'+str(i)+'.jpg'
        cv.imwrite(cut, frame)
        print(cut)
else :
    print('[ERROR] Camera not open')
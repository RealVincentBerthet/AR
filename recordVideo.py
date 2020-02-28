import cv2 as cv
import numpy as np
from datetime import datetime

# Create a VideoCapture object
cap = cv.VideoCapture(0,cv.CAP_DSHOW)
path='output_'+str(datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"))+'.mp4'
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print('[INFO] Frame size : '+str(frame_width)+', '+str(frame_height))
print('[INFO] Path write : '+str(path))
print('[INFO] Press \"q\" to end the capture')
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv.VideoWriter(path,cv.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))
while(True):
  ret, frame = cap.read()
 
  if ret == True: 
     
    # Write the frame into the file 'output.avi'
    out.write(frame)
 
    # Display the resulting frame    
    cv.imshow('frame',frame)
 
    # Press Q on keyboard to stop recording
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else:
    break 
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv.destroyAllWindows() 
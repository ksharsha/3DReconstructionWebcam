#Caapture frames from webcam and store it in a folder
import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0);
numframes=0;
string='frames/frame'
os.mkdir('frames')
while(True):
    # Capture frame-by-frame
    if numframes>30:
    	print("Captured thirty frames of the image")
    	break
    
    wait = raw_input("Press enter to capture the next frame")
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fstr=string+str(numframes)+'.png'
    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imwrite(fstr,gray)
    numframes=numframes+1
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
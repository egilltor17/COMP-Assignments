import numpy as np
import sys
import signal
import time
if sys.platform == 'win32':
    # location of precompiled windows OpenCV binaries 
    sys.path.insert(1, 'D:\\Downloads\\opencv\\build\\python')
import cv2 as cv

cap = cv.VideoCapture(2) # Connect to USB webcam
# cap = cv.VideoCapture("http://192.168.1.128:8080/video") # Connect to Android IP camera

print(f'Resolution: {cap.get(3)}x{cap.get(4)}')

def shutdownSignal(signalNumber, frame):
    print(' Received signal:', signalNumber)
    cap.release()
    # Destroys the OpenCv windows
    cv.destroyAllWindows()
    sys.exit()

signal.signal(signal.SIGINT, shutdownSignal)

while(True):
    ##1 Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    ##3 Start of time measurement
    startTime = time.time()

    ##4 Find and mark the brightspot
    ## Convert to monotone
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    cv.circle(frame, maxLoc, 5, (10,10,10), 2)

    ##5 Find and mark the readest spot
    ## Extract red channel
    red = frame[:,:,2]
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(red)
    cv.circle(frame, maxLoc, 5, (10,10,250), 2)

    ##6 4&5 in for loop
    # maxValM, maxLocM, maxValR, maxLocR = 0, (0, 0), 0, (0, 0)
    # for i in range(int(cap.get(4))):
    #     for j in range(int(cap.get(3))):
    #         if sum(frame[i,j]) >= maxValM:
    #             maxValM = sum(frame[i,j])
    #             maxLocM = (j,i)
    #         if int(frame[i,j,2]) > maxValR:
    #             maxValR = int(frame[i,j,2])
    #             maxLocR = (j,i)

    # cv.circle(frame, maxLocM, 5, (250,250,250), 2)
    # cv.circle(frame, maxLocR, 5, (100,10,250), 2)

    ##3 Display the FPS
    cv.putText(frame, f'FPS: {1 / (time.time() - startTime):.2f}', (10,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    ##2 Display the resulting frame
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
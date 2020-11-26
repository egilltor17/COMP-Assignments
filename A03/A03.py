import numpy as np
import random
import sys
import signal
import time
from scipy import sparse
from scipy import optimize
from functools import reduce
import operator
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if sys.platform == 'win32':
    # location of precompiled windows OpenCV binaries 
    # sys.path.insert(1, 'D:\\Downloads\\opencv\\build\\python')
    sys.path.insert(1, 'E:/Libraries/opencv/build/python')
import cv2 as cv

cap = cv.VideoCapture(0) # Connect to USB webcam
# cap = cv.VideoCapture("http://192.168.1.128:8080/video") # Connect to Android IP camera

def halfRes():
    cap.set(3, 320)
    cap.set(4, 240)

print(f'Resolution: {cap.get(3)}x{cap.get(4)}')

def shutdownSignal(signalNumber, frame):
    print(' Received signal:', signalNumber)
    cap.release()
    # Destroys the OpenCv windows
    cv.destroyAllWindows()
    sys.exit()

def findObjects(outputs, frame):
    hT, wT, cT = frame.shape
    bbox = []
    classIds = []
    conf = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int(det[0] * wT - w / 2), int(det[1] * hT - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                conf.append(float(confidence))
    
    indices =  cv.dnn.NMSBoxes(bbox, conf, confThreshold, nmsThreshold)
    for i in indices:
        x, y, w, h = bbox[i[0]]
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        cv.putText(frame, f'{classNames[classIds[i[0]]].upper()}: {conf[i[0]]*100:.2f}%', (x,y-5), cv.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)

    # return bbox, classIds, conf

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdownSignal)

    classFile    = 'data/coco.names'
    modelConfig  = 'data/yolov3-320.cfg'
    modelWeights = 'data/yolov3-320.weights'
    
    size = 320
    confThreshold = 0.5
    nmsThreshold = 0.3
    FPS = 0

    classNames = []
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    net = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    while(True):
        ## Start of time measurement
        startTime = time.time()

        ##1 Capture frame-by-frame
        ret, frame = cap.read()

        blob = cv.dnn.blobFromImage(frame, 1/255, (size, size), [0,0,0], 1, False)
        net.setInput(blob)
        
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)
        findObjects(outputs, frame)


        ## Display the FPS
        cv.putText(frame, f'FPS: {FPS:.2f}', (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
        
        ##4 Display the resulting frame
        cv.imshow('frame', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv.waitKey(-1)
        ## End time measurements
        FPS = 1 / (time.time() - startTime)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
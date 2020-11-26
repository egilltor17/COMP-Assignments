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
    sys.path.insert(1, 'D:\\Downloads\\opencv\\build\\python')
import cv2 as cv

cap = cv.VideoCapture(0) # Connect to USB webcam
# cap = cv.VideoCapture("http://192.168.1.128:8080/video") # Connect to Android IP camera

def halfRes():
    cap.set(3, 320)
    cap.set(4, 240)
def quarterRes():
    cap.set(3, 160)
    cap.set(4, 120)
def deciRes():
    cap.set(3, 64)
    cap.set(4, 48)


# halfRes()
# quarterRes()
# deciRes()

print(f'Resolution: {cap.get(3)}x{cap.get(4)}')

def shutdownSignal(signalNumber, frame):
    print(' Received signal:', signalNumber)
    cap.release()
    # Destroys the OpenCv windows
    cv.destroyAllWindows()
    sys.exit()

def pointSlopeIntercept(pt1, pt2):
    # y = mx + b
    m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0]) if pt2[0] != pt1[0] else float('inf')
    b = pt1[1] - m * pt1[0]
    return m, b

def distFromLineToPointSlope(m, b, point):
    return abs(m * point[0] - point[1] + b) / np.sqrt(m * m + 1)

def pointLine(pt1, pt2):
    # Ax + By + C = 0
    a = (pt2[1] - pt1[1])
    b = (pt2[0] - pt1[0])
    c = -a * pt1[0] - b * pt1[1]
    return a, b, c

def distFromLineToPoint(a, b, c, points):
    return np.abs(a * points[:,0] + b * points[:,1] + c) / np.sqrt(a * a + b * b) if a != b else np.zeros((points.shape[0]))

def calcPoint(x, m, b):
    return (int(x), int(x * m + b))

def lineIntercept(m1, b1, m2, b2):
    return (b1 - b2) / (m2 - m1) if m1 != m2 else float('inf')

def RANSAC(points, n=2, k=200, t=20, d=100):
    """
    Given:
    data – A set of observations.
    model – A model to explain observed data points.
    n – Minimum number of data points required to estimate model parameters.
    k – Maximum number of iterations allowed in the algorithm.
    t – Threshold value to determine data points that are fit well by model.
    d – Number of close data points required to assert that a model fits well to data.

    Return:
        bestFit – model parameters which best fit the data (or null if no good model is found)
    """
    iterations = 0
    bestFit = ((0, 0), (int(cap.get(3)), int(cap.get(4))))
    bestErr = float('inf')
    bestPnt = None
    bestInt = None
    bestM = float('nan')
    bestB = float('nan')

    while iterations < k:
        sampledInliersIndex = np.random.choice(np.arange(points.shape[0]), size=n, replace=False)
        sampledInliers = points[sampledInliersIndex]
        nonSampledPoints = np.delete(points, sampledInliersIndex, 0)
        
        a, b, c = pointLine(*sampledInliers)
        # alsoInliers = nonSampledPoints[distFromLineToPoint(a, b, c, nonSampledPoints) < t]
        alsoInliersIndex = np.argwhere(distFromLineToPoint(a, b, c, nonSampledPoints) < t).flatten()
        alsoInliers = nonSampledPoints[alsoInliersIndex]

        if len(alsoInliers) > d:
            ## This implies that we may have found a good model
            indexes = np.concatenate((sampledInliersIndex, alsoInliersIndex), 0)
            inliers = np.concatenate((sampledInliers, alsoInliers), 0)
            # print(np.polyfit(inliers[:,0], inliers[:,1], 1))
            m, b = np.polyfit(inliers[:,0], inliers[:,1], 1)
            # betterModel = (calcPoint(sampledInliers[0][0], m, b), calcPoint(sampledInliers[1][0], m, b))
            # betterModel = (calcPoint(inliers[0][0], m, b), calcPoint(inliers[-1][0], m, b))

            actualInliersIndex = np.argwhere(distFromLineToPoint(m, -1, b, points) < t).flatten()
            actualInliers = points[actualInliersIndex]

            thisErr = np.sum(distFromLineToPoint(m, -1, b, actualInliers))
            
            # (m, b), c, _, _, _ = np.polyfit(inliers[:,0], inliers[:,1], 1, full=True)
            # thisErr = c
            
            # model = LinearRegression().fit(inliers[:,0].reshape(-1, 1), inliers[:,1])
            # betterModel = model.predict(sampledInliers[:,0].reshape(-1, 1))
            # betterModel =((sampledInliers[0,0], int(betterModel[0])), (sampledInliers[1,0], int(betterModel[1])))
            # # thisErr = model.score(inliers[:,0].reshape(-1, 1), inliers[:,1])
            # thisErr = mean_squared_error(model.predict(inliers[:,0].reshape(-1, 1)), inliers[:,1])
            
            if thisErr < bestErr:
                inliers = inliers[inliers[:,0].argsort()]
                # bestFit = (calcPoint(sampledInliers[0][0], m, b), calcPoint(sampledInliers[1][0], m, b))
                # bestFit = ((sampledInliers[0,0], int(betterModel[0])), (sampledInliers[1,0], int(betterModel[1])))
                bestFit = (calcPoint(np.min(inliers[:,0]), m, b), calcPoint(np.max(inliers[:,0]), m, b))
                bestErr = thisErr
                bestPnt = actualInliers
                bestInt = actualInliersIndex
                bestM = m
                bestB = b
        
        iterations += 1
    
    return bestFit, bestM, bestB,  bestPnt, bestInt

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdownSignal)

    while(True):
        ##1 Capture frame-by-frame
        ret, frame = cap.read()
        frame_rec = np.zeros_like(frame)

        ## Start of time measurement
        startTime = time.time()

        # frame = cv.flip(frame, 1)

        ##2 Use an edge detector
        edges = cv.Canny(frame, 100, 200)

        ## Covert the sparse edges matrix to a list of points
        points = np.transpose(np.nonzero(edges)[::-1])

        ## Overlay Canny edges on frame
        # edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        # frame = cv.addWeighted(frame, 0.5, edges, 0.5, 0)

        if points.shape[0] > 2:
            ##3 Use RANSAC to fit a single straight line with the greatest support to the extracted edge pixels.
            line, m, b, inliers, indexes = RANSAC(points)
            ##4 Display the line in the live image
            # cv.line(frame, line[0], line[1], (0,0,255), 3)
            
            # while inliers is not None:
            #     for i in inliers:
            #         cv.circle(frame, tuple(i), 1, (50,50,50), 2)
            #     points = np.delete(points, indexes, 0)
            #     if points.shape[0] > 2:
            #         line, inliers, indexes = RANSAC(points)
            #     cv.line(frame, line[0], line[1], (0,0,255), 3)
            
            ## Draw multiple lines
            if inliers is not None:
                # for i in inliers:
                #     cv.circle(frame, tuple(i), 1, (0,0,128), 2)
                # s1 = points.shape
                points2 = np.delete(points, indexes, 0)
                if points2.shape[0] > 2:
                    line2, m2, b2, inliers2, indexes2 = RANSAC(points2)
                    if inliers2 is not None:
                    #     for i in inliers2:
                    #         cv.circle(frame, tuple(i), 1, (0,128,0), 2)
                    #     s2 = points2.shape
                    #     cv.line(frame, line2[0], line2[1], (0,255,0), 3)
                        points3 = np.delete(points2, indexes2, 0)
                        if points3.shape[0] > 2:
                            line3, m3, b3, inliers3, indexes3 = RANSAC(points3)
                            if inliers3 is not None:                
                        #         for i in inliers3:
                        #             cv.circle(frame, tuple(i), 1, (128,0,0), 2)
                        #         s3 = points3.shape
                        #         cv.line(frame, line3[0], line3[1], (255,0,0), 3)
                                points4 = np.delete(points3, indexes3, 0)
                                if points4.shape[0] > 2:
                                    line4, m4, b4, inliers4, indexes4 = RANSAC(points4)
                                    if inliers4 is not None:
                                    #     for i in inliers4:
                                    #         cv.circle(frame, tuple(i), 1, (0,128,128), 2)
                                    #     cv.line(frame, line4[0], line4[1], (0,255,255), 3)
                                        # print(s1, s2, s3, points4.shape)

                                        mbs = sorted([[m, b], [m2, b2], [m3, b3], [m4, b4]], key=lambda x: x[0])
                                        # print(mbs, [[m, b], [m2, b2], [m3, b3], [m4, b4]])
                                        # pt1 = calcPoint(lineIntercept(*(mbs[0]), *(mbs[2])), *(mbs[0]))
                                        # pt2 = calcPoint(lineIntercept(*(mbs[0]), *(mbs[3])), *(mbs[0]))
                                        # pt3 = calcPoint(lineIntercept(*(mbs[1]), *(mbs[2])), *(mbs[1]))
                                        # pt4 = calcPoint(lineIntercept(*(mbs[1]), *(mbs[3])), *(mbs[1]))
                                        pt1 = calcPoint(lineIntercept(mbs[0][0], mbs[0][1], mbs[2][0], mbs[2][1]), mbs[0][0], mbs[0][1])
                                        pt2 = calcPoint(lineIntercept(mbs[0][0], mbs[0][1], mbs[3][0], mbs[3][1]), mbs[0][0], mbs[0][1])
                                        pt3 = calcPoint(lineIntercept(mbs[1][0], mbs[1][1], mbs[2][0], mbs[2][1]), mbs[1][0], mbs[1][1])
                                        pt4 = calcPoint(lineIntercept(mbs[1][0], mbs[1][1], mbs[3][0], mbs[3][1]), mbs[1][0], mbs[1][1])
                                        
                                        corners = [pt1, pt2, pt3, pt4]
                                        if math.isinf(sum([c[0] for c in corners])):
                                            pt1 = calcPoint(lineIntercept(mbs[0][0], mbs[0][1], mbs[2][0], mbs[2][1]), mbs[2][0], mbs[2][1])
                                            pt2 = calcPoint(lineIntercept(mbs[1][0], mbs[1][1], mbs[2][0], mbs[2][1]), mbs[2][0], mbs[2][1])
                                            pt3 = calcPoint(lineIntercept(mbs[0][0], mbs[0][1], mbs[3][0], mbs[3][1]), mbs[3][0], mbs[3][1])
                                            pt4 = calcPoint(lineIntercept(mbs[1][0], mbs[1][1], mbs[3][0], mbs[3][1]), mbs[3][0], mbs[3][1])
                                            corners = [pt1, pt2, pt3, pt4]

                                        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), corners), [len(corners)] * 2))
                                        corners = sorted(corners, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
                                        

                                        cv.line(frame, corners[0], corners[1], (255,255,0), 1)
                                        cv.line(frame, corners[1], corners[2], (255,255,0), 1)
                                        cv.line(frame, corners[2], corners[3], (255,255,0), 1)
                                        cv.line(frame, corners[3], corners[0], (255,255,0), 1)
                                        
                                        # cv.circle(frame, corners[0], 2, (255,0,255), 2)
                                        # cv.circle(frame, corners[1], 2, (255,0,255), 2)
                                        # cv.circle(frame, corners[2], 2, (255,0,255), 2)
                                        # cv.circle(frame, corners[3], 2, (255,0,255), 2)

                                        # print(corners)

                                        height, width, channels = frame.shape
                                        h, mask = cv.findHomography(cv.UMat(np.array(corners)), cv.UMat(np.array([(0,0), (0,width), (height,width), (height,0)])), cv.RANSAC)

                                        # Use homography
                                        frame_rec = cv.warpPerspective(frame, h, (width, height))


        ## Display the FPS
        cv.putText(frame, f'FPS: {1 / (time.time() - startTime):.2f}', (10,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        
        ##4 Display the resulting frame
        cv.imshow('frame', frame)
        cv.imshow('frame rectifieded', frame_rec)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv.waitKey(-1)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
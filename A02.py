import numpy as np
import random
import sys
import signal
import time
from scipy import sparse
from scipy import optimize
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

def RANSAC(points, n=2, k=200, t=3, d=150):
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

    while iterations < k:
        sampledInliersIndex = np.random.choice(np.arange(points.shape[0]), size=n, replace=False)
        sampledInliers = points[sampledInliersIndex]
        nonSampledPoints = np.delete(points, sampledInliersIndex, 0)
        
        a, b, c = pointLine(*sampledInliers)
        alsoInliers = nonSampledPoints[distFromLineToPoint(a, b, c, nonSampledPoints) < t]

        if len(alsoInliers) > d:
            ## This implies that we may have found a good model
            inliers = np.concatenate((sampledInliers, alsoInliers), 0)
            
            # print(np.polyfit(inliers[:,0], inliers[:,1], 1))
            m, b = np.polyfit(inliers[:,0], inliers[:,1], 1)
            betterModel = (calcPoint(sampledInliers[0][0], m, b), calcPoint(sampledInliers[1][0], m, b))
            thisErr = np.sum(distFromLineToPoint(m, -1, b, inliers))
            
            # (m, b), c, _, _, _ = np.polyfit(inliers[:,0], inliers[:,1], 1, full=True)
            # thisErr = c
            
            # model = LinearRegression().fit(inliers[:,0].reshape(-1, 1), inliers[:,1])
            # betterModel = model.predict(sampledInliers[:,0].reshape(-1, 1))
            # betterModel =((sampledInliers[0,0], int(betterModel[0])), (sampledInliers[1,0], int(betterModel[1])))
            # # thisErr = model.score(inliers[:,0].reshape(-1, 1), inliers[:,1])
            # thisErr = mean_squared_error(model.predict(inliers[:,0].reshape(-1, 1)), inliers[:,1])
            
            if thisErr < bestErr:
                # bestFit = (calcPoint(sampledInliers[0][0], m, b), calcPoint(sampledInliers[1][0], m, b))
                # bestFit = ((sampledInliers[0,0], int(betterModel[0])), (sampledInliers[1,0], int(betterModel[1])))
                bestFit = betterModel
                bestErr = thisErr
                bestPnt = inliers
        
        iterations += 1
    
    return bestFit

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdownSignal)

    while(True):
        ##1 Capture frame-by-frame
        ret, frame = cap.read()

        ## Start of time measurement
        startTime = time.time()

        # cv.resize(frame, (320, 240))
        # frame = cv.flip(frame, 1)

        ##2 Use an edge detector
        edges = cv.Canny(frame, 100, 200)

        ## Covert the sparse edges matrix to a list of points
        points = np.transpose(np.nonzero(edges)[::-1])

        edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        frame = cv.addWeighted(frame, 0.5, edges, 0.5, 0)

        if points.shape[0] > 2:
            ##3 Use RANSAC to fit a single straight line with the greatest support to the extracted edge pixels.
            line = RANSAC(points)
            ##4 Display the line in the live image
            cv.line(frame, line[0], line[1], (0,0,255), 3)

        ## Display the FPS
        cv.putText(frame, f'FPS: {1 / (time.time() - startTime):.2f}', (10,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        
        ##4 Display the resulting frame
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
## Importing a pretrained onnx model and importing it into OpenCV
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import sys
import signal
import time
if sys.platform == 'win32':
    # location of precompiled windows OpenCV binaries 
    # sys.path.insert(1, 'D:/Downloads/opencv/build/python')
    sys.path.insert(1, 'E:/Libraries/opencv/build/python')
import cv2 as cv
import matplotlib.pyplot as plt
import urllib.request
import tarfile

import onnx
import keras2onnx

cap = cv.VideoCapture(0) # Connect to USB webcam
# cap = cv.VideoCapture("http://192.168.1.128:8080/video") # Connect to Android IP camera

def halfRes():
    cap.set(3, 320)
    cap.set(4, 240)
def square():
    cap.set(3, 224)
    cap.set(4, 224)

square()

print(f'Resolution: {cap.get(3)}x{cap.get(4)}')


def downloadImages():
    filename = 'flower_photos.tgz'
    dir_path = filename.split('.')[0]

    if not (os.path.exists(filename) or os.path.exists(dir_path)):
        url = ' https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
        urllib.request.urlretrieve(url, filename)

    if not os.path.exists(dir_path):
        tf = tarfile.open(filename)
        tf.extractall()
    
    # Initialize classes list, this list will contain the names of our classes.
    classes = []
    if os.path.exists(dir_path):
        # Iterate over the names of each class
        for class_name in os.listdir(dir_path):
            # Get the full path of each class
            class_path = os.path.join(dir_path, class_name)
            # Check if the class is a directory/folder
            if os.path.isdir(class_path):
                # Get the number of images in each class and print them
                No_of_images = len(os.listdir(class_path))
                print("Found {} images of {}".format(No_of_images , class_name))
                # Also store the name of each class
                classes.append(class_name)

        # Sort the list in alphabatical order and print it    
        classes.sort()
        print(classes)

    return classes, dir_path



# Here we are creating a function for displaying images of flowers from the data generators
def display_images(data_generator, no = 15):
    sample_training_images, labels = next(data_generator)

    plt.figure(figsize=[25,25])

    # By default we're displaying 15 images, you can show more examples
    total_samples = sample_training_images[:no]

    cols = 5
    rows = int(len(total_samples) / cols)

    for i, img in enumerate(total_samples, 1):
            plt.subplot(rows, cols, i )
            plt.imshow(img)

            # Converting One hot encoding labels to string labels and displaying it.
            class_name = classes[np.argmax(labels[i-1])]
            plt.title(class_name)
            plt.axis('off')


def shutdownSignal(signalNumber, frame):
    print(' Received signal:', signalNumber)
    cap.release()
    # Destroys the OpenCv windows
    cv.destroyAllWindows()
    sys.exit()


def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):  
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdownSignal)

    FPS = 0

    # Read the ONNX model
    model_dir = '/home/egill/Documents/HR/Term_7/3. Vikna Námskeið [Computer Vison]/Assignments/A03/data/resnet50-v1-7.onnx'
    # model_dir = 'data/resnet50-v1-7.onnx'
    net = cv.dnn.readNetFromONNX(model_dir)
        
    label_names = []
    with open('data/synset.txt', 'rt') as f:
        label_names = f.read().rstrip('\n').split('\n')

    while(True):
        ## Start of time measurement
        startTime = time.time()

        ##1 Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.resize(frame, (224,224))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Normalize the image and format it
        # frame = np.array([frame]).astype('float32') / 255.0
        # frame = preprocess(np.array([[frame[2,:,:], frame[0,:,:], frame[1,:,:]]]))
        inputFrame = preprocess(np.array([np.swapaxes(np.swapaxes(frame, 0, 1), 0, 2)]))

        # Input Image to the network
        net.setInput(inputFrame)

        # Perform a Forward pass
        Out = net.forward()

        # Get the top predicted index
        index = np.argmax(Out[0])

        # Get the probability of the class.
        prob = np.max(Out[0])

        # Get the class name by putting index number
        label = ' '.join(label_names[index].split(' ')[1:])
        # text = "Predicted: {} {:.2f}%".format(label, prob)

        # Write predicted flower on image
        cv.putText(frame, label, (10, 40),  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv.putText(frame, f'{prob:.2f}%', (10, 60),  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)


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
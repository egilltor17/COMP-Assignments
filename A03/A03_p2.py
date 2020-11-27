## Training a keras model and importing it into OpenCV
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


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdownSignal)

    FPS = 0
    
    ## ========================================================== ##
    ## Part 1: Training a Custom Image Classifier with Tensorflow ##
    ## ========================================================== ##


    # Set the batch size, width, height and the percentage of the validation split.
    batch_size = 60
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    split = 0.2
    epochs = 60

    classes, dir_path = downloadImages()

    model_dir = 'data/flowers.h5'
    if not os.path.exists(model_dir):
        #  Setup the ImagedataGenerator for training, pass in any supported augmentation schemes, notice that we're also splitting the data with split argument.
        datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, 
            validation_split= split,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect')

        # Setup the ImagedataGenerator for validation, no augmentation is done, only rescaling is done, notice that we're also splitting the data with split argument.
        datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, 
            validation_split=split) 

        # Data Generation for Training with a constant seed valued 40, notice that we are specifying the subset as 'training'
        train_data_generator = datagen_train.flow_from_directory(
            batch_size=batch_size,
            directory=dir_path,
            shuffle=True,
            seed = 40,
            subset= 'training',
            interpolation = 'bicubic',
            target_size=(IMG_HEIGHT, IMG_WIDTH))

        # Data Generator for validation images with the same seed to make sure there is no data overlap, notice that we are specifying the subset as 'validation'
        vald_data_generator = datagen_val.flow_from_directory(
            batch_size=batch_size, 
            directory=dir_path,
            shuffle=True,
            seed = 40,
            subset = 'validation',
            interpolation = 'bicubic',
            target_size=(IMG_HEIGHT, IMG_WIDTH))

        # The "subset" variable tells the Imagedatagerator class which generator gets 80% and which gets 20% of the data

        # Display Original Images
        display_images(vald_data_generator)
        # Display Augmented Images
        display_images(train_data_generator)

        # First Reset the generators, since we used the first batch to display the images.
        vald_data_generator.reset()
        train_data_generator.reset()

        # Here we are creating Sequential model also defing its layers
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
            MaxPooling2D(),
            Dropout(0.10),

            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),

            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),

            Conv2D(128, 3, padding='same', activation='relu'),
            MaxPooling2D(),

            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dropout(0.10),
            Dense(len(classes), activation ='softmax')
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

    
        # Start Training
        history = model.fit( train_data_generator,  steps_per_epoch= train_data_generator.samples // batch_size, epochs=epochs, validation_data= vald_data_generator,
            validation_steps = vald_data_generator.samples // batch_size )

            # Plot the accuracy and loss curves for both training and validation
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training loss')
        plt.legend()

        plt.show()

        # Saving your model to disk allows you to use it later
        model.save(model_dir)
    else:
        # Load the keras model
        model = load_model(model_dir)

    # Read the rose image
    rose_dir = '/home/egill/Documents/HR/Term_7/3. Vikna Námskeið [Computer Vison]/Assignments/A03/flower_photos/roses/512578026_f6e6f2ad26.jpg'
    img = cv.imread(rose_dir)

    # Resize the image to the size you trained on.
    imgr = cv.resize(img,(224,224))

    # Convert image BGR TO RGB, since OpenCV works with BGR and tensorflow in RGB.
    imgrgb = cv.cvtColor(imgr, cv.COLOR_BGR2RGB)

    # Normalize the image to be in range 0-1 and then convert to a float array.
    final_format = np.array([imgrgb]).astype('float64') / 255.0

    # Perform the prediction
    pred = model.predict(final_format)

    # Get the index of top prediction
    index = np.argmax(pred[0])

    # Get the max probablity for that prediction
    prob = np.max(pred[0])

    # Get the name of the predicted class using the index
    label = classes[index]

    # Display the image and print the predicted class name with its confidence.
    print("Predicted Flowers is : {} {:.2f}%".format(label, prob*100))
    plt.imshow(img[:,:,::-1])
    plt.axis("off")
    plt.show()

    ## ========================================================== ##
    ## Part 2: Converting Our Classifier to ONNX format           ##
    ## ========================================================== ##
    

    onnx_model_dir = 'data/flowers.onnx'
    if not os.path.exists(onnx_model_dir):
        # Convert it into onnx
        onnx_model = keras2onnx.convert_keras(model, model.name)

        # Save the model as flower.onnx
        onnx.save_model(onnx_model, onnx_model_dir)
    else:
        # Read the ONNX model
        net = cv.dnn.readNetFromONNX('flowers_model.onnx')
        

    # Define class names and sort them alphabatically as this is how tf.keras remembers them 
    label_names = ['daisy','dandelion','roses','sunflowers','tulips']
    label_names.sort()


    # Read the image
    daisy_dir = '/home/egill/Documents/HR/Term_7/3. Vikna Námskeið [Computer Vison]/Assignments/A03/flower_photos/daisy/163978992_8128b49d3e_n.jpg'
    img_original = cv.imread(daisy_dir)

    img = img_original.copy()
    # Resize Image
    img = cv.resize(img_original,(224,224))

    # Convert BGR TO RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Normalize the image and format it
    img = np.array([img]).astype('float64') / 255.0

    # Input Image to the network
    net.setInput(img)

    # Perform a Forward pass
    Out = net.forward()

    # Get the top predicted index
    index = np.argmax(Out[0])

    # Get the probability of the class.
    prob = np.max(Out[0])

    # Get the class name by putting index number
    label =  label_names[index]
    text = "Predicted: {} {:.2f}%".format(label, prob)

    # Write predicted flower on image
    cv.putText(img_original, text, (5, 4*26),  cv.FONT_HERSHEY_COMPLEX, 4, (100, 20, 255), 6)
    
    # Display image
    plt.figure(figsize=(10,10))
    plt.imshow(img_original[:,:,::-1])
    plt.axis("off")

    while(True and False):
        ## Start of time measurement
        startTime = time.time()

        ##1 Capture frame-by-frame
        ret, frame = cap.read()


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
from tensorflow_large_model_support import LMSKerasCallback
import keras
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.applications.vgg16 import decode_predictions
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import multi_gpu_model
#from keras.preprocessing import image
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras import backend as K
import numpy as np
import os
from tensorflow.python.client import device_lib

lms_callback = LMSKerasCallback()


TRAIN_DIR = 'data/train/'
VALIDATION_DIR = 'data/validation/'
NUM_TRAIN = 10000
NUM_VALIDATION = 3000
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 256
EPOCHS = 75
SAMPLES_TRAIN = 5000
SAMPLES_VAL = 500

def loadDataSet():
    
    train_datagen = ImageDataGenerator( 
        rescale = .1/255, 
        shear_range = .2,
        zoom_range = 0.2, 
        horizontal_flip = True
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size = (IMG_WIDTH, IMG_HEIGHT),
        
        batch_size = BATCH_SIZE,
        class_mode = "categorical"
    )

    validation_generator = train_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size = (IMG_WIDTH, IMG_HEIGHT),
        
        batch_size = BATCH_SIZE,
        class_mode = "categorical"
    )

    return train_generator,validation_generator 


def VGG_16():
    with tf.device('/cpu:0'):
        model = Sequential()
        
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        
        # top layer of the VGG net
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
    
        model.add(Dense(2, activation='softmax', name="posts"))
    
    parallel_model = multi_gpu_model(model, gpus=8)

    return parallel_model


def main(): # load the model
    trainData, validData = loadDataSet()
    model = VGG_16()
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    print('Starting training...')
    model.fit_generator(
        trainData,
        steps_per_epoch= SAMPLES_TRAIN // BATCH_SIZE,
        epochs = EPOCHS,
        validation_steps = SAMPLES_VAL // BATCH_SIZE,
        validation_data = validData,
        callbacks=[lms_callback]
    )
    print("Training Completed~~~~~~~~~~")
    model.save("VGG_trained.h5")
   
main()






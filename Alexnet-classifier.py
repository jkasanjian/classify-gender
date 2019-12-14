# from shutil import copyfile
# import glob
import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.preprocessing import image
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_DIR = 'data/train/'
VALIDATION_DIR = 'data/validation/'
NUM_TRAIN = 10000
NUM_VALIDATION = 3000
IMG_WIDTH = 227
IMG_HEIGHT = 227
BATCH_SIZE = 256
EPOCHS = 75
SAMPLES_TRAIN = 5000
SAMPLES_VAL = 500

def creatingSets():
    
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

#Instantiate an empty model
with tf.device('/cpu:0'):
    model = Sequential()
# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())
model.add(Flatten())


# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()
parallel_model = multi_gpu_model(model, gpus=4)
# (4) Compile 
parallel_model.compile(loss='categorical_crossentropy', optimizer='adam',\
metrics=['accuracy'])

def main():
    trainData, validData = creatingSets()
    print("Starting training~~~~~~~~~~")
    parallel_model.fit_generator(
        trainData,
        steps_per_epoch= SAMPLES_TRAIN // BATCH_SIZE,
        epochs = EPOCHS,
        validation_steps = SAMPLES_VAL // BATCH_SIZE,
        validation_data = validData,
    )
    print("Training Completed~~~~~~~~~~")
    model.save("trainedModel.h5")
main()

import keras
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, ZeroPadding2D, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.preprocessing import image
import numpy as np


trainingDir ="data/train/"
validationDir = "data/validation/"

# Training Model
# VGG Stats According to Paper

# Epoch = 74
# Dense = 24.8/7.5
# Multi Crop = 24.6/7.5
num_train_samples = 5000
num_val_samples = 500
epochs = 70
batch_size = 96

def loadDataSet():
    img_width, img_height = 224, 224


    trainData = ImageDataGenerator(
        rescale = 1. /225,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

    testData = ImageDataGenerator(rescale=1./225)

    trainGen = trainData.flow_from_directory(
        trainingDir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = "categorical")
    
    validationGen = trainData.flow_from_directory(
        validationDir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = "categorical")
    
    
    
    return trainGen,validationGen
    #return trainGen, validationGen

#     strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
#     with strategy.scope():

with tf.device('/cpu:0'):
    model = Sequential()
#1 Convolutional Layer
model.add(Conv2D(64, (3, 3), input_shape= (224, 224, 3), padding='same', activation='relu'))

#2 Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#3 Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

#4 Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#5 Convolutional Layer
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

#6 Convolutional Layer
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

#7 Convolutional Layer
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#8 Convolutional Layer
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

#9 Convolutional Layer
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

#10 Convolutional Layer
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#11 Convolutional Layer
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

#12 Convolutional Layer
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

#13 Convolutional Layer
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Flatten())

#14 Dense Layer 1
model.add(Dense(4096, activation='relu'))

#15 Dense Layer 2
model.add(Dense(4096, activation='relu'))

#Output Layer
model.add(Dense(2, activation='softmax'))

#model.load_weights("VGGTrainedWeightsV2")
parallel_model = multi_gpu_model(model, gpus=8)

# (4) Compile 
print("[Compiling Started]")
#Compiling model
parallel_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
print("[Compiling Complete]")
model.summary()


    

def main(): # load the model
    train,valid = loadDataSet()
    print("Data Loaded")
    print("[Training Started]")
    parallel_model.fit_generator(train, steps_per_epoch= num_train_samples//batch_size, epochs = 74, validation_steps=num_val_samples//batch_size, validation_data = valid)
    print("Training Completed~~~~~~~~~~")
    parallel_model.save("VGGtrainedModel.h5")
    parallel_model.save_weights("VGGTrainedWeightsV2")
    


def test():
    print('Generating data')
    test_datagen = ImageDataGenerator( 
        rescale = .1/255, 
        shear_range = .2,
        zoom_range = 0.2, 
        horizontal_flip = True
    )

    testSet = test_datagen.flow_from_directory(
        'data/test/',
        target_size = (224,224),    
        batch_size = 96,
        class_mode = "categorical"
        )
    print('Data generated')

    print('Loading model...')
    model = keras.models.load_model('VGGtrainedModel.h5')
    print('Model loaded')

    print('Testing')
    score = model.evaluate(testSet)
    print(model.metrics_names, score)
    print('Done testing')


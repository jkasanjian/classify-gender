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

trainingDir ="data/train/"
validationDir = "data/validation/"

# Training Model
# VGG Stats According to Paper
# Batch Size = 256
# Epoch = 74
# Dense = 24.8/7.5
# Multi Crop = 24.6/7.5
num_train_samples = 5000
num_val_samples = 500
epochs = 74
batch_size = 256

def loadDataSet():
    img_width, img_height = 224, 224

    if K.image_data_format() == "channels_first":
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

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

    validationGen = testData.flow_from_directory(
        validationDir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = "categorical")
    
    xTrain, yTrain = trainGen.next()
    xVal, yVal = validationGen.next()

    return xTrain, yTrain, xVal, yVal, trainGen, validationGen
    #return trainGen, validationGen

def VGG_16():
#     strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
#     with strategy.scope():
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
        
#     print("[Compiling Started]")
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#     print("[Compiling Complete]")
    
#     model.summary()
# #     try:
# #         model = multi_gpu_model(model)
# #     except:
# #         print("Failure")
# #         pass
    
#     #model.load_weights("VGGTrainedWeights")
#     xTrain, yTrain, xVal, yVal, trainGen, validationGen = loadDataSet()
#     print("TGEN: ", trainGen)
#     print("VGEN: ", validationGen)
#     print("XTRAIN: ", xTrain)
#     print("YTRAIN: ", yTrain)
#     print("XVAL: ", xVal)
#     print("YVAL: ", yVal)
#     spe = num_train_samples // batch_size
#     vs = num_val_samples // batch_size
#     yTrain = keras.utils.to_categorical(yTrain, 3)
#     yVal = keras.utils.to_categorical(yVal, 3)
#     model.fit(xTrain, yTrain,
#               steps_per_epoch=spe,
#               epochs=epochs,
#               validation_data=(xVal,yVal),
#               validation_steps=vs)
#     model.save_weights("VGGTrainedWeightsV2")
#     print("ITSATASGARWGWEGHIEWGAPIWEGSNSAIDPFNDPISFHWAEIPFHAWIESFNAIEWPFGHW")

    return model

def loadDataV2():
    

def main(): # load the model
    model = VGG_16()
    xTrain, yTrain, xVal, yVal, trainGen, validationGen = loadDataSet()
    #xTrain, yTrain, xVal, yVal = loadDataSet()
    #model.compile(optimizer='sgd', loss='categorical_crossentropy')
    
    
    print("[Compiling Started]")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], name='labels')
    print("[Compiling Complete]")
    print("[Training Started]")
    
    NUM_GPUS = 4
    strategy = tf.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)
    estimator = tf.keras.estimator.model_to_estimator(model,
                                                  config=config)
#     estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
#                                                       keras_model_path=None,
#                                                       custom_objects=None,
#                                                       model_dir=None,
#                                                       config=config,
#                                                       checkpoint_format='checkpoint')

    spe = num_train_samples // batch_size
    vs = num_val_samples // batch_size
    print("DATATYPE: ", type(tuple(xTrain)))
    shapes = ([None], ())
    types = (tf.string, tf.int32)
    dataset = tf.data.Dataset.from_generator(lambda:(trainGen),
    output_shapes=shapes, output_types=types)
#     estimator.train(trainData,
#                         steps_per_epoch=num_train_samples // batch_size,
#                         epochs=epochs,
#                         validation_data=valData,
#                         validation_steps=num_val_samples // batch_size)
    estimator.train((xTrain, yTrain))
                    #hooks=None,
                    #steps=None,
                    
                    #saving_listeners=None)
#     model.fit_generator(trainData,
#               steps_per_epoch=spe,
#               epochs=epochs,
#               validation_data=valData,
#               validation_steps=vs)
#     estimator.save_weights("VGGTrainedWeightsV2")
    print("[Training Complete]")

    # load an image from file
    image = load_img('data/test/male/000030.jpg', target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)

    #image_pred = np.expand_dims(image_pred, axis = 0)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    rslt = model.predict(image)
    # print("Laberl: ", label)
    #list(filter(lambda num: num != 0, rslt[0]))
    print(rslt)
    # if(rslt[0][0] > rslt[0][1]):
    #     prediction = "Male"
    # else:
    #     prediction = "Female"
    # print(prediction)

    # # reshape data for the model
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # # prepare the image for the VGG model
    # image = preprocess_input(image)
    # # predict the probability across all output classes
    # yhat = model.predict(image)
    # # convert the probabilities to class labels
#     label = decode_predictions(rslt)
#     print(label)
#     # # retrieve the most likely result, e.g. highest probability
#     label = label[0][0]
#     # # print the classification
#     print('%s (%.2f%%)' % (label[1], label[2] * 100))

main()






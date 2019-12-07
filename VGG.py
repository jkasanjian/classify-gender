from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
#from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np

trainingDir ="data/train/"
validationDir = "data/validation/"

def loadDataSet():
    img_width, img_height = 224, 224
    batch_size = 128

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

    return trainGen, validationGen

def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', border_mode='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', border_mode='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', border_mode='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', border_mode='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', border_mode='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', border_mode='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', border_mode='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())

    # top layer of the VGG net
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    model.summary()
    model = multi_gpu_model(model, gpus=4)

    model.load_weights("VGGTrainedWeights")

    return model

def main(): # load the model
    model = VGG_16()
    trainData, valData = loadDataSet()
    #model.compile(optimizer='sgd', loss='categorical_crossentropy')
    print("[Compiling Complete]")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("[Compiling Complete]")

    # Training Model
    # VGG Stats According to Paper
    # Batch Size = 256
    # Epoch = 74
    # Dense = 24.8/7.5
    # Multi Crop = 24.6/7.5
    num_train_samples = 5000
    num_val_samples = 500
    epochs = 74
    batch_size = 128
    print("[Training Started]")

#     model.fit_generator(trainData,
#                         steps_per_epoch=num_train_samples // batch_size,
#                         epochs=epochs,
#                         validation_data=valData,
#                         validation_steps=num_val_samples // batch_size)
#     model.save_weights("VGGTrainedWeightsV2")
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
    label = decode_predictions(rslt)
    print(label)
    # # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # # print the classification
    print('%s (%.2f%%)' % (label[1], label[2] * 100))

main()






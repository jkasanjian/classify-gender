from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

trainingDir ="data/train"
validationDir = "data/validation"

def loadDataSet():
    img_width, img_height = 224, 224
    batch_size = 20

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
        class_mode = "binary")

    validationGen = testData.flow_from_directory(
        validationDir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = "binary")

    return trainGen, validationGen

def main(): # load the model
    model = VGG16()
    trainData, valData = loadDataSet()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    print("[Compiling Complete]")

    # Training Model
    # VGG Stats According to Paper
    # Batch Size = 256
    # Epoch = 74
    # Dense = 24.8/7.5
    # Multi Crop = 24.6/7.5
    num_train_samples = 1000
    num_val_samples = 100
    epochs = 50
    batch_size = 20
    model.fit_generator(trainData,
                        steps_per_epoch=num_train_samples // batch_size,
                        epochs=epochs,
                        validation_data=valData,
                        validation_steps=num_val_samples // batch_size)
    model.save_weights("WeightsFile")
    print("[Training Complete]")

    # load an image from file
    image = load_img('data/test/male/000050.jpg', target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2] * 100))

main()






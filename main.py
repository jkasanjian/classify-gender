import csv
#import cv2
import numpy as np
#from PIL import Image
#from torchvision import models
#import torch
#import os

NUM_TRAINING = 20 #should be 20000!!!!
NUM_VALIDATION = 5000
NUM_TEST = 20000

data = [] 
with open('list_attr_celeba.csv') as Fin:
    reader = csv.reader(Fin, skipinitialspace = True, quotechar = "'")
    for row in reader:
        data.append(row)
        
# removes feature labels
del data[0]

training = []
validation = []
test = []

training_label = []
validation_label = []
test_label = []

#Date extraction

for i in range(NUM_TRAINING):
    index = np.random.randint(0, len(data))
    training.append(data.pop(index))
    training[i][0] = 1 # bias weight
    training_label.append(training[i].pop(21))

for i in range(NUM_VALIDATION):
    index = np.random.randint(0, len(data))
    validation.append(data.pop(index))
    validation[i][0] = 1
    validation_label.append(validation[i].pop(21))

for i in range(NUM_TEST):
    index = np.random.randint(0, len(data))
    test.append(data.pop(index))
    test[i][0] = 1
    test_label.append(test[i].pop(21))


# FEATURE SPACES
training = np.array(training)
training = training.astype(np.float)

validation = np.array(validation)
validation = validation.astype(np.float)

test = np.array(test)
test = test.astype(np.float)


# LABEL SPACES
training_label = np.array(training_label)
training_label = training_label.astype(np.float)

validation_label = np.array(validation_label)
validation_label = validation_label.astype(np.float)

test_label = np.array(test_label)
test_label = test_label.astype(np.float)


print(training_label.shape)
print(validation_label.shape)
print(test_label.shape)
print(training.shape)
print(validation.shape)



# while(count < 20):
#     for filename in os.listdir(folder):
#         target_file =  ((6-len(str(count)))*("0")) + str(count) + ".jpg"
#         if(filename == target_file): 
#             f= "new" + filename
#             filenames.append(f)
#             count += 1
#             im1 = Image.open(folder + "/" + filename)
#             im1 = im1.resize((227,227))
#             im1.save("/Users/samantharain/Desktop/gender-detection/train_imgs/" + f)


# for f in filenames:
#     path = "/Users/samantharain/Desktop/gender-detection/train_imgs/" + f
#     img = cv2.imread(path)
#     if img is not None:
#         images.append(img)


# Evaluate testing dataset, must be compatible for DatasetLoader, make sure BATCH SIZE is 1 for incoming dataset 
def eval(dataset, aNet):
    correct = 0
    alexnet = aNet.load_state_dict('Trained')
    for x,y in dataset:
        output = alexnet(x)
        # Need to check whether torch.no_grad makes a difference one at a time. Should we have it for every iteration?
        with torch.no_grad:
            _, pred = torch.max(output,1)
            if(pred == y):
                correct += 1
    return (correct/len(dataset))

def main():
    eval()



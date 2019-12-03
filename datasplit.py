
from shutil import copyfile
import json
import csv
import numpy as np


NUM_TRAIN = 20000
NUM_VALIDATION = 5000
NUM_TEST = 5000

PATH_TRAIN = 'data/train/'
PATH_VALIDATION = 'data/validation/'
PATH_TEST = 'data/test/'
SOURCE = 'img_align_celeba/'

def split_data():
    data = [] 
    with open('list_attr_celeba.csv') as Fin:
        reader = csv.reader(Fin, skipinitialspace = True, quotechar = "'")
        for row in reader:
            data.append(row)
            

   
    print(len(data))

    # --------------- TRANING --------------- 
    count_male = 0
    count_female = 0
    while (count_male + count_female) < NUM_TRAIN:
        index = np.random.randint(1, len(data))
        entry = (data.pop(index))
        file_name, label = entry[0], entry[21]
        if label == '1':
            if count_male == NUM_TRAIN / 2:
                continue
            count_male += 1
            class_label = 'male/'
        else:
            if count_female == NUM_TRAIN / 2:
                continue
            count_female += 1
            class_label = 'female/'
        copyfile(SOURCE + file_name, PATH_TRAIN + class_label + file_name)

    print('Training male:', count_male)
    print('Training female', count_female)

    
    # --------------- VALIDATION --------------- 
    count_male = 0
    count_female = 0
    while (count_male + count_female) < NUM_VALIDATION:
        index = np.random.randint(1, len(data))
        entry = (data.pop(index))
        file_name, label = entry[0], entry[21]
        if label == '1':
            if count_male == NUM_VALIDATION / 2:
                continue 
            count_male += 1
            class_label = 'male/'
        else:
            if count_female == NUM_VALIDATION / 2:
                continue 
            count_female += 1
            class_label = 'female/'
        copyfile(SOURCE + file_name, PATH_VALIDATION + class_label + file_name)

    print('Validation male:', count_male)
    print('Validation female', count_female)
    

    # --------------- TESTING --------------- 
    count_male = 0
    count_female = 0
    while (count_male + count_female) < NUM_TEST:
        index = np.random.randint(1, len(data))
        entry = (data.pop(index))
        file_name, label = entry[0], entry[21]
        if label == '1':
            if count_male == NUM_TEST / 2:
                continue 
            count_male += 1
            class_label = 'male/'
        else:
            if count_female == NUM_TEST / 2:
                continue 
            count_female += 1
            class_label = 'female/'
        copyfile(SOURCE + file_name, PATH_TEST + class_label + file_name)
        
    print('Test male:', count_male)
    print('Test female', count_female)
    

split_data()
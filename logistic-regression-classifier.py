# IMPORTS

import csv
import numpy as np
import random
from os import listdir, path
from os.path import isfile, join



def load_data():
    male_train = 'data/train/male'
    female_train = 'data/train/female'
    male_train_files = [f for f in listdir(male_train) if isfile(join(male_train, f))]
    female_train_files = [f for f in listdir(female_train) if isfile(join(female_train, f))]

    male_test = 'data/test/male'
    female_test = 'data/test/female'
    male_test_files = [f for f in listdir(male_test) if isfile(join(male_test, f))]
    female_test_files = [f for f in listdir(female_test) if isfile(join(female_test, f))]

    data = {}
    with open('list_attr_celeba.csv') as Fin:
        reader = csv.reader(Fin, skipinitialspace = True, quotechar = "'")
        for row in reader:
            file_name = row[0]
            row[0] = 1
            data[file_name] = row 

    features = data['image_id'] # male is on 21 
    print(features)

    train = [] 
    test = []
    train_label = []
    test_label = [] 

    for f in male_train_files:
        train.append(data[f]) 
    for f in female_train_files:
        train.append(data[f])
    random.shuffle(train)
    for t in train:
        train_label.append(t.pop(21))
 
    for f in male_test_files:
        test.append(data[f]) 
    for f in female_test_files:
        test.append(data[f])
    random.shuffle(test)
    for t in test:
        test_label.append(t.pop(21))

    train = np.array(train)
    train = train.astype(np.float)
    test = np.array(test)
    test = test.astype(np.float)
    train_label = np.array(train_label)
    train_label = train_label.astype(np.float)
    test_label = np.array(test_label)
    test_label = test_label.astype(np.float)

    return train, train_label, test, test_label


def logistic_regression_SGD(data, label, max_iter, learning_rate): 
    N = len(data)
    d = len(data[0])
    # initialize w0
    w = np.zeros((d, 1))
    w = np.transpose(w)

    for t in range(max_iter):
        # pick random point
        i = np.random.randint(0, N)
        # calculate gradient
        gradient = (-label[i] * data[i]) / (1 + np.exp(label[i] * w * data[i]))
        # update weights
        w -= (learning_rate * gradient)

    return np.transpose(w)


def logistic_regression(data, label, max_iter, learning_rate):
    N = len(data)
    d = len(data[0])
    # initialize w0
    w = np.zeros((d, 1))
    w = np.transpose(w)

    for t in range(max_iter):
        # calculate gradient
        gradientSum = np.zeros((1, d))
        for n in range(N):
            gradientSum += ( (label[n] * data[n]) / (1 + np.exp(label[n]* np.dot(w, data[n]))) )
        gradient = (-1/N) * gradientSum
        # update weights 
        w -= (learning_rate * gradient)

    return np.transpose(w)


def accuracy(x, y, w):
    mistakes = 0
    n = len(y)
    w = np.transpose(w)
    for z in range(n):
        y_pred = 1.0 if sigmoid(np.dot(w,x[z])) > .5 else -1.0
        if(y_pred != y[z]):
            mistakes += 1
        
    return (n-mistakes)/n


def sigmoid(s):
    return (1/(1 + np.exp(-s)))


def main(): 
    learning_rate = [.1, .2, .5]
    max_iter = [100, 500, 1000, 5000]
    
    train, train_label, test, test_label = load_data()

    print('TESTING WITH SGD\n')
    for i, m_iter in enumerate(max_iter):
        w = logistic_regression_SGD(train, train_label, m_iter, learning_rate[1])
        Ain, Aout = accuracy(train, train_label, w), accuracy(test, test_label, w)
        print("max iteration testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))

    for i, l_rate in enumerate(learning_rate):
        w = logistic_regression_SGD(train, train_label, max_iter[3], l_rate)
        Ain, Aout = accuracy(train, train_label, w), accuracy(test, test_label, w)
        print("learning rate testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))
        

        
    print('\n\nTESTING LOGISTIC REGRESSION\n')
    for i, m_iter in enumerate(max_iter):
        w = logistic_regression(train, train_label, m_iter, learning_rate[1])
        Ain, Aout = accuracy(train, train_label, w), accuracy(test, test_label, w)
        print("max iteration testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))

    for i, l_rate in enumerate(learning_rate):
        w = logistic_regression(train, train_label, max_iter[2], l_rate)
        Ain, Aout = accuracy(train, train_label, w), accuracy(test, test_label, w)
        print("learning rate testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))



# main()
load_data()

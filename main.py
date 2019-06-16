import numpy as np 
import pandas as pd 
import csv
import cv2
import os, glob
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import datetime
import time

FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# -------------------------------------------------------------------------------

# Creating a list of data for training

def load_data(data_dir):
    labels = []
    images = []
    category = 0
    for d in FISH_CLASSES:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        stop = 0
        for f in file_names:
            img = cv2.imread(f)
            imresize = cv2.resize(img, (200, 125))
            images.append(imresize)
            labels.append(category)
            if stop > 200:
                break
            stop += 1
        print(d,category)
        category += 1

    return images, labels

data_dir = "./train/"
images, labels = load_data(data_dir)

print("Creating a list of data for training -- DONE")

# -------------------------------------------------------------------------------

# defining x_train, x_test, y_train, y_test 

def cross_validate(Xs, ys):
    X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = cross_validate(images, labels)

X_train = np.array(X_train).astype('float32')
X_test = np.array(X_test).astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print("defining x_train, x_test, y_train, y_test -- DONE")

# ------------------------------------------------------------------------------

# Data Augmentation

# from keras.preprocessing.image import ImageDataGenerator
  
# ImageDataGenerator(
#     rotation_range=10.,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.,
#     zoom_range=.1.,
#     horizontal_flip=True,
#     vertical_flip=True)

# -------------------------------------------------------------------------------

def train():

    # Defining CNN model

    def createCNNModel(num_classes):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(125, 200, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        epochs = 17
        lrate = 0.01
        decay = lrate/epochs
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print(model.summary())
        return model, epochs
    model, epochs = createCNNModel(num_classes)
    print("CNN Model created.")
    seed = 7
    np.random.seed(seed)

    print("Defining CNN model -- DONE")

    # training 

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=64)
    scores = model.evaluate(X_test, y_test, verbose=0)
    
    print("training -- DONE")

    # loss curve

    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()
    print("loss curve -- DONE")

    # accuracy curve

    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    plt.show()
    print("accuracy curve -- DONE")

    # Saving the model

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    return model

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

# model = train()
model = load_model()

--------------------------------------------------------------------------------

Neural Network Prediction   

from os import listdir
from os.path import isfile, join
import csv
import math

prediction_output_list = []  

true = []
pred = []
category = -1
total = 0
correct = 0
csv_output_list =[]
csv_output_list.append(".")
csv_output_list.append(FISH_CLASSES)
csv_output_list.append("predicted")
csv_output_list.append("expected")
prediction_output_list.append(csv_output_list)
for d in FISH_CLASSES:
    i = 0
    category += 1
    label_dir = os.path.join(data_dir, d)
    file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
    for f in file_names:
        total += 1
        print("Evaluating at ......")
        print(f)
        img = cv2.imread(f)  
        img = cv2.resize(img, (200, 125))
        imlist = np.array([img])
        print("Neural Net Prediction:")
        cnn_prediction = model.predict_proba(imlist)
        print(cnn_prediction)
        csv_output_list = []
        csv_output_list.append(f)
        maximum = 0
        flag = 0
        j = 0
        for elem in cnn_prediction:
            for value in elem:
                j += 1
                value = value.item()
                value = value * 100
                value = int(math.floor(value))
                if value > maximum:
                    maximum = value
                    flag = j
                csv_output_list.append(value)
        true.append(category)
        pred.append(flag)
        if flag == category:
            correct += 1
        csv_output_list.append(flag)
        csv_output_list.append(category)
        print("CSV Output List Formatted:")
        print(csv_output_list)
        prediction_output_list.append(csv_output_list)

        if i > 10:
            break
        i += 1

# Confusion Matrix 
print(confusion_matrix(true, pred))

cm = metrics.confusion_matrix(true,pred)

plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

#Classification report
report = classification_report(true, pred, FISH_CLASSES)
print(report)

print("ACCURACY = ")
print(correct/total)

# Writing output in a csv file

with open('./output.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(prediction_output_list)

# -------------------------------------------------------------------------------
# def predict_one(f) :
#     from os import listdir
#     from os.path import isfile, join
#     import csv
#     import math

#     print("Evaluating at ......")
#     print(f)
#     img = cv2.imread(f)  
#     img = cv2.resize(img, (200, 125))
#     imlist = np.array([img])
#     print("Neural Net Prediction:")
#     cnn_prediction = model.predict_proba(imlist)
#     print(cnn_prediction)
#     csv_output_list = []
#     csv_output_list.append(f)
#     for elem in cnn_prediction:
#         for value in elem:
#             value = value.item()
#             value = value * 100
#             value = int(math.floor(value))
#             csv_output_list.append(value)
#     print("CSV Output List Formatted:")
#     print(csv_output_list)

# predict_one("/home/miten/Downloads/img_02758.jpg")
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:04:59 2020

@author: Karthi
"""

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np

(X_train, y_train) , (X_test, y_test) = cifar10.load_data()

label = ["Planes","Cars","Birds","Cats","Deer","Dogs","Frogs","Horses","Ships","Trucks"]
index = 100
plt.imshow(X_train[index])
print(y_train[index])
#ship - 8
plt.imshow(X_train[2000])
print(y_train[2000])
#horse - 7
plt.imshow(X_train[1000])
print(y_train[1000])
#truck - 9

def indices(index):
    for x in range(len(label)+1):
        if int(y_train[index]) == x:
            return label[x]
    
print(indices(1000))

width = 4
height = 4

fig, axes = plt.subplots(width,height, figsize = (25, 25))
axes = axes.ravel()


for i in np.arange(0, width*height):
    index = np.random.randint(0, len(X_train)) # pick a random number
    axes[i].imshow(X_train[index])
    axes[i].set_title(indices(index))
    axes[i].axis('off')
    
plt.subplots_adjust(hspace = 0.4)

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

###Data Preparation
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
number_cat = 10

#One hot encoding the labels
y_train = keras.utils.to_categorical(y_train, number_cat)
y_test = keras.utils.to_categorical(y_test, number_cat)

#Normalizing the image
X_train = X_train/255
X_test = X_test/255

weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same',activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same',activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same',activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same',activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same',activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same',activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())

model.add(Dense(units=10, activation='softmax'))
 
model.summary()

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(X_train)

fig = plt.figure(figsize = (20,2))
for x_batch in dataget_train.flow(X_train_sample, batch_size = n):
     for i in range(0,n):
            ax = fig.add_subplot(1, n, i+1)
            ax.imshow(toimage(x_batch[i]))
     fig.suptitle('Augmented images (rotated 90 degrees)')
     plt.show()
     break;
 
#training
batch_size = 64

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=X_train.shape[0] // batch_size,epochs=125,\
                    verbose=1,callbacks=[LearningRateScheduler(lr_schedule)])


#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')


#testing
scores = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
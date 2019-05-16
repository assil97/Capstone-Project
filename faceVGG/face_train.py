#training of the model
import numpy as np
import cv2
import os
import random
from sklearn.model_selection import train_test_split 

from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.models import Model, Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from keras.preprocessing import image
import matplotlib.pyplot as plt

file='/Users/syedshakeeb/Desktop/dataSet/'

Person=['shakeeb','billgates']

leraning_rate=0.01
epoch=30

X=[]
y=[]

def create_training_data():
    image=[]
    for person in Person:
        path=os.path.join(file,person)
        class_num=Person.index(person)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img))
            new_array=cv2.resize(img_array ,(224,224))
            image.append([new_array,class_num])
    
    return image

image=create_training_data()

random.shuffle(image)

for features,labels in image:
    X.append(features)
    y.append(labels)

X=np.array(X)
y=np.array(y)

train_face,test_face,train_target,test_target=train_test_split(X,y,test_size=0.3,random_state=1)

def dataset(train_files,test_files):
    train_files=np_utils.to_categorical(train_files,2)
    test_files=np_utils.to_categorical(test_files,2)
    return train_files,test_files

train_target,test_target=dataset(train_target,test_target)

train_face=train_face.astype('float32')/255
test_face=test_face.astype('float32')/255
train_target=train_target.astype('float32')/255
test_target=test_target.astype('float32')/255

print(np.shape(train_face))
print(np.shape(test_face))
print(np.shape(train_target))
print(np.shape(test_target))

model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer=ModelCheckpoint(filepath='/Users/syedshakeeb/Desktop/savedmodels/weights.best.vgg19face.hdf5',verbose=1,save_best_only=True)

model.fit(train_face,train_target,validation_data=(test_face,test_target),epochs=epoch, batch_size=20, callbacks=[checkpointer], verbose=1)

model.save('Users/syedshakeeb/Desktop/savedmodels/vgg19.MODEL')


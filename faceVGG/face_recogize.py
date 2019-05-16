import numpy as np
import os 
import datetime
import csv
import cv2

from PIL import Image

from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.preprocessing import image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input 

os.environ['KMP_DUPLICATE_LIB_OK']='True'

file_path='/Users/syedshakeeb/Desktop/dataSet'

Person=['shakeeb','billgates']

def preprocessing_image(img):

    img_array = load_img(img, target_size=(224,224))
    img = img_to_array(img_array)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    return img

def vggface():

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

    #loading the predefined weight
    model.load_weights('/Users/syedshakeeb/Desktop/savedmodels/vgg_face_weights.h5')

    vgg_face_descriptor=Model(inputs=model.layers[0].input, output=model.layers[-2].output)

    return vgg_face_descriptor

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

model=vggface()
Persons=dict()

for person in Person:
    img_path=os.path.join(file_path,person)
    for img in os.listdir(img_path):
        img=os.path.join(img_path,img)
        person_name=person
        Persons[person_name]=model.predict(preprocessing_image(img))[0,:]


for i in Persons:
    print(Persons[i])

#recognizing from the live camera
cam=cv2.VideoCapture(0)
face=cv2.CascadeClassifier('/Users/syedshakeeb/Desktop/savedmodels/faces.xml')

while True:
    ret, frame=cam.read()

    faces=face.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,169,30), 2)

        detected_face=frame[y:y+h,x:x+h]
        detected_face=cv2.resize(detected_face, (224,224))

        img_pixels=image.img_to_array(detected_face)
        img_pixels=np.expand_dims(img_pixels, axis=0)

        img_pixels=img_pixels/127.5
        img_pixels=img_pixels-1

        result = model.predict(img_pixels)[0,:]

        found=0
        for i in Persons:
            person_name=i
            representation=Persons[i]

            similarity = findCosineSimilarity(representation, result)

            if similarity < 0.30:
                cv2.putText(frame, person_name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                row=[person_name, 'Present', str(datetime.datetime.now())]

                with open('/Users/syedshakeeb/Desktop/faceVGG/attendance.csv', mode='w') as writeFile:
                    writer = csv.writer(writeFile, delimiter='|', quotechar='"', quoting=csv.QUOTE_ALL,skipinitialspace=False)
                    writer.writerow(row)

                writeFile.close()
                
                found = 1
                break
        
        if(found==0):
            cv2.putText(frame, 'unknown', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
    cv2.imshow('face',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cam.release()
        break

cam.release()
cv2.destroyAllWindows()


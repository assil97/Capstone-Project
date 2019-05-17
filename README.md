# Capstone-Project

Capstone project is about the Attenance using Face Recognition where the face can be recogized from taking the image in real time.

This project is about taking attendance using your face as a identity with the help of Convolutional Neural Network instead of the biometric system.

I created a Attendance Face-Recognition application in Python using Keras and OpenCV where the user have to place the face in front of the camera using VGG Face as a classifier for face recognition and attendance will be added accordingly.

In this project, I will be using VGG-Face Model for face recognition which was built under VGG16 architecture which was developed by University of Oxford using Keras.

I have used a predefined load of weights of the model which was already trained by 14 million of images scoring a accuracy of 92.7%.

Training the model with our own datasets is done by predicting the model where the dictinary is created which matches the target name to the matrix vector of image and using cosine similarity to match the matrix vector of the new image to the matrix vector of the preprocessed image when predicting the new image.

Files:
  1. dataset is the file which contains the images of the user
  2. faceVGG is the file which contains the programs to run the application 
  
    1. generate.py is the generation of the new images
    2. face_recogize.py is the image preprocessing and recognition of the face in real-time camera
    3. vgg.py is the architecture of the neural network
    
 Please read Capstone.pdf file for the documentation of the project.

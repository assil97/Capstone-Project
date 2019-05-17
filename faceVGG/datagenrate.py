import cv2

cam=cv2.VideoCapture(0)
#face xml file to recognize the pattern of the face
face=detector=cv2.CascadeClassifier('/Users/syedshakeeb/Desktop/savedmodels/faces.xml')

samplenum=0

#taking the name of the user
user=input("Please enter your name:\n")

while(cam.isOpened()): #check if the camera is opened 
    ret,frame=cam.read()
    frame=cv2.flip(frame, 1)
    faces=face.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:
        #creating the rectangle to detect the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #incrementing sample number 
        samplenum=samplenum+1
        #saving the captured face in the dataset folder
        cv2.imwrite('/Users/syedshakeeb/Desktop/dataSet/' + user + str(samplenum) + '.jpg', frame[y:y+h,x+w])

    cv2.imshow('img',frame)

    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        cam.release()
    #break if the sample number is more than 25
    elif samplenum>=25:
        break

cam.release()
cv2.destroyAllWindows()

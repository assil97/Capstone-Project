import cv2

cam=cv2.VideoCapture(0)
face=detector=cv2.CascadeClassifier('/Users/syedshakeeb/Desktop/savedmodels/faces.xml')

samplenum=0

user=input("Please enter your name:\n")

while(cam.isOpened()):
    ret,frame=cam.read()
    frame=cv2.flip(frame, 1)
    faces=face.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        samplenum=samplenum+1
        cv2.imwrite('/Users/syedshakeeb/Desktop/dataSet/firas/' + user + str(samplenum) + '.jpg', frame[y:y+h,x+w])

    cv2.imshow('img',frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        cam.release()
    elif samplenum>=25:
        break

cam.release()
cv2.destroyAllWindows()

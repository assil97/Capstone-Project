import cv2
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('/Users/syedshakeeb/Downloads/Attendance-using-face/faces.xml')

Id=input('enter your name:\n')

sampleNum=0
while(cam.isOpened()): #check 
    ret, img = cam.read()
    img=cv2.flip(img,1)
    faces = detector.detectMultiScale(img,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("/Users/syedshakeeb/Desktop/dataSet/shakeeb/" + Id + str(sampleNum) + ".jpg", img[y:y+h,x:x+w])
        
    cv2.imshow('img',img)
            
    #wait for 10 miliseconds 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cam.release()
    # break if the sample number is morethan 50
    elif sampleNum>=25:
        break
        

cam.release()
cv2.destroyAllWindows()
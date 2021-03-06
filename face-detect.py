#Reading video from webcam(frame by frame)
import cv2

cap=cv2.VideoCapture(0) #Capture device

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cap.read()    #ret is False when image has not been captured 
    #gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if(ret==False):
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Video Frame",frame)
    #Stop only if user presses 'q':-
    key_pressed=cv2.waitKey(1) & 0xFF
    if(key_pressed==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
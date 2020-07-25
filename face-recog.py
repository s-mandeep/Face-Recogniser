import numpy as np
import cv2
import os

##KNN Algorithm
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(train,test,k=5):
    vals=[]
    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=dist(test,ix)
        vals.append((d,iy))
        
    vals=sorted(vals,key=lambda x: x[0])
    vals=vals[:k]
    vals=np.array(vals)
    freq=np.unique(vals[:,-1],return_counts=True)
    index=freq[1].argmax()
    pred=freq[0][index]
    return(pred)

#Initialising cam.
cap=cv2.VideoCapture(0)

#Face detection
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
dataset_path='./data/'
face_data=[]
label=[]
class_id=0  #Lables for the given file
names={}

#Data Prep.

for fx in os.listdir(dataset_path):
    if(fx.endswith('.npy')):
        names[class_id]=fx[:-4]         #Mandeep.npy -> Mandeep
        print("Loaded "+fx)
        data= np.load(dataset_path+fx)
        face_data.append(data)
        #Create labels for a class
        target=class_id*np.ones((data.shape[0],1))
        class_id+=1
        label.append(target)
face_dataset=np.concatenate(face_data,axis=0)
face_label=np.concatenate(label,axis=0)
print(face_dataset.shape)
print(face_label.shape) 
trainset=np.concatenate((face_dataset,face_label),axis=1)
print(trainset.shape)


while True:
    ret,frame=cap.read()
    if(ret==False):
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h=face

        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        out=knn(trainset,face_section.flatten())
        pred_name=names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Faces",frame)
    key= cv2.waitKey(1)&0xFF
    if(key==ord('q')):
        break

cv2.release()
cv2.destroyAllWindows() 
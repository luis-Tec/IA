import numpy as np
import cv2
import pickle

# Loading the facecascader
face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

#loading the eye cascader
eye_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

# Loading the recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face-trainner.yml')

labels={}

# Loading the labels from pickle file
with open("face-labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0)
while(True):
    # Capturing the frames
    ret,frame=cap.read()

    # converting the frames to gray for the cascade to determine
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        # print(x,y,w,h)
        # this will be the coordinates of the rectangle around our face
        roi_gray=gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Recognizing-deep learning model
        id_,conf=recognizer.predict(roi_gray)
        if conf>=75 and conf<=150:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        # print("Id is ",id_)
        # print("confidence is ",conf)

        img_item="mg_img.png"
        cv2.imwrite(img_item,roi_gray)

        # Drawing a rectangle arround are face
        color=(255,0,0)
        stroke=2
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)

        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Displaying the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

# Releasing the capture
cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2 as cv
import time

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

capture = cv.VideoCapture(0)

while(capture.isOpened()):
    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', frame)

    #if cv.waitKey(1) == ord('q'):
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:

        output = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
        output[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        cv.imshow('output', output)

        draw = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv.imshow('draw',draw)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]


        '''eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)'''

    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()

import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

img = cv.imread('andreaScaled.JPG')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img, 1.3, 5)
for (x,y,w,h) in faces:
    #gray[y:y+h, x:x+w, None] = img[y:y+h, x:x+w]
    output = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    output[y:y+h, x:x+w] = img[y:y+h, x:x+w]

    draw = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]


    '''eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)'''

cv.imshow('output', output)
cv.imshow('draw',draw)
cv.waitKey(0)
cv.destroyAllWindows()

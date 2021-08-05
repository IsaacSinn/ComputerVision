import cv2 as cv
import numpy as np
import argparse
from position import position

parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, required = True)
args = parser.parse_args()

def nothing(x):
    pass

def scale_contour(contour, scale):
    moments = cv.moments(contour)
    midX = int(round(moments["m10"] / moments["m00"]))
    midY = int(round(moments["m01"] / moments["m00"]))
    mid = np.array([midX, midY])
    contour = contour - mid
    contour = (contour * scale).astype(np.int32)
    contour = contour + mid
    return contour

SubwayTop = cv.imread(args.file)

cv.namedWindow("HSV")
cv.createTrackbar("LowH", "HSV", 0, 255, nothing)
cv.createTrackbar("LowS", "HSV", 0, 255, nothing)
cv.createTrackbar("LowV", "HSV", 0, 255, nothing)
cv.createTrackbar("HighH", "HSV", 0, 255, nothing)
cv.createTrackbar("HighS", "HSV", 0, 255, nothing)
cv.createTrackbar("HighV", "HSV", 0, 255, nothing)

while True:

    LowH = cv.getTrackbarPos("LowH", "HSV")
    LowS = cv.getTrackbarPos("LowS", "HSV")
    LowV = cv.getTrackbarPos("LowV", "HSV")
    HighH = cv.getTrackbarPos("HighH", "HSV")
    HighS = cv.getTrackbarPos("HighS", "HSV")
    HighV = cv.getTrackbarPos("HighV", "HSV")

    LowerBound = np.array([LowH, LowS, LowV])
    UpperBound = np.array([HighH, HighS, HighV])

    #LowerBound = np.array([0, 127, 0])
    #UpperBound = np.array([255, 255, 255])

    SubwayTopGrey = cv.cvtColor(SubwayTop, cv.COLOR_BGR2GRAY)
    SubwayTopHSV = cv.cvtColor(SubwayTop, cv.COLOR_BGR2HSV)
    SubwayTopBlur = cv.blur(SubwayTopGrey, (3,3))

    mask = cv.inRange(SubwayTop, LowerBound, UpperBound)
    cv.imshow("mask", mask)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('a'):
        result = SubwayTop.copy()
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print(result)

        MaxContourArea = 0
        idx = 0
        for i, contour in enumerate(contours):
            if cv.contourArea(contour) >= MaxContourArea:
                MaxContourArea = cv.contourArea(contour)
                idx = i
                contour = scale_contour(contour, 0.97)
                rect = cv.minAreaRect(contour)
                bbox = np.int0(cv.boxPoints(rect))

        cv.drawContours(result,[bbox],0,(255,255,255),-1)
        out = np.zeros_like(SubwayTop)
        out[mask == 255] = SubwayTop[mask == 255]
        cv.imshow("out", out)
        cv.imshow("result", result)





cv.destroyAllWindows()

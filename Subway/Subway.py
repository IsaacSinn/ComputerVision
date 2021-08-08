import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type = str, required = True)
parser.add_argument("--test", type = bool, required = True)
args = parser.parse_args()

# Constants
faces = [None for i in range(5)]
for i in range(5):
    faces[i] = cv.imread(f"{args.folder}\{i+1}.png") # Back slash between

LowerBoundSubway = np.array([0, 127, 0])
UpperBoundSubway = np.array([255, 255, 255])

trackbar = {
            "HSV": ["LowH", "LowS", "LowV", "HighH", "HighS", "HighV"],
}

# Functions
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

def unwarp(img, src, dst):
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR)
    return warped, M

# identify color and amount
def identify(idx):
    h, w = faces[idx].shape[:2]
    # Contour of Subway
    ContourImage = faces[idx].copy()
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    MaxContourArea = 0
    for i, contour in enumerate(contours):
        if cv.contourArea(contour) >= MaxContourArea:
            MaxContourArea = cv.contourArea(contour)
            #contour = scale_contour(contour, 0.95)
            contour = cv.minAreaRect(contour)
            bbox = np.int0(cv.boxPoints(contour))

    cv.drawContours(ContourImage,[bbox],0,(0,0,255),3)
    cv.imshow("Contour Image", ContourImage)

    # Perspective Fix Image
    bbox = np.float32(bbox)
    dst = np.float32([[0,0], [w,0], [w,h], [0,h]])

    warped, M = unwarp(faces[idx], bbox, dst)
    cv.imshow("warped", warped)

    # Show Masked Image
    #out = np.zeros_like(faces[idx])
    #out[mask == 255] = faces[idx][mask == 255]
    #cv.imshow("out", out)

# Create track bar
if args.test:
    for window in trackbar.keys():
        cv.namedWindow(window)
        for name in trackbar[window]:
            cv.createTrackbar(name, window, 0, 255, nothing)

    while True:

        LowH = cv.getTrackbarPos("LowH", "HSV")
        LowS = cv.getTrackbarPos("LowS", "HSV")
        LowV = cv.getTrackbarPos("LowV", "HSV")
        HighH = cv.getTrackbarPos("HighH", "HSV")
        HighS = cv.getTrackbarPos("HighS", "HSV")
        HighV = cv.getTrackbarPos("HighV", "HSV")

        LowerBound = np.array([LowH, LowS, LowV])
        UpperBound = np.array([HighH, HighS, HighV])

        # Mask of Subway
        mask = cv.inRange(faces[1], LowerBound, UpperBound)
        cv.imshow("mask", mask)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('a'):
            identify(faces[1])




cv.destroyAllWindows()

import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import imutils
import sys

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type = str, required = True)
parser.add_argument("--test", type = str2bool, required = False, default = False)
args = parser.parse_args()
np.set_printoptions(threshold = sys.maxsize)

# Constants
faces = [None for i in range(5)]
hsv = [None for i in range(5)]
for i in range(5):
    faces[i] = cv.imread(f"{args.folder}\J{i+1}.png") # Back slash between
    hsv[i] = cv.cvtColor(faces[i], cv.COLOR_BGR2HSV)

LowerBoundSubway = np.array([60, 0, 0])
UpperBoundSubway = np.array([115, 75, 255])

window_width = 900
window_height = 850

kernel = np.ones((10,10), np.uint8)

trackbar = {
            "HSV": ["LowH", "LowS", "LowV", "HighH", "HighS", "HighV"],
}

# Functions
def nothing(x):
    pass

def centroid(contour):
    moments = cv.moments(contour)
    midX = int(round(moments["m10"] / moments["m00"]))
    midY = int(round(moments["m01"] / moments["m00"]))
    mid = np.array([midX, midY])
    return mid

def scale_contour(contour, scale):
    moments = cv.moments(contour)
    if moments["m00"] == 0:
        pass
    else:
        mid = centroid(contour)
        contour = contour - mid
        contour = (contour * scale).astype(np.int32)
        contour = contour + mid
        return contour

def unwarp(img, src, dst, w, h):
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR)
    return warped, M

def check_boundaries(value, tolerance, boundary, upper_or_lower):
    if(value + tolerance > boundary) and upper_or_lower == 1:
        value = boundary
    elif (value - tolerance < 0) and upper_or_lower == 0:
        value = 0
    else:
        if upper_or_lower == 1:
            value = value + tolerance
        else:
            value = value - tolerance
    return value

def crop_contour(idx):
    h, w = faces[idx].shape[:2]
    if w > h*1.2:
        w, h = 1000, 500
    else:
        w, h = 500, 500
    # Contour of Subway
    ContourImage = faces[idx].copy()
    mask = cv.inRange(hsv[idx], LowerBoundSubway, UpperBoundSubway)
    #cv.imshow("mask", mask)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    MaxContourArea = 0
    for i, contour in enumerate(contours):
        if cv.contourArea(contour) >= MaxContourArea:
            MaxContourArea = cv.contourArea(contour)
    try:
        contour = scale_contour(contour, 0.95)
        contour = cv.minAreaRect(contour)
        bbox = np.int0(cv.boxPoints(contour))
    except:
        print("No Contour Found")

    cv.drawContours(ContourImage,[bbox],0,(0,0,255),3)
    #cv.imshow("Contour Image", ContourImage)

    # Perspective Fix Image
    bbox = np.float32(bbox)
    if bbox[0][1] > bbox[2][1]:
        dst = np.float32([[0,h], [0,0], [w,0], [w,h]])
    else:
        dst = np.float32([[0,0], [w,0], [w,h], [0,h]])
    warped, _ = unwarp(faces[idx], bbox, dst, w, h)
    #cv.imshow("warped", warped)

    # Mask of tape
    warped_hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
    warped_mask = cv.inRange(warped_hsv, LowerBoundSubway, UpperBoundSubway)
    warped_mask = cv.morphologyEx(warped_mask, cv.MORPH_CLOSE, kernel)
    warped_mask_inv = cv.bitwise_not(warped_mask)
    #cv.imshow("warped_mask", warped_mask_inv)

    # Show Masked Image of tape
    masked_image = cv.bitwise_and(warped, warped, mask = warped_mask_inv)
    #cv.imshow("masked_image", masked_image)

    # Select colors of tape
    tapes = {}
    centroid_X = []
    centroid_Y = []
    tape_color = []
    no_color = 0
    tape_contours, _ = cv.findContours(warped_mask_inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, tape_contour in enumerate(tape_contours):
        tape_contour = scale_contour(tape_contour, 0.85)
        tape_contour_min = cv.minAreaRect(tape_contour)
        tape_bbox = np.int0(cv.boxPoints(tape_contour_min))

        tape_mask = np.zeros(warped.shape[:2], np.uint8)
        cv.drawContours(tape_mask, [tape_bbox], 0, (255,255,255), -1)
        tape_mean = list(map(int, cv.mean(warped_hsv, mask = tape_mask)[:3]))
        tape_mean = np.asarray(tape_mean)

        # show tape bbox -->
        # cv.drawContours(warped, [tape_bbox], 0, (0,0,255), 1)
        # cv.imshow("warped", warped)

        mid = centroid(tape_contour)
        centroid_X.append(mid[0])
        centroid_Y.append(mid[1])
        tape_color.append(tape_mean)
        no_color += 1


    # n, e, s, w
    if no_color == 4:
        tapes["e"] = tape_color[centroid_X.index(max(centroid_X))]
        tapes["w"] = tape_color[centroid_X.index(min(centroid_X))]
        tapes["s"] = tape_color[centroid_Y.index(max(centroid_Y))]
        tapes["n"] = tape_color[centroid_Y.index(min(centroid_Y))]
        return tapes, warped, no_color
    else:
        tapes["n"] = tape_color[centroid_Y.index(min(centroid_Y))]
        return tapes, warped, no_color


# Create track bar
if args.test:
    for window in trackbar.keys():
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        cv.resizeWindow(window, window_width, window_height)
        for name in trackbar[window]:
            limit = 255
            if name == "HighH" or name == "LowH":
                limit = 180
            cv.createTrackbar(name, window, 0, limit, nothing)

    while True:

        # LowH = cv.getTrackbarPos("LowH", "HSV")
        # LowS = cv.getTrackbarPos("LowS", "HSV")
        # LowV = cv.getTrackbarPos("LowV", "HSV")
        # HighH = cv.getTrackbarPos("HighH", "HSV")
        # HighS = cv.getTrackbarPos("HighS", "HSV")
        # HighV = cv.getTrackbarPos("HighV", "HSV")
        #
        # LowerBoundSubway = np.array([LowH, LowS, LowV])
        # UpperBoundSubway = np.array([HighH, HighS, HighV])

        # Mask of Subway
        mask = cv.inRange(hsv[0], LowerBoundSubway, UpperBoundSubway)
        cv.imshow("HSV", mask)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            tape, warped, no_color = crop_contour(3)
            for tape_color in tape.values():
                LowH = check_boundaries(tape_color[0], 20, 180, 0)
                LowS = check_boundaries(tape_color[1], 10, 255, 0)
                LowV = check_boundaries(tape_color[2], 10, 255, 0)
                HighH = check_boundaries(tape_color[0], 20, 180, 1)
                HighS = check_boundaries(tape_color[1], 10, 255, 1)
                HighV = check_boundaries(tape_color[2], 10, 255, 1)

                lower_bound_tape = np.array([LowH, LowS, LowV])
                upper_bound_tape = np.array([HighH, HighS, HighV])

                # Visualize color patches of subway top -->
                # tape_color_patch = np.ones(shape = (500,500,3), dtype = np.uint8)*np.uint8(tape_color)
                # tape_color_patch = cv.cvtColor(tape_color_patch, cv.COLOR_HSV2BGR)
                # cv.imshow(f"{tape}", tape_color_patch)


else:
    tapes = [None for i in range(5)]
    warped = [None for i in range(5)]
    no_color = [None for i in range(5)]
    idx = None
    for i in range(5):
        tapes[i], warped[i], no_color[i] = crop_contour(i)
        if no_color[i] == 4:
            idx = i

    output = np.zeros((1500,2000,3), dtype = np.uint8)
    output[500:1000, 500:1500,:] = warped[idx]

    for tape_position in tapes[idx].keys():
        tape_color = tapes[idx][tape_position]

        LowH = check_boundaries(tape_color[0], 10, 180, 0)
        LowS = check_boundaries(tape_color[1], 15, 255, 0)
        LowV = check_boundaries(tape_color[2], 15, 255, 0)
        HighH = check_boundaries(tape_color[0], 10, 180, 1)
        HighS = check_boundaries(tape_color[1], 15, 255, 1)
        HighV = check_boundaries(tape_color[2], 15, 255, 1)

        lower_bound_tape = np.array([LowH, LowS, LowV])
        upper_bound_tape = np.array([HighH, HighS, HighV])

        # Visualize color patches of subway top -->
        tape_color_patch = np.ones(shape = (500,500,3), dtype = np.uint8)*np.uint8(tape_color)
        tape_color_patch = cv.cvtColor(tape_color_patch, cv.COLOR_HSV2BGR)
        cv.imshow(f"{tape_color}", tape_color_patch)

        for i, tape in enumerate(tapes):
            if i == idx:
                pass
            else:
                in_range = cv.inRange(tape["n"], lower_bound_tape, upper_bound_tape)
                if in_range[0] == 255:
                    if tape_position == "n":
                        rotated_warp = imutils.rotate(warped[i], 180)
                        output[0:500, 500:1500,:] = rotated_warp
                    elif tape_position == "s":
                        output[1000:1500, 500:1500, :] = warped[i]
                    elif tape_position == "e":
                        rotated_warp = imutils.rotate_bound(warped[i], -90)
                        output[500:1000, 1500:2000, :] = rotated_warp
                    elif tape_position == "w":
                        rotated_warp = imutils.rotate_bound(warped[i], 90)
                        output[500:1000, 0:500, :] = rotated_warp
                    else:
                        pass



output = plt.imshow(output)
plt.show()
cv.destroyAllWindows()

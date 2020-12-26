import cv2
import numpy as np


def findContourCenter(img):
    contours, hierarchy = cv2.findContours(
        np.uint8(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=cv2.contourArea, reverse=True)
    M = cv2.moments(contours[0])
    return [int(M['m01']/M['m00']), int(M['m10']/M['m00'])], contours[0]

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import math


def findContourCenter(img):
    contours, hierarchy = cv2.findContours(
        np.uint8(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=cv2.contourArea, reverse=True)
    M = cv2.moments(contours[0])
    return [int(M['m01']/M['m00']), int(M['m10']/M['m00'])], contours[0]


def findCollision(folder, is_debug):
    # folder = './dataset/train/toothpaste_box/21/'
    before = 0
    after = 1
    row = 0
    col = 1
    mask_img_files = sorted(glob.glob(f'{folder}mask/*.png'))
    mask_img = np.array([plt.imread(mask_img_files[0]),
                         plt.imread(mask_img_files[-1])])

    center_before, cnt = findContourCenter(mask_img[before])
    center_after, cnt = findContourCenter(mask_img[after])

    distance = math.sqrt((center_after[row] - center_before[row])
                         ** 2 + (center_after[col] - center_before[col])**2)
    angle = math.atan2((center_after[row] - 220),
                       (center_after[col] - 220))
    if is_debug:
        print(folder)
        print(f'angle = {angle}', f'distance = {distance}')
        inter_img = np.array([mask_img[before], mask_img[after],
                              np.zeros(mask_img[after].shape)])
        inter_img = np.moveaxis(inter_img, 0, -1)
        inter_img = cv2.UMat(inter_img)
        inter_img = cv2.UMat.get(inter_img)
        cv2.drawContours(inter_img, [cnt], -1, (0, 0, 255), 2)
        plt.imshow(inter_img)
        plt.plot([center_before[1], center_after[1]],
                 [center_before[0], center_after[0]])
        plt.plot(center_after[1],
                 center_after[0], marker='o')
        plt.show()

    return angle, distance

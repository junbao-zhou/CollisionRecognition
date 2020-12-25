import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import math

path = './dataset/train'

labels = ['061_foam_brick', 'green_basketball', 'salt_cylinder',
          'shiny_toy_gun', 'stanley_screwdriver', 'strawberry',
          'toothpaste_box', 'toy_elephant', 'whiteboard_spray',
          'yellow_block']
label = 0

folders = glob.glob(f'{path}/{labels[label]}/*/')
folder = random.choice(folders)
# print(folder)
mask_img_files = glob.glob(f'{folder}mask/*.png')
mask_img_before = plt.imread(mask_img_files[0])
# print(mask_img_before.shape)
mask_img_after = plt.imread(mask_img_files[-1])

cols = np.array([np.arange(mask_img_before.shape[0])])
cols = cols.repeat(mask_img_before.shape[1], axis=0)
rows = cols.T

before_row = np.sum(rows * mask_img_before) / np.sum(mask_img_before)
before_col = np.sum(cols * mask_img_before) / np.sum(mask_img_before)

after_row = np.sum(rows * mask_img_after) / np.sum(mask_img_after)
after_col = np.sum(cols * mask_img_after) / np.sum(mask_img_after)

print((before_row, before_col))
print((after_row, after_col))

angle = math.atan2((after_row - before_row), (after_col - before_col))
angle = angle / math.pi * 180

print(angle)

plt.subplot(1, 2, 1)
plt.imshow(mask_img_before, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(mask_img_after, cmap='gray')
plt.show()

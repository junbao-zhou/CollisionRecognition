import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import math
import os

path = './dataset/train'

labels = os.listdir(path)
label = 0

folders = glob.glob(f'{path}/{labels[label]}/*/')
folder = random.choice(folders)
print(folder)
mask_img_files = glob.glob(f'{folder}mask/*.png')
mask_img_before = plt.imread(mask_img_files[0])
# print(mask_img_before.shape)
mask_img_after = plt.imread(mask_img_files[-1])

cols = np.array([np.arange(mask_img_before.shape[0])])
cols = cols.repeat(mask_img_before.shape[1], axis=0)
rows = cols.T

before_pos = np.array([np.sum(rows * mask_img_before),
                       np.sum(cols * mask_img_before)]) / np.sum(mask_img_before)

after_pos = np.array([np.sum(rows * mask_img_after),
                      np.sum(cols * mask_img_after)]) / np.sum(mask_img_after)

print(before_pos)
print(after_pos)

angle = math.atan2((after_pos[0] - before_pos[0]),
                   (after_pos[1] - before_pos[1]))
angle = angle / math.pi * 180

distance = math.sqrt((after_pos[0] - before_pos[0])
                     ** 2 + (after_pos[1] - before_pos[1])**2)

print(angle, distance)

with open(f'{folder}position.npy', 'wb') as f:
    np.save(f, before_pos)
    np.save(f, after_pos)
    np.save(f, angle)
    np.save(f, distance)

with open(f'{folder}position.npy', 'rb') as f:
    print(np.load(f))
    print(np.load(f))
    print(np.load(f))
    print(np.load(f))

inter_img = np.array([mask_img_before, mask_img_after,
                      np.zeros(mask_img_after.shape)])
inter_img = np.moveaxis(inter_img, 0, -1)
plt.imshow(inter_img)
plt.plot([before_pos[1], after_pos[1]], [before_pos[0], after_pos[0]])
plt.plot(after_pos[1], after_pos[0], marker='o')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import math
import os

path = './dataset/train'

labels = os.listdir(path)
label = 0

# define a length in need
sp = 68

folders = glob.glob(f'{path}/{labels[label]}/*')
folder = random.choice(folders)
print(folder)
mask_img_files = glob.glob(f'{folder}/mask/*.png')
rgb_img_files = glob.glob(f'{folder}/rgb/*.jpg')
# print(rgb_img_files)
mask_img_before = plt.imread(mask_img_files[0])
rgb_img_before = plt.imread(rgb_img_files[0])
# print(mask_img_before.shape)

mask_img_after = plt.imread(mask_img_files[-1])
rgb_img_after = plt.imread(rgb_img_files[-1])

cols = np.array([np.arange(mask_img_before.shape[0])])
cols = cols.repeat(mask_img_before.shape[1], axis=0)
rows = cols.T

before_pos = np.array([np.sum(rows * mask_img_before),
                       np.sum(cols * mask_img_before)]) / np.sum(mask_img_before)

after_pos = np.array([np.sum(rows * mask_img_after),
                      np.sum(cols * mask_img_after)]) / np.sum(mask_img_after)

print(before_pos)
print(after_pos)

before_pos_rgb = np.array(
    [before_pos[0] + 20, before_pos[1] + 100], dtype=np.float64)
after_pos_rgb = np.array(
    [after_pos[0] + 20, after_pos[1]+100], dtype=np.float64)


row_before_rgb = np.array([before_pos_rgb[0] - sp, before_pos_rgb[0] - sp, before_pos_rgb[0] +
                           sp, before_pos_rgb[0] + sp, before_pos_rgb[0] - sp], dtype=np.float64)
col_before_rgb = np.array([before_pos_rgb[1] + sp, before_pos_rgb[1] - sp, before_pos_rgb[1] -
                           sp, before_pos_rgb[1] + sp, before_pos_rgb[1] + sp], dtype=np.float64)

row_after_rgb = np.array([after_pos_rgb[0] - sp, after_pos_rgb[0] - sp, after_pos_rgb[0] +
                          sp, after_pos_rgb[0] + sp, after_pos_rgb[0] - sp], dtype=np.float64)
col_after_rgb = np.array([after_pos_rgb[1] + sp, after_pos_rgb[1] - sp, after_pos_rgb[1] -
                          sp, after_pos_rgb[1] + sp, after_pos_rgb[1] + sp], dtype=np.float64)


x = before_pos_rgb[0]
y = before_pos_rgb[1]

plt.imshow(rgb_img_before)
plt.plot(col_before_rgb, row_before_rgb, color='r')
plt.scatter(y, x, color='b')
plt.show()

x_1 = after_pos_rgb[0]
y_1 = after_pos_rgb[1]

plt.imshow(rgb_img_after)
plt.plot(col_after_rgb, row_after_rgb, color='k')
plt.scatter(y_1, x_1, color='g')
plt.show()


'''
# 得到所需矩形四个顶点
row_before = np.array([before_pos[0] - sp, before_pos[0] - sp, before_pos[0] + sp, before_pos[0] + sp, before_pos[0] - sp], dtype=np.float64)
col_before = np.array([before_pos[1] + sp, before_pos[1] - sp, before_pos[1] - sp, before_pos[1] + sp, before_pos[1] + sp], dtype=np.float64)

row_after = np.array([after_pos[0] - sp, after_pos[0] - sp, after_pos[0] + sp, after_pos[0] + sp, after_pos[0] - sp], dtype=np.float64)
col_after = np.array([after_pos[1] + sp, after_pos[1] - sp, after_pos[1] - sp, after_pos[1] + sp, after_pos[1] + sp], dtype=np.float64)

# 通过以上4个顶点绘制 bounding-box

plt.plot(col_before, row_before, color='r')
plt.scatter(col_before,row_before, color='b')
plt.plot(col_after, row_after, color='k')
plt.scatter(col_after,row_after, color='g')


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
'''

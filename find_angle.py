import cv2


class Direction(Enum):
    No = 0
    Up = 1
    RightUp = 2
    Right = 3
    RightDown = 4
    Down = 5
    LeftDown = 6
    Left = 7
    LeftUp = 8


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
    mask_img_files = glob.glob(f'{folder}mask/*.png')
    mask_img = np.array([plt.imread(mask_img_files[0]),
                         plt.imread(mask_img_files[-1])])

    cols = np.array([np.arange(mask_img.shape[1])])
    cols = cols.repeat(mask_img.shape[2], axis=0)
    rows = cols.T

    pos = np.einsum('chw,thw->tchw', np.array([rows, cols]), mask_img)
    # centers = np.einsum(
    #     'chw,thw->tc', np.array([rows, cols]), mask_img) / np.einsum('thw->t', mask_img).reshape(2, 1)

    after_min_row = np.min(
        pos[after][row] + 1000 * (1 - mask_img[after]))
    after_max_row = np.max(pos[after][row])
    after_min_col = np.min(
        pos[after][col] + 1000 * (1 - mask_img[after]))
    after_max_col = np.max(pos[after][col])

    center_before, cnt = findContourCenter(mask_img[before])
    center_after, cnt = findContourCenter(mask_img[after])

    # threshold = 30

    # if after_min_row < threshold:
    #     if after_min_col < threshold:
    #         direction = Direction.LeftUp
    #     elif mask_img[before].shape[0] - after_max_col < threshold:
    #         direction = Direction.RightUp
    #     else:
    #         direction = Direction.Up
    # elif mask_img[after].shape[0] - after_max_row < threshold:
    #     if after_min_col < threshold:
    #         direction = Direction.LeftDown
    #     elif mask_img[before].shape[0] - after_max_col < threshold:
    #         direction = Direction.RightDown
    #     else:
    #         direction = Direction.Down
    # else:
    #     if after_min_col < threshold:
    #         direction = Direction.Left
    #     elif mask_img[before].shape[0] - after_max_col < threshold:
    #         direction = Direction.Right
    #     else:
    #         direction = Direction.No

    distance = math.sqrt((center_after[row] - center_before[row])
                         ** 2 + (center_after[col] - center_before[col])**2)
    angle = math.atan2((center_after[row] - 220),
                       (center_after[col] - 220))

    # if distance < 2:
    #     direction = Direction.No
    if is_debug:
        print(folder)
        # print(f'min_row = {after_min_row}')
        # print(f'max_row = {after_max_row}')
        # print(f'min_col = {after_min_col}')
        # print(f'max_col = {after_max_col}')
        print(angle, distance)
        # print(direction)
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
